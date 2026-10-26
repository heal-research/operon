// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "algorithms/budget_ledger.hpp"
#include "checkpoint_helpers.hpp"
#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "run_signature.hpp"

// The evaluation budget limits committed work, and a generation is committed or discarded as a whole. What is
// committed (parents, fitness, ranks, worker RNG states, counters) must therefore be the same for every thread count,
// also when the budget or the offspring-selection pressure binds in the middle of a generation.
namespace Operon::Test {
namespace {

    using DTable = DispatchTable<Operon::Scalar>;

    enum class Algorithm { GP, NSGA2 };

    constexpr std::size_t TrainRows = 60;
    constexpr std::size_t MaxLength = 14;
    constexpr std::size_t MaxDepth = 6;
    constexpr std::size_t Unlimited = std::numeric_limits<std::size_t>::max();

    struct Scenario {
        Algorithm Algo { Algorithm::GP };
        std::string Generator { "basic" }; // basic, os, brood, poly, capped
        bool Structural { false }; // select a zero-cost TreePropertyEvaluator
        bool LocalSearch { false };
        bool Cache { false }; // transposition cache, CacheVisibility::Committed
        std::size_t Population { 40 };
        std::size_t Pool { 40 };
        std::size_t Generations { 1000 };
        std::size_t Budget { Unlimited };
        std::size_t Pressure { 1000 }; // os: MaxSelectionPressure
        double Factor { 0 }; // os: comparison factor
        std::optional<std::size_t> Cap; // capped: MaxAttempts() of the test generator
        bool ReplaceWorst { false }; // reinserter: ReplaceWorst instead of KeepBest
        std::size_t SaturateOnPrepare {
            0
        }; // the evaluator's n-th Prepare() (1-based) saturates its counters; 0 = never
        bool Terminate { false }; // capped: the deprecated Terminate() always reports true
        std::size_t Threads { 1 };
        std::uint64_t Seed { 4242 };
    };

    // What is committed by a run, and must not depend on the thread count.
    struct State {
        std::size_t Generation {};
        std::vector<std::uint64_t> Population; // genotype, coefficients, fitness, rank and distance of every parent
        std::vector<std::uint64_t> Rngs; // state of every worker RNG
        std::vector<std::uint64_t> Best; // NSGA2::Best(), bit for bit; empty for GP
        [[nodiscard]] auto operator==(State const&) const -> bool = default;
    };

    struct Outcome {
        State Committed;
        std::size_t GenerationsCommitted {};
        std::size_t EvaluationsCommitted {};
        std::size_t BudgetOvershoot {};
        bool Discarded {}; // a generation attempt was not committed
        // Informational: what the discarded attempt executed before the workers stopped (depends on the thread count).
        BudgetAccounting Accounting;
        std::optional<OffspringProgress> Progress;
        bool StopRequested {};
        std::uint64_t Calls {}; // calls to the active evaluator, discarded work included (informational)
        std::uint64_t Attempts {}; // invocations of a capped test generator, discarded work included (informational)
        DiscardReason Reason { DiscardReason::None }; // why the discarded attempt was not committed, if one was
        RunEnd End { RunEnd::None }; // why Run() ended

        // The committed part of the outcome.
        [[nodiscard]] auto operator==(Outcome const& other) const -> bool
        {
            return Committed == other.Committed && GenerationsCommitted == other.GenerationsCommitted
                && EvaluationsCommitted == other.EvaluationsCommitted && BudgetOvershoot == other.BudgetOvershoot
                && Discarded == other.Discarded && Reason == other.Reason && StopRequested == other.StopRequested
                && End == other.End;
        }
    };

    // NSGA2::Best() is the first front of the last sort of parents plus offspring. It is observable state that is not
    // derivable from the retained parents (the reinserter may drop members of that front), so a run that mutates it
    // while discarding a generation, or builds it differently at another thread count, must be caught.
    auto StateOf(GeneticAlgorithmBase const& algo) -> State
    {
        State state { algo.Generation(), Signature(algo.Parents()), Signature(algo.WorkerRngs()), {} };
        if (auto const* nsga = dynamic_cast<Operon::NSGA2 const*>(&algo)) {
            state.Best = Signature(nsga->Best());
        }
        return state;
    }

    auto Capture(GeneticAlgorithmBase const& algo) -> Outcome
    {
        auto const& accounting = algo.BudgetReport();
        return { .Committed = StateOf(algo),
            .GenerationsCommitted = accounting.GenerationsCommitted,
            .EvaluationsCommitted = accounting.EvaluationsCommitted,
            .BudgetOvershoot = accounting.BudgetOvershoot,
            .Discarded = algo.IncompleteGeneration().has_value(),
            .Accounting = accounting,
            .Progress = algo.IncompleteGeneration(),
            .StopRequested = algo.StopRequested(),
            .Reason = algo.IncompleteGeneration() ? algo.IncompleteGeneration()->Reason : DiscardReason::None,
            .End = accounting.End };
    }

    // Offspring generator with a deterministic, slot-local rejection rule (decided by the slot's own RNG stream, so
    // the attempts of a generation are a function of the committed state alone) and a configurable attempt cap.
    // Without a transposition cache every attempt makes exactly one evaluator call.
    class CappedGenerator final : public OffspringGeneratorBase {
    public:
        using OffspringGeneratorBase::OffspringGeneratorBase;

        std::optional<std::size_t> Cap;
        bool TerminateAlways { false }; // what the deprecated Terminate() reports
        mutable std::atomic<std::uint64_t> Invocations { 0 }; // every call of operator(), whatever the outcome

        [[nodiscard]] auto Terminate(OptimizerAccounting const& /*accounting*/) const -> bool override
        {
            return TerminateAlways;
        }

        [[nodiscard]] auto MaxAttempts(std::size_t /*parents*/) const -> std::optional<std::size_t> override
        {
            return Cap;
        }

        auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
            double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
            -> std::optional<Individual> override
        {
            Invocations.fetch_add(1, std::memory_order_relaxed);
            auto res = Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, accounting);
            constexpr std::uint64_t rejectOneIn = 3;
            if (random() % rejectOneIn == 0) {
                return std::nullopt;
            }
            return std::move(res.Child);
        }
    };

    // Evaluator that, on its SaturateOnPrepare-th Prepare() call (Prepare is the first thing a generation attempt
    // does to the evaluator, after the ledger took its start value), sets a counter to the representable maximum: a
    // run that would need billions of evaluations to get there.
    class SaturatingEvaluator final : public Operon::Evaluator<DTable> {
    public:
        using Operon::Evaluator<DTable>::Evaluator;

        std::size_t SaturateOnPrepare { 0 };

        auto Prepare(Operon::Span<Individual const> pop) const -> void override
        {
            Operon::Evaluator<DTable>::Prepare(pop);
            if (++prepares_ == SaturateOnPrepare) {
                ResidualEvaluations = std::numeric_limits<std::uint64_t>::max();
            }
        }

    private:
        mutable std::size_t prepares_ { 0 };
    };

    // Everything a run needs, owned in one place so that several algorithm objects (a run and its warm resume) can
    // share the evaluator, the generator and the transposition cache.
    class Fixture {
    public:
        Fixture(Dataset& ds, Scenario const& sc)
            : scenario_(sc)
            , problem_(gsl::not_null<Dataset*>(&ds))
            , cacheRng_(sc.Seed)
        {
            problem_.SetTrainingRange(0, static_cast<int>(TrainRows));
            problem_.SetTestRange(static_cast<int>(TrainRows), static_cast<int>(2 * TrainRows));
            problem_.SetTarget("Y");
            problem_.ConfigurePrimitiveSet(PrimitiveSet::Arithmetic);
            auto const inputs = std::vector<Operon::Hash>(problem_.GetInputs().begin(), problem_.GetInputs().end());

            creator_
                = std::make_unique<Operon::BalancedTreeCreator>(&problem_.GetPrimitiveSet(), inputs, 0.0, MaxLength);
            treeInit_ = std::make_unique<Operon::UniformTreeInitializer>(creator_.get());
            treeInit_->ParameterizeDistribution(std::size_t { 1 }, MaxLength);
            treeInit_->SetMinDepth(1);
            treeInit_->SetMaxDepth(MaxDepth);
            coeffInit_.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });

            crossover_ = std::make_unique<Operon::SubtreeCrossover>(0.9, MaxDepth, MaxLength);
            onePoint_.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            changeVar_ = std::make_unique<Operon::ChangeVariableMutation>(problem_.GetInputs());
            changeFunc_ = std::make_unique<Operon::ChangeFunctionMutation>(problem_.GetPrimitiveSet());
            replaceSubtree_
                = std::make_unique<Operon::ReplaceSubtreeMutation>(creator_.get(), &coeffInit_, MaxDepth, MaxLength);
            insertSubtree_
                = std::make_unique<Operon::InsertSubtreeMutation>(creator_.get(), &coeffInit_, MaxDepth, MaxLength);
            removeSubtree_ = std::make_unique<Operon::RemoveSubtreeMutation>(creator_.get(), &coeffInit_, MaxDepth);
            for (Operon::MutatorBase* m : std::initializer_list<Operon::MutatorBase*> { &onePoint_, changeVar_.get(),
                     changeFunc_.get(), replaceSubtree_.get(), insertSubtree_.get(), removeSubtree_.get() }) {
                mutator_.Add(m, 1.0);
            }

            if (sc.Cache) {
                cache_ = std::make_unique<Operon::Zobrist>(cacheRng_, static_cast<int>(MaxLength), problem_.GetInputs(),
                    ZobristOptions { .MaxAge = 0, .Visibility = CacheVisibility::Committed });
            }

            optimizer_ = std::make_unique<Operon::LevenbergMarquardtOptimizer<DTable, Operon::OptimizerType::Eigen>>(
                &dtable_, &problem_);
            optimizer_->SetIterations(sc.LocalSearch ? 3 : 0);
            coeffOptimizer_ = std::make_unique<Operon::CoefficientOptimizer>(optimizer_.get());

            auto saturating = std::make_unique<SaturatingEvaluator>(&problem_, &dtable_);
            saturating->SaturateOnPrepare = sc.SaturateOnPrepare;
            errorEvaluator_ = std::move(saturating);
            lengthEvaluator_ = std::make_unique<Operon::TreePropertyEvaluator>(
                &problem_, [](Operon::Tree const& tree) { return static_cast<Operon::Scalar>(tree.Length()); },
                static_cast<Operon::Scalar>(MaxLength));
            multiEvaluator_ = std::make_unique<Operon::MultiEvaluator>(&problem_);
            multiEvaluator_->Add(errorEvaluator_.get());
            multiEvaluator_->Add(lengthEvaluator_.get());

            Operon::ComparisonCallback comp = Operon::SingleObjectiveComparison { 0 };
            if (sc.Algo == Algorithm::NSGA2) {
                comp = Operon::CrowdedComparison {};
            }
            fem_ = std::make_unique<Operon::TournamentSelector>(comp);
            male_ = std::make_unique<Operon::TournamentSelector>(comp);
            auto const elite = sc.Algo == Algorithm::GP ? 1U : 0U;
            if (sc.ReplaceWorst) {
                reinserter_ = std::make_unique<Operon::ReplaceWorstReinserter>(comp, elite);
            } else {
                reinserter_ = std::make_unique<Operon::KeepBestReinserter>(comp, elite);
            }

            auto* evaluator = static_cast<EvaluatorBase*>(errorEvaluator_.get());
            if (sc.Structural) {
                evaluator = lengthEvaluator_.get();
            } else if (sc.Algo == Algorithm::NSGA2) {
                evaluator = multiEvaluator_.get();
            }
            generator_ = MakeGenerator(sc, evaluator);
        }

        [[nodiscard]] auto Config(std::size_t generations, std::size_t budget) const -> GeneticAlgorithmConfig
        {
            GeneticAlgorithmConfig config;
            config.Generations = generations;
            config.Evaluations = budget;
            config.Iterations = scenario_.LocalSearch ? 3 : 0;
            config.PopulationSize = scenario_.Population;
            config.PoolSize = scenario_.Pool;
            config.Seed = scenario_.Seed;
            config.LocalSearchProbability = 1.0;
            config.LamarckianProbability = 1.0;
            config.Cache = cache_.get();
            return config;
        }

        [[nodiscard]] auto Config() const -> GeneticAlgorithmConfig
        {
            return Config(scenario_.Generations, scenario_.Budget);
        }

        // Calls f(algorithm) with a GP or an NSGA2 object (as the scenario says) built from `config`.
        template <typename F> auto With(GeneticAlgorithmConfig const& config, F&& f)
        {
            if (scenario_.Algo == Algorithm::GP) {
                Operon::GeneticProgrammingAlgorithm gp { config, &problem_, treeInit_.get(), &coeffInit_,
                    generator_.get(), reinserter_.get() };
                return f(gp);
            }
            Operon::NSGA2 nsga { config, &problem_, treeInit_.get(), &coeffInit_, generator_.get(), reinserter_.get(),
                &sorter_ };
            return f(nsga);
        }

        [[nodiscard]] auto Generator() -> OffspringGeneratorBase& { return *generator_; }
        [[nodiscard]] auto ErrorEvaluator() -> Operon::Evaluator<DTable>& { return *errorEvaluator_; }
        [[nodiscard]] auto ActiveEvaluator() -> EvaluatorBase&
        {
            return scenario_.Algo == Algorithm::GP ? static_cast<EvaluatorBase&>(*errorEvaluator_)
                                                   : static_cast<EvaluatorBase&>(*multiEvaluator_);
        }

    private:
        auto MakeGenerator(Scenario const& sc, EvaluatorBase* eval) -> std::unique_ptr<OffspringGeneratorBase>
        {
            auto const* opt = coeffOptimizer_.get();
            if (sc.Generator == "basic") {
                return std::make_unique<BasicOffspringGenerator>(
                    eval, crossover_.get(), &mutator_, fem_.get(), male_.get(), opt);
            }
            if (sc.Generator == "os") {
                auto g = std::make_unique<OffspringSelectionGenerator>(
                    eval, crossover_.get(), &mutator_, fem_.get(), male_.get(), opt);
                g->MaxSelectionPressure(sc.Pressure);
                g->ComparisonFactor(sc.Factor);
                return g;
            }
            if (sc.Generator == "brood") {
                auto g = std::make_unique<BroodOffspringGenerator>(
                    eval, crossover_.get(), &mutator_, fem_.get(), male_.get(), opt);
                g->BroodSize(3);
                return g;
            }
            if (sc.Generator == "capped") {
                auto g = std::make_unique<CappedGenerator>(
                    eval, crossover_.get(), &mutator_, fem_.get(), male_.get(), opt);
                g->Cap = sc.Cap;
                g->TerminateAlways = sc.Terminate;
                return g;
            }
            auto g = std::make_unique<PolygenicOffspringGenerator>(
                eval, crossover_.get(), &mutator_, fem_.get(), male_.get(), opt);
            g->PolygenicSize(3);
            return g;
        }

        Scenario scenario_;
        Operon::Problem problem_;
        DTable dtable_;
        Operon::RandomGenerator cacheRng_;
        std::unique_ptr<Operon::BalancedTreeCreator> creator_;
        std::unique_ptr<Operon::UniformTreeInitializer> treeInit_;
        Operon::NormalCoefficientInitializer coeffInit_;
        std::unique_ptr<Operon::SubtreeCrossover> crossover_;
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint_;
        std::unique_ptr<Operon::ChangeVariableMutation> changeVar_;
        std::unique_ptr<Operon::ChangeFunctionMutation> changeFunc_;
        std::unique_ptr<Operon::ReplaceSubtreeMutation> replaceSubtree_;
        std::unique_ptr<Operon::InsertSubtreeMutation> insertSubtree_;
        std::unique_ptr<Operon::RemoveSubtreeMutation> removeSubtree_;
        Operon::MultiMutation mutator_;
        std::unique_ptr<Operon::Zobrist> cache_;
        std::unique_ptr<Operon::LevenbergMarquardtOptimizer<DTable, Operon::OptimizerType::Eigen>> optimizer_;
        std::unique_ptr<Operon::CoefficientOptimizer> coeffOptimizer_;
        std::unique_ptr<Operon::Evaluator<DTable>> errorEvaluator_;
        std::unique_ptr<Operon::TreePropertyEvaluator> lengthEvaluator_;
        std::unique_ptr<Operon::MultiEvaluator> multiEvaluator_;
        std::unique_ptr<Operon::TournamentSelector> fem_;
        std::unique_ptr<Operon::TournamentSelector> male_;
        std::unique_ptr<Operon::ReinserterBase> reinserter_;
        Operon::RankIntersectSorter sorter_;
        std::unique_ptr<OffspringGeneratorBase> generator_;
    };

    auto Poly10() -> Dataset& // loaded once: tests run from the repository root, like the other data-driven tests
    {
        static Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
        return ds;
    }

    auto Run(Scenario const& sc) -> Outcome
    {
        Fixture f { Poly10(), sc };
        return f.With(f.Config(), [&](auto& algo) {
            Operon::RandomGenerator random { sc.Seed };
            algo.Run(random, nullptr, sc.Threads);
            auto outcome = Capture(algo);
            outcome.Calls = f.ActiveEvaluator().CallCount;
            if (auto const* capped = dynamic_cast<CappedGenerator const*>(&f.Generator())) {
                outcome.Attempts = capped->Invocations.load();
            }
            return outcome;
        });
    }

    // The clock of the transposition cache is advanced by a generation that is then discarded; the rollback removes
    // its entries but not the clock value, which the next attempt sets again before it looks at anything. Everything
    // else a checkpoint holds must be identical, so comparisons of checkpoints ignore the clock.
    auto WithoutCacheClock(std::string const& bytes) -> std::string
    {
        auto checkpoint = Serialization::CheckpointFromBeve(bytes);
        REQUIRE(checkpoint.has_value());
        if (checkpoint->Cache) {
            checkpoint->Cache->Clock = 0;
        }
        return Serialization::ToBeve(*checkpoint);
    }

    // Continues from a serialized checkpoint in a brand-new Fixture (new dataset view, operators, generator,
    // evaluator and transposition cache, as in a new process): nothing but the bytes is shared with the run that
    // wrote them, so whatever state a checkpoint does not carry is missing here, as it is after `--resume`.
    auto ResumeFromBytes(Scenario const& sc, std::string const& bytes, std::size_t generations, std::size_t threads)
        -> State
    {
        Fixture f { Poly10(), sc };
        Operon::RandomGenerator random { 0 };
        return f.With(f.Config(generations, Unlimited), [&](auto& algo) {
            RestoreCheckpoint(algo, random, bytes);
            algo.Run(random, nullptr, threads, /*warmStart=*/true);
            return StateOf(algo);
        });
    }

    auto Label(Scenario const& sc) -> std::string
    {
        return std::string(sc.Algo == Algorithm::GP ? "gp" : "nsga2") + " " + sc.Generator
            + (sc.LocalSearch ? " ls" : " no-ls") + (sc.Cache ? " cache" : " no-cache")
            + (sc.ReplaceWorst ? " replace-worst" : " keep-best") + " pop " + std::to_string(sc.Population) + " pool "
            + std::to_string(sc.Pool);
    }

    constexpr std::array<std::size_t, 3> ThreadCounts { 1, 4, 16 };

    // Which seeds make a limit bind depends on the platform's std::uniform_*_distribution (libstdc++, libc++ and MSVC
    // draw differently), so no test may assume that a fixed seed binds a cap, discards a generation or produces a
    // cache hit. Scenarios that need such a property scan seeds at run time: `body(seed)` returns true when the seed
    // had the property (and has then run its checks); the scan stops after `wanted` such seeds, or after `limit`
    // seeds, and the caller REQUIREs that at least one seed qualified.
    template <typename F>
    auto ScanSeeds(std::uint64_t first, std::size_t limit, std::size_t wanted, F&& body) -> std::size_t
    {
        std::size_t qualified = 0;
        for (std::size_t i = 0; i < limit && qualified < wanted; ++i) {
            if (body(first + i)) {
                ++qualified;
            }
        }
        return qualified;
    }

    // The informational flags of a discarded attempt. With one finite limit (budget below unlimited, or a generator
    // with an attempt cap, not both) they are deterministic: the workers stop on that limit alone.
    auto Flags(Outcome const& outcome) -> std::pair<bool, bool>
    {
        return outcome.Progress ? std::pair { outcome.Progress->BudgetExceeded, outcome.Progress->AttemptCapExceeded }
                                : std::pair { false, false };
    }

    void CheckOnlyBudgetObserved(Outcome const& outcome)
    {
        REQUIRE(outcome.Progress.has_value());
        CHECK(outcome.Progress->Reason == DiscardReason::ResourceLimit);
        CHECK(outcome.Progress->BudgetExceeded);
        CHECK_FALSE(outcome.Progress->AttemptCapExceeded);
    }

    void CheckOnlyCapObserved(Outcome const& outcome)
    {
        REQUIRE(outcome.Progress.has_value());
        CHECK(outcome.Progress->Reason == DiscardReason::ResourceLimit);
        CHECK(outcome.Progress->AttemptCapExceeded);
        CHECK_FALSE(outcome.Progress->BudgetExceeded);
    }

} // namespace

// The boundary of the budget: one evaluation less than init + k generations commits k - 1 generations, exactly
// init + k commits k, and the committed state is the one an unbounded run reaches after that many generations,
// at every thread count. (Before the ledger, 16 threads committed the generation that did not fit.)
TEST_CASE("Budget boundary: committed state is the same at 1, 4 and 16 threads", "[budget][determinism]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = GENERATE(as<std::string> {}, "basic", "os", "brood", "poly");
    sc.LocalSearch = GENERATE(false, true);
    sc.Cache = GENERATE(false, true);
    // population = pool and population != pool, each with keep-best; replace-worst for both shapes
    std::tie(sc.Population, sc.Pool, sc.ReplaceWorst)
        = GENERATE(std::tuple<std::size_t, std::size_t, bool> { 40, 40, false },
            std::tuple<std::size_t, std::size_t, bool> { 32, 48, false },
            std::tuple<std::size_t, std::size_t, bool> { 40, 40, true },
            std::tuple<std::size_t, std::size_t, bool> { 32, 48, true });
    CAPTURE(Label(sc));

    // committed[k]: what an unbounded run commits with k generations (k = 0 is initialization alone).
    constexpr std::size_t maxGenerations = 2;
    std::vector<Outcome> reference;
    for (std::size_t k = 0; k <= maxGenerations; ++k) {
        auto unbounded = sc;
        unbounded.Generations = k;
        reference.push_back(Run(unbounded));
        REQUIRE(reference.back().GenerationsCommitted == k);
        REQUIRE_FALSE(reference.back().Discarded);
        if (k > 0) {
            // non-vacuous: every generation costs something, so budgets one below and at the boundary differ
            REQUIRE(reference[k].EvaluationsCommitted > reference[k - 1].EvaluationsCommitted);
        }
    }

    for (std::size_t k = 1; k <= maxGenerations; ++k) {
        for (auto const threads : ThreadCounts) {
            CAPTURE(k, threads);
            auto bounded = sc;
            bounded.Threads = threads;

            // One evaluation short: generation k does not fit; the state is that after k - 1 generations,
            // including the worker RNG states (the discarded generation was rolled back).
            bounded.Budget = reference[k].EvaluationsCommitted - 1;
            auto const below = Run(bounded);
            CHECK(below.Committed == reference[k - 1].Committed);
            CHECK(below.GenerationsCommitted == k - 1);
            CHECK(below.EvaluationsCommitted == reference[k - 1].EvaluationsCommitted);
            CHECK(below.BudgetOvershoot == 0);
            CHECK(below.Discarded);
            CHECK_FALSE(below.StopRequested);
            REQUIRE(below.Progress.has_value());
            CHECK(below.Progress->Reason == DiscardReason::ResourceLimit);
            CHECK(below.End == RunEnd::ResourceLimit);
            if (sc.Generator != "os") { // os has a finite attempt cap as well: then the flags are not deterministic
                CheckOnlyBudgetObserved(below);
            }

            // Exactly enough: generation k commits and the budget is spent, so nothing is attempted afterwards.
            bounded.Budget = reference[k].EvaluationsCommitted;
            auto at = Run(bounded);
            CHECK(at.End == RunEnd::BudgetSpent); // the budget is spent before the generation limit (1000)
            at.End = reference[k].End; // the same committed outcome as the unbounded run, but for why it ended
            CHECK(at == reference[k]);
            CHECK(at.GenerationsCommitted == k);
            CHECK(at.EvaluationsCommitted == bounded.Budget);
            CHECK_FALSE(at.Discarded);
        }
    }
}

// Offspring selection: a generation commits iff its pool is full and the attempts it makes stay within
// MaxSelectionPressure * parents (the sequential rule of BudgetLedger). Before the ledger this depended on how many
// attempts the other threads had in flight: with pop = pool = 8 and a pressure of 2, 1 thread committed generation 0
// only and 4 to 16 threads sometimes committed generation 1.
TEST_CASE("Offspring-selection pressure binds identically at 1, 4 and 16 threads", "[budget][determinism][os]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Cache = GENERATE(false, true);
    sc.ReplaceWorst = GENERATE(false, true);
    sc.Pressure = GENERATE(std::size_t { 1 }, std::size_t { 2 });
    sc.Generator = "os";
    sc.Population = 8;
    sc.Pool = 8;
    sc.Factor = 1.0;
    sc.Generations = 100;
    CAPTURE(Label(sc), sc.Pressure);

    auto const bound = ScanSeeds(13, 64, 3, [&](std::uint64_t seed) {
        sc.Seed = seed;
        auto const reference = Run(sc);
        // the pressure, not the generation limit, must have ended the run: either a generation was discarded for the
        // attempt cap, or the last committed generation crossed it (which ends the run without a discard)
        if (reference.GenerationsCommitted >= sc.Generations) {
            return false;
        }
        CAPTURE(seed);
        if (reference.Discarded) {
            CheckOnlyCapObserved(reference); // the budget is unlimited: one finite limit
            CHECK(reference.End == RunEnd::ResourceLimit);
        } else {
            CHECK(reference.End == RunEnd::AttemptCap); // the last committed generation crossed the cap
        }
        CHECK_FALSE(reference.StopRequested);

        for (auto const threads :
            { std::size_t { 1 }, std::size_t { 4 }, std::size_t { 16 }, std::size_t { 16 }, std::size_t { 16 } }) {
            CAPTURE(threads);
            auto bounded = sc;
            bounded.Threads = threads;
            auto const outcome = Run(bounded);
            CHECK(outcome == reference);
            CHECK(Flags(outcome) == Flags(reference));
        }
        return true;
    });
    REQUIRE(bound >= 1); // non-vacuous: some seed let the pressure end the run
}

// Transposition-cache hits make no evaluator call but are attempts: the offspring-selection pressure is candidates
// generated per parent, whether they were evaluated or found in the cache. Generations after the first reuse
// structures through the cache, so their attempts exceed their evaluator calls, and the cap acts on the attempts.
TEST_CASE("Attempt cap: transposition-cache hits are attempts", "[budget][determinism][os]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = "capped";
    sc.Cache = true;
    sc.Population = 40;
    sc.Pool = 40;
    sc.Generations = 4;
    CAPTURE(Label(sc));

    auto const reused = ScanSeeds(4242, 64, 2, [&](std::uint64_t seed) {
        sc.Seed = seed;
        CAPTURE(seed);
        // attempts and evaluator calls of generation k at one thread, from unbounded runs
        std::vector<std::uint64_t> attempts;
        std::vector<std::uint64_t> calls;
        std::vector<Outcome> reference;
        for (std::size_t k = 0; k <= sc.Generations; ++k) {
            auto unbounded = sc;
            unbounded.Generations = k;
            reference.push_back(Run(unbounded));
            REQUIRE(reference.back().GenerationsCommitted == k);
            if (k > 0) {
                attempts.push_back(reference[k].Attempts - reference[k - 1].Attempts);
                calls.push_back(reference[k].Calls - reference[k - 1].Calls);
            }
        }
        auto const totalAttempts = std::accumulate(attempts.begin(), attempts.end(), std::uint64_t { 0 });
        auto const totalCalls = std::accumulate(calls.begin(), calls.end(), std::uint64_t { 0 });
        if (totalAttempts == totalCalls) {
            return false; // this seed never reused a structure: attempts == evaluator calls, nothing to show
        }
        CHECK(totalAttempts > totalCalls); // a hit is an attempt without a call, never the other way round

        auto const maxAttempts = *std::ranges::max_element(attempts);
        for (auto const threads : ThreadCounts) {
            CAPTURE(threads);
            auto capped = sc;
            capped.Threads = threads;

            // a cap equal to the busiest generation's attempts lets every generation through
            capped.Cap = maxAttempts;
            CHECK(Run(capped) == reference.back());

            // two attempts below the busiest generation's, the run cannot complete: either that generation cannot
            // commit (at least maxAttempts - 1 count) and is discarded, or an earlier generation of maxAttempts - 1
            // attempts commits and, crossing the cap with its last attempt, ends the run
            capped.Cap = maxAttempts - 2;
            auto const limited = Run(capped);
            CHECK(limited.GenerationsCommitted < sc.Generations);
            if (limited.Discarded) {
                CheckOnlyCapObserved(limited);
                CHECK(limited.End == RunEnd::ResourceLimit);
            } else {
                CHECK(limited.End == RunEnd::AttemptCap);
            }
            CHECK(limited.Committed == reference[limited.GenerationsCommitted].Committed);
        }
        return true;
    });
    REQUIRE(reused >= 1); // non-vacuous: some seed let the cache serve attempts, so attempts != evaluator calls
}

// The edge of the attempt cap, with a generator whose every attempt makes one evaluator call (no cache): the
// sequential rule lets an attempt start while the attempts so far are <= cap, so a generation whose *last accepted
// attempt* brings the attempts to cap + 1 still commits (this is what the pre-ledger single-threaded stop did), and
// one more attempt before it does not.
TEST_CASE("Attempt cap: the edge is the sequential one, at every thread count", "[budget][determinism][os]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = "capped";
    sc.Population = 24;
    sc.Pool = 32;
    sc.ReplaceWorst = GENERATE(false, true);
    CAPTURE(Label(sc));

    // Rejections come from the generator's raw random draw, but how many draws the rest of an attempt consumes depends
    // on the platform: take the first seed whose first generation needs more attempts than the pool has slots.
    for (std::uint64_t seed = 4242; seed < 4242 + 64; ++seed) {
        sc.Seed = seed;
        auto probe = sc;
        probe.Generations = 0;
        auto const before = Run(probe).Calls;
        probe.Generations = 1;
        if (Run(probe).Calls - before > sc.Pool) {
            break;
        }
    }
    CAPTURE(sc.Seed);

    auto initOnly = sc;
    initOnly.Generations = 0;
    auto const initial = Run(initOnly);
    auto oneGeneration = sc;
    oneGeneration.Generations = 1;
    auto const one = Run(oneGeneration);
    REQUIRE(one.GenerationsCommitted == 1);
    auto const needed = one.Attempts - initial.Attempts; // attempts of the first generation, at one thread
    REQUIRE(needed == one.Calls - initial.Calls); // no cache: an attempt is one evaluator call
    REQUIRE(needed > sc.Pool); // some attempts were rejected, so the cap is not the pool size by accident

    for (auto const threads :
        { std::size_t { 1 }, std::size_t { 4 }, std::size_t { 16 }, std::size_t { 16 }, std::size_t { 16 } }) {
        CAPTURE(threads);
        auto capped = oneGeneration;
        capped.Threads = threads;

        // cap = needed - 1: attempts reach cap + 1 on the last accepted attempt, which started within the cap
        capped.Cap = needed - 1;
        auto atEdge = Run(capped);
        CHECK(atEdge.End == RunEnd::AttemptCap); // the generation crossed the cap: the run ends, nothing is discarded
        atEdge.End = one.End; // the same committed outcome as the unbounded generation, but for why it ended
        CHECK(atEdge == one);
        CHECK_FALSE(atEdge.Discarded);

        // ... and, as the single-threaded pressure stop always did, the run ends after that generation (it is not a
        // discard: the generation is committed and reported)
        auto longer = capped;
        longer.Generations = 3;
        auto ended = Run(longer);
        CHECK(ended.End == RunEnd::AttemptCap); // ended by the committed generation that crossed the cap, no discard
        ended.End = one.End;
        CHECK(ended == one);
        CHECK_FALSE(ended.Discarded);

        capped.Cap = needed;
        auto const within = Run(capped);
        CHECK(within == one);
        CHECK(within.End == RunEnd::GenerationLimit);
        longer.Cap = 2 * needed;
        CHECK(Run(longer).GenerationsCommitted == 3); // well within the cap: the run goes on

        // cap = needed - 2: the attempt before the last one already started above the cap
        capped.Cap = needed - 2;
        auto const below = Run(capped);
        CHECK(below.Committed == initial.Committed);
        CHECK(below.Discarded);
        CHECK(below.GenerationsCommitted == 0);
        CHECK_FALSE(below.StopRequested); // a cap does not call RequestStop()
        CheckOnlyCapObserved(below);
        CHECK(below.End == RunEnd::ResourceLimit);
        // Executed work can pass the cap by the attempts that were in flight, and is discarded with the generation:
        // workers do not start an attempt once the published attempts exceed the cap, so at most one attempt per worker
        // (plus the uncounted last accepted one) runs beyond it.
        auto const executed = below.Attempts - initial.Attempts;
        CHECK(executed > capped.Cap.value());
        CHECK(executed <= capped.Cap.value() + threads + 1);
    }
}

TEST_CASE("Attempt cap: cap 0 discards a multi-slot pool; a pool of one slot commits iff its first attempt is accepted",
    "[budget][determinism][os]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = "capped";
    sc.Generations = 1;
    sc.Cap = 0;
    CAPTURE(Label(sc));

    SECTION("pool of 32")
    {
        sc.Population = 24;
        sc.Pool = 32;
        auto initOnly = sc;
        initOnly.Generations = 0;
        auto const initial = Run(initOnly);
        for (auto const threads : ThreadCounts) {
            CAPTURE(threads);
            sc.Threads = threads;
            auto const outcome = Run(sc);
            CHECK(outcome.Committed == initial.Committed);
            CHECK(outcome.Discarded);
            CheckOnlyCapObserved(outcome);
        }
    }

    SECTION("pool of 1")
    {
        // The only slot is the last one: its accepted attempt is not counted, its rejected attempts are, so with cap 0
        // the generation commits iff the very first attempt is accepted.
        sc.Population = 8;
        sc.Pool = 1;
        std::size_t committed = 0;
        std::size_t discarded = 0;
        // which seeds reject the first attempt depends on the platform's distributions: scan until both were seen
        for (std::uint64_t seed = 1; seed <= 64 && (committed == 0 || discarded == 0); ++seed) {
            sc.Seed = seed;
            sc.Threads = 1;
            auto const reference = Run(sc);
            (reference.Discarded ? discarded : committed) += 1;
            for (auto const threads : { std::size_t { 4 }, std::size_t { 16 } }) {
                sc.Threads = threads;
                CHECK(Run(sc) == reference);
            }
        }
        CHECK(committed > 0);
        CHECK(discarded > 0);
    }
}

TEST_CASE("A discarded generation does not change a warm continuation", "[budget][determinism][rollback]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = GENERATE(as<std::string> {}, "basic", "brood");
    sc.LocalSearch = GENERATE(false, true);
    sc.Cache = GENERATE(false, true);
    sc.ReplaceWorst = GENERATE(false, true);
    CAPTURE(Label(sc));

    constexpr std::size_t committedGenerations = 2;
    constexpr std::size_t totalGenerations = 4;

    // reference: two generations without interruption, then a warm continuation to the generation limit
    auto clean = sc;
    clean.Generations = committedGenerations;
    auto const twoGenerations = Run(clean);
    REQUIRE(twoGenerations.GenerationsCommitted == committedGenerations);
    auto threeGenerations = sc;
    threeGenerations.Generations = committedGenerations + 1;
    auto const interruptedBudget = Run(threeGenerations).EvaluationsCommitted - 1; // generation 3 does not fit
    REQUIRE(interruptedBudget >= twoGenerations.EvaluationsCommitted);

    for (auto const threads : ThreadCounts) {
        CAPTURE(threads);

        // reference continuation: from the checkpoint of a clean two-generation run
        std::string cleanCheckpoint;
        {
            Fixture f { Poly10(), sc };
            Operon::RandomGenerator random { sc.Seed };
            cleanCheckpoint = f.With(f.Config(committedGenerations, Unlimited), [&](auto& algo) {
                algo.Run(random, nullptr, threads);
                return CaptureCheckpoint(algo, random);
            });
        }
        auto const expected = ResumeFromBytes(sc, cleanCheckpoint, totalGenerations, threads);

        // interrupted: generation 3 is discarded for lack of budget; the checkpoint written then is the one of the
        // clean run (population, RNG states, cache entries), and the continuation from it is the same
        std::string interruptedCheckpoint;
        {
            Fixture f { Poly10(), sc };
            Operon::RandomGenerator random { sc.Seed };
            interruptedCheckpoint = f.With(f.Config(totalGenerations, interruptedBudget), [&](auto& algo) {
                algo.Run(random, nullptr, threads);
                REQUIRE(algo.Generation() == committedGenerations);
                REQUIRE(algo.IncompleteGeneration().has_value());
                CHECK(StateOf(algo) == twoGenerations.Committed);
                return CaptureCheckpoint(algo, random);
            });
        }
        CHECK(Same(WithoutCacheClock(interruptedCheckpoint), WithoutCacheClock(cleanCheckpoint)));
        auto const actual = ResumeFromBytes(sc, interruptedCheckpoint, totalGenerations, threads);
        CHECK(actual.Generation == totalGenerations);
        CHECK(actual == expected);
    }
}

TEST_CASE("An unusable population, pool or budget is a configuration error", "[budget][config]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));
    auto const warmStart = GENERATE(false, true);

    struct Bad {
        std::size_t Population;
        std::size_t Pool;
        std::size_t Budget;
        char const* Message;
    };
    auto const bad = GENERATE(Bad { 0, 16, 100, "population size must be at least 1" },
        Bad { 0, 0, 0, "population size must be at least 1" }, Bad { 16, 0, 100, "pool size must be at least 1" });
    CAPTURE(bad.Population, bad.Pool, bad.Budget);
    sc.Population = bad.Population;
    sc.Pool = bad.Pool;

    Fixture f { Poly10(), sc };
    f.With(f.Config(/*generations=*/3, bad.Budget), [&](auto& algo) {
        Operon::RandomGenerator random { sc.Seed };
        auto const before = Operon::RandomGenerator { random };
        REQUIRE_THROWS_AS(algo.Run(random, nullptr, 1, warmStart), std::invalid_argument);
        REQUIRE_THROWS_WITH(algo.Run(random, nullptr, 1, warmStart), Catch::Matchers::ContainsSubstring(bad.Message));
        // rejected before anything happened
        CHECK(random.state() == before.state());
        CHECK_FALSE(algo.IsFitted());
        CHECK(algo.Generation() == 0);
        CHECK(algo.BudgetReport() == BudgetAccounting {});
        CHECK(algo.WorkerRngs().empty());
        CHECK(f.ActiveEvaluator().CallCount == 0);
        return 0;
    });
}

TEST_CASE("A budget of exactly the population size is accepted", "[budget][config]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));
    auto const warmStart = GENERATE(false, true);
    Fixture f { Poly10(), sc };
    f.With(f.Config(3, sc.Population), [&](auto& algo) {
        Operon::RandomGenerator random { sc.Seed };
        REQUIRE_NOTHROW(algo.Run(random, nullptr, 1));
        CHECK(algo.IsFitted());
        if (warmStart) { // a warm start of the run that just completed (a fitted population) is accepted as well
            REQUIRE_NOTHROW(algo.Run(random, nullptr, 1, /*warmStart=*/true));
            CHECK(algo.IsFitted());
        }
        return 0;
    });
}

TEST_CASE("A zero-cost structural evaluator accepts a zero evaluation budget", "[budget][config]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Structural = true;
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));

    Fixture f { Poly10(), sc };
    f.With(f.Config(/*generations=*/3, /*budget=*/0), [&](auto& algo) {
        Operon::RandomGenerator random { sc.Seed };
        REQUIRE_NOTHROW(algo.Run(random, nullptr, 1));
        CHECK(algo.IsFitted());
        CHECK(algo.BudgetReport().EvaluationsCommitted == 0);
        CHECK(algo.BudgetReport().BudgetOvershoot == 0);
        CHECK(algo.BudgetReport().End == RunEnd::BudgetSpent);
        return 0;
    });
}


TEST_CASE("Initialization always commits: overshoot is recorded and no generation starts", "[budget][config]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.LocalSearch = true; // local search makes initialization cost more than one evaluation per individual
    sc.Cache = GENERATE(false, true);
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));

    auto initOnly = sc;
    initOnly.Generations = 0;
    auto const reference = Run(initOnly);
    auto const cost = reference.EvaluationsCommitted;
    REQUIRE(cost > sc.Population);
    REQUIRE(reference.BudgetOvershoot == 0);

    for (auto const threads : ThreadCounts) {
        CAPTURE(threads);
        auto bounded = sc;
        bounded.Threads = threads;

        // The budget covers the population but not the initialization: the initial population is kept anyway.
        bounded.Budget = sc.Population;
        auto const over = Run(bounded);
        CHECK(over.Committed == reference.Committed);
        CHECK(over.EvaluationsCommitted == cost);
        CHECK(over.BudgetOvershoot == cost - sc.Population);
        CHECK(over.GenerationsCommitted == 0);
        CHECK_FALSE(over.Discarded); // no generation was even attempted
        CHECK(over.Accounting.EvaluationsDiscarded == 0);
        CHECK(over.Accounting.Budget == sc.Population);

        // A budget equal to the cost leaves nothing for a generation: none is attempted.
        bounded.Budget = cost;
        auto exact = Run(bounded);
        CHECK(exact.End == RunEnd::BudgetSpent);
        exact.End = reference.End; // the initial population either way, but for why the run ended
        CHECK(exact == reference);
        CHECK(exact.BudgetOvershoot == 0);
        CHECK_FALSE(exact.Discarded);

        // One evaluation more is not enough for a generation: it is attempted, discarded and rolled back.
        bounded.Budget = cost + 1;
        auto const spare = Run(bounded);
        CHECK(spare.Committed == reference.Committed);
        CHECK(spare.Discarded);
        CHECK(spare.GenerationsCommitted == 0);
    }
}

TEST_CASE("Accounting: committed, executed and discarded work", "[budget][accounting]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.LocalSearch = GENERATE(false, true);
    sc.Generations = 3;
    CAPTURE(Label(sc));

    SECTION("an unbounded run commits everything it executes")
    {
        Fixture f { Poly10(), sc };
        f.With(f.Config(), [&](auto& algo) {
            CHECK(algo.BudgetReport() == BudgetAccounting {}); // before the first Run()
            Operon::RandomGenerator random { sc.Seed };
            algo.Run(random, nullptr, 1);
            auto const& accounting = algo.BudgetReport();
            CHECK(accounting.Budget == Unlimited);
            CHECK(accounting.GenerationsCommitted == 3);
            CHECK(accounting.EvaluationsCommitted == accounting.EvaluationsExecuted);
            CHECK(accounting.EvaluationsDiscarded == 0);
            CHECK(accounting.BudgetOvershoot == 0);
            CHECK(accounting.EvaluationsExecuted == algo.DetailedStats().Consumed());

            auto const copy = algo; // the accounting travels with the algorithm and is cleared by Reset()
            CHECK(copy.BudgetReport() == accounting);
            algo.Reset();
            CHECK(algo.BudgetReport() == BudgetAccounting {});
            return 0;
        });
    }

    SECTION("a discarded generation is executed but not committed")
    {
        auto unbounded = sc;
        auto const cost = Run(unbounded).EvaluationsCommitted;
        for (auto const threads : ThreadCounts) {
            CAPTURE(threads);
            auto bounded = sc;
            bounded.Threads = threads;
            bounded.Budget = cost - 1;
            auto const outcome = Run(bounded);
            auto const& accounting = outcome.Accounting;
            CHECK(accounting.Budget == cost - 1);
            CHECK(accounting.GenerationsCommitted == 2);
            CHECK(accounting.EvaluationsCommitted < cost);
            CHECK(accounting.EvaluationsDiscarded > 0);
            CHECK(accounting.EvaluationsExecuted >= accounting.EvaluationsCommitted);
            // nothing but the committed work and the discarded attempt used the evaluator
            CHECK(accounting.EvaluationsExecuted == accounting.EvaluationsCommitted + accounting.EvaluationsDiscarded);
            // The attempt stopped starting work once it had spent more than the budget had left.
            CHECK(accounting.EvaluationsDiscarded > cost - 1 - accounting.EvaluationsCommitted);
            CHECK(accounting.EvaluationsCommitted + accounting.EvaluationsDiscarded <= cost + sc.Pool);
        }
    }

    SECTION("independent expected value: basic generator, no local search, no cache")
    {
        // every offspring costs exactly one evaluation, so a generation costs Pool; with one evaluation less than
        // that left, the attempt spends all Pool before it can be told it exceeded, at every thread count
        sc.LocalSearch = false;
        sc.Generator = "basic";
        sc.Population = 20;
        sc.Pool = 30;
        for (auto const threads : ThreadCounts) {
            CAPTURE(threads);
            auto bounded = sc;
            bounded.Threads = threads;
            bounded.Budget = sc.Population + sc.Pool - 1;
            auto const outcome = Run(bounded);
            CHECK(outcome.Accounting.EvaluationsCommitted == sc.Population);
            CHECK(outcome.Accounting.EvaluationsDiscarded == sc.Pool);
            CHECK(outcome.Accounting.EvaluationsExecuted == sc.Population + sc.Pool);
        }
    }

    SECTION("evaluator work done outside the generations is executed, not discarded")
    {
        sc.Generator = "basic";
        auto const plain = Run(sc);
        Fixture f { Poly10(), sc };
        f.With(f.Config(), [&](auto& algo) {
            Operon::RandomGenerator random { sc.Seed };
            Operon::RandomGenerator scoring { 1 };
            std::uint64_t reportEvaluations = 0;
            algo.Run(
                random,
                [&]() -> bool {
                    // a report callback that scores the best parent (e.g. a validation pass)
                    auto const consumed = [&] {
                        auto const stats = f.ActiveEvaluator().Stats();
                        return std::get<0>(stats) + std::get<1>(stats);
                    };
                    auto const before = consumed();
                    auto const fitness = f.ActiveEvaluator()(scoring, algo.Parents()[0]);
                    REQUIRE_FALSE(fitness.empty());
                    reportEvaluations += consumed() - before;
                    return false;
                },
                1);
            REQUIRE(reportEvaluations > 0);
            auto const& accounting = algo.BudgetReport();
            CHECK(accounting.EvaluationsCommitted == plain.EvaluationsCommitted);
            CHECK(accounting.EvaluationsDiscarded == 0);
            CHECK(accounting.EvaluationsExecuted == accounting.EvaluationsCommitted + reportEvaluations);
            CHECK(accounting.GenerationsCommitted == sc.Generations);
            CHECK(StateOf(algo) == plain.Committed); // the callback's scoring did not touch the committed state
            return 0;
        });
    }
}

// Counter arithmetic never wraps: sums and differences saturate, and a saturated evaluator total is an irreversible
// "exceeds the budget" within the generation.
TEST_CASE("Ledger arithmetic saturates", "[budget][accounting]")
{
    using detail::kSaturated;
    CHECK(detail::SaturatingAdd(kSaturated, 11) == kSaturated);
    CHECK(detail::SaturatingAdd(kSaturated - 5, 5) == kSaturated);
    CHECK(detail::SaturatingAdd(kSaturated - 5, 6) == kSaturated);
    CHECK(detail::SaturatingAdd(2, 3) == 5);
    CHECK(detail::SaturatingSub(3, 5) == 0);
    CHECK(detail::SaturatingSub(5, 3) == 2);

    Operon::Problem problem { gsl::not_null<Dataset*>(&Poly10()) };
    Operon::Evaluator<DTable> evaluator { &problem, std::make_unique<DTable>().get() };
    constexpr std::size_t budget = 100;
    OptimizerAccounting optimizer;
    detail::BudgetLedger ledger { &evaluator, &optimizer, budget };

    ledger.Begin();
    ledger.CommitInitialization();
    ledger.BeginGeneration(std::nullopt);
    CHECK_FALSE(ledger.BudgetExceeded());
    evaluator.ResidualEvaluations = 100;
    CHECK_FALSE(ledger.BudgetExceeded()); // exactly the budget is not more than the budget
    evaluator.ResidualEvaluations = 101;
    CHECK(ledger.BudgetExceeded());

    SECTION("the reported failure: residual = max, Jacobian = 11 (the sum would wrap to 10)")
    {
        evaluator.ResidualEvaluations = kSaturated;
        evaluator.JacobianEvaluations = 0;
        CHECK(ledger.BudgetExceeded());
        evaluator.JacobianEvaluations = 11;
        CHECK(ledger.Consumed() == kSaturated);
        CHECK(ledger.BudgetExceeded()); // stays exceeded after the sum would have wrapped
        CHECK(ledger.Exceeded());
        CHECK(ledger.Reason(false) == DiscardReason::ResourceLimit);

        ledger.DiscardGeneration();
        auto const accounting = ledger.Report();
        CHECK(accounting.EvaluationsExecuted == kSaturated);
        CHECK(accounting.EvaluationsDiscarded == kSaturated);
        CHECK(accounting.EvaluationsCommitted == 0);
        CHECK(accounting.EvaluationsExecuted >= accounting.EvaluationsCommitted);
    }

    SECTION("committing a saturated generation saturates the committed work instead of wrapping")
    {
        evaluator.ResidualEvaluations = kSaturated - 3;
        evaluator.JacobianEvaluations = 10;
        ledger.CommitGeneration();
        CHECK(ledger.Report().EvaluationsCommitted == kSaturated);
        CHECK(ledger.Report().BudgetOvershoot == kSaturated - budget);
        CHECK(ledger.Exhausted());
        CHECK(ledger.Remaining() == 0);
    }

    SECTION("counters reset behind the ledger's back do not produce wrapped differences")
    {
        evaluator.ResidualEvaluations = 50;
        ledger.Begin(); // baseline 50
        evaluator.Reset(); // e.g. a report callback
        auto const accounting = ledger.Report();
        CHECK(accounting.EvaluationsExecuted == 0);
        CHECK(accounting.EvaluationsDiscarded == 0);
        ledger.BeginGeneration(std::nullopt);
        CHECK_FALSE(ledger.BudgetExceeded());
    }

    SECTION("optimizer work consumes the budget together with scoring work")
    {
        evaluator.Reset();
        ledger.Begin();
        ledger.CommitInitialization();
        ledger.BeginGeneration(std::nullopt);
        evaluator.ResidualEvaluations = 60;
        CHECK_FALSE(ledger.BudgetExceeded());

        FitDiagnostics fit;
        fit.FunctionEvaluations = 30;
        fit.JacobianEvaluations = 10;
        optimizer.Record(fit, 1);
        CHECK(ledger.Consumed() == 100);
        CHECK_FALSE(ledger.BudgetExceeded()); // exactly the budget is not more than the budget

        fit = {};
        fit.JacobianEvaluations = 1;
        optimizer.Record(fit, 1);
        CHECK(ledger.Consumed() == 101);
        CHECK(ledger.BudgetExceeded());

        ledger.DiscardGeneration();
        CHECK(ledger.Report().EvaluationsExecuted == 101);
        CHECK(ledger.Report().EvaluationsDiscarded == 101);
    }

    SECTION("the attempt cap counts the attempts of every slot but the last accepted one")
    {
        auto const add = [&](std::size_t n, bool lastSlot, bool accepted) {
            for (std::size_t i = 0; i < n; ++i) {
                ledger.AddAttempt(lastSlot, accepted);
            }
        };
        ledger.BeginGeneration(5);
        add(3, false, true); // earlier slots, accepted
        add(2, true, false); // the last slot, rejected
        CHECK_FALSE(ledger.AttemptCapExceeded()); // 5 <= 5
        CHECK_FALSE(ledger.AttemptCapReached()); // 5 <= 5 in total
        add(1, true, true); // the last slot's accepted attempt is not counted ...
        CHECK_FALSE(ledger.AttemptCapExceeded());
        CHECK(ledger.AttemptCapReached()); // ... except to end the run after a committed generation (6 > 5)
        add(1, false, false);
        CHECK(ledger.AttemptCapExceeded()); // 6 > 5
        CHECK(ledger.Reason(false) == DiscardReason::ResourceLimit);
        CHECK(ledger.Reason(true) == DiscardReason::ResourceLimit); // a resource limit comes before the async causes
        CHECK(ledger.Progress(1, 2, 3, false).AttemptCapExceeded);
        CHECK_FALSE(ledger.Progress(1, 2, 3, false).BudgetExceeded);
        evaluator.ResidualEvaluations = 1000; // the budget fails as well: both flags, one reason
        CHECK(ledger.BudgetExceeded());
        auto const both = ledger.Progress(1, 2, 3, true);
        CHECK(both.Reason == DiscardReason::ResourceLimit);
        CHECK(both.BudgetExceeded);
        CHECK(both.AttemptCapExceeded);
        evaluator.ResidualEvaluations = 101;
        ledger.BeginGeneration(std::nullopt);
        add(1000, false, false);
        CHECK_FALSE(ledger.AttemptCapExceeded()); // no cap, nothing counted
        CHECK(ledger.Reason(true) == DiscardReason::StopRequested);
        CHECK(ledger.Reason(false) == DiscardReason::TimeLimit);
    }
}

// A run whose evaluator saturates mid-generation discards that generation (even with an unlimited budget) and keeps
// the committed state, at every thread count.
TEST_CASE("A saturated evaluator total discards the generation", "[budget][accounting][determinism]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generations = 4;
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));

    auto oneGeneration = sc;
    oneGeneration.Generations = 1;
    auto const reference = Run(oneGeneration);

    // Prepare() calls: 1 = initialization, 2 = generation 1, 3 = generation 2 (the generation that saturates)
    sc.SaturateOnPrepare = 3;
    for (auto const threads : ThreadCounts) {
        CAPTURE(threads);
        sc.Threads = threads;
        auto const outcome = Run(sc);
        CHECK(outcome.Committed == reference.Committed);
        CHECK(outcome.GenerationsCommitted == 1);
        CheckOnlyBudgetObserved(outcome);
        CHECK(outcome.Accounting.EvaluationsDiscarded > outcome.Accounting.EvaluationsCommitted);
        CHECK(outcome.Accounting.EvaluationsExecuted >= outcome.Accounting.EvaluationsCommitted);
    }
}

// Both limits finite and competing: the workers stop on whichever trips first, and work in flight may or may not trip
// the other one, so the informational flags depend on the thread count. The decision, the reason and everything
// committed do not: the reason is ResourceLimit at every thread count. (A reason that named the limit used to flip
// between 1 and 4+ threads in all of these scenarios.)
TEST_CASE("Competing budget and attempt cap: one reason at every thread count", "[budget][determinism][os]")
{
    // The scenarios are listed, not SECTIONs: each is built the same way on every platform and a seed is only used if
    // the generation does not fit in the budget, which depends on the platform's distributions (see ScanSeeds).
    enum class Case { OsSmall, OsLarge, CappedShort2, CappedShort10, CappedShort20 };
    auto const which
        = GENERATE(Case::OsSmall, Case::OsLarge, Case::CappedShort2, Case::CappedShort10, Case::CappedShort20);
    Scenario base;
    base.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    base.Cache = GENERATE(false, true);
    base.Generations = 1;
    std::size_t capShortfall = 0; // capped: the cap is the attempts of an unbounded generation minus this
    switch (which) {
    case Case::OsSmall: // population 8, pool 16, pressure 1, factor 1, budget 18
        base.Generator = "os";
        base.Population = 8;
        base.Pool = 16;
        base.Pressure = 1;
        base.Factor = 1.0;
        base.Budget = 18;
        break;
    case Case::OsLarge: // population 24, pool 64, pressure 1, budget 49
        base.Generator = "os";
        base.Population = 24;
        base.Pool = 64;
        base.Pressure = 1;
        base.Factor = 0.0;
        base.Budget = 49;
        break;
    case Case::CappedShort2:
        capShortfall = 2;
        break;
    case Case::CappedShort10:
        capShortfall = 10;
        break;
    case Case::CappedShort20:
        capShortfall = 20;
        break;
    }
    if (capShortfall != 0) {
        base.Generator = "capped";
        base.LocalSearch = true;
        base.Population = 64;
        base.Pool = 64;
    }
    CAPTURE(Label(base), base.Pressure);

    auto const bound = ScanSeeds(1, 32, 2, [&](std::uint64_t seed) {
        auto sc = base;
        sc.Seed = seed;
        CAPTURE(seed);
        auto initOnly = sc;
        initOnly.Generations = 0;
        initOnly.Budget = Unlimited;
        initOnly.Cap = std::nullopt;
        auto const initial = Run(initOnly);

        if (capShortfall != 0) {
            // Both limits bind by construction, whatever the platform's random draws: the cap is below what an
            // unbounded generation needs, and the budget is what the cap-only run spent (committed + discarded).
            auto unbounded = sc;
            unbounded.Generations = 1;
            auto const full = Run(unbounded);
            auto const needed = full.Attempts - initial.Attempts;
            if (needed <= capShortfall + 1) {
                return false;
            }
            sc.Cap = needed - capShortfall;
            auto const probe = Run(sc);
            if (!probe.Discarded) {
                return false;
            }
            sc.Budget = probe.EvaluationsCommitted + probe.Accounting.EvaluationsDiscarded;
        }
        CAPTURE(sc.Budget, sc.Cap.value_or(0));

        sc.Threads = 1;
        auto const reference = Run(sc);
        if (!reference.Discarded) {
            return false; // the generation fits on this platform's draws (e.g. with cache hits): nothing to compare
        }
        CHECK(reference.Reason == DiscardReason::ResourceLimit);
        CHECK(reference.End == RunEnd::ResourceLimit);
        CHECK(reference.Committed == initial.Committed);
        CHECK_FALSE(reference.StopRequested);
        for (auto const threads :
            { std::size_t { 4 }, std::size_t { 8 }, std::size_t { 16 }, std::size_t { 16 }, std::size_t { 16 } }) {
            CAPTURE(threads);
            sc.Threads = threads;
            auto const outcome = Run(sc);
            CHECK(outcome == reference); // includes Reason, StopRequested and End; not the informational flags
            CHECK(outcome.Reason == DiscardReason::ResourceLimit);
            REQUIRE(outcome.Progress.has_value());
            // at least one limit was seen exceeded: the workers stopped on it
            CHECK((outcome.Progress->BudgetExceeded || outcome.Progress->AttemptCapExceeded));
        }
        return true;
    });
    REQUIRE(bound >= 1); // non-vacuous: some seed made the generation not fit
}

// Why a run ended is part of the accounting, and deterministic except for the time limit and a stop request.
TEST_CASE("Run end is reported", "[budget][accounting]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));

    SECTION("generation limit")
    {
        sc.Generations = 2;
        for (auto const threads : ThreadCounts) {
            sc.Threads = threads;
            auto const outcome = Run(sc);
            CHECK(outcome.End == RunEnd::GenerationLimit);
            CHECK_FALSE(outcome.Discarded);
        }
    }

    SECTION("zero generations")
    {
        sc.Generations = 0;
        CHECK(Run(sc).End == RunEnd::GenerationLimit);
    }

    SECTION("initialization alone spends the budget")
    {
        sc.Budget = sc.Population;
        auto const outcome = Run(sc);
        CHECK(outcome.End == RunEnd::BudgetSpent);
        CHECK(outcome.GenerationsCommitted == 0);
        CHECK_FALSE(outcome.Discarded);
    }

    SECTION("a generation is discarded for a resource limit")
    {
        sc.Budget = 2 * sc.Population - 1;
        auto const outcome = Run(sc);
        CHECK(outcome.End == RunEnd::ResourceLimit);
        CHECK(outcome.Discarded);
    }

    SECTION("not started")
    {
        Fixture f { Poly10(), sc };
        f.With(f.Config(), [&](auto& algo) {
            CHECK(algo.BudgetReport().End == RunEnd::None);
            return 0;
        });
    }
}

// Composite evaluators aggregate in 64 bit and saturate: a child beyond 2^32 (unsigned long on LLP64) and sums beyond
// 2^64 neither truncate nor wrap.
TEST_CASE("MultiEvaluator::Stats() aggregates in 64 bit and saturates", "[budget][accounting]")
{
    using detail::kSaturated;
    Operon::Problem problem { gsl::not_null<Dataset*>(&Poly10()) };
    DTable dtable;
    Operon::Evaluator<DTable> childA { &problem, &dtable };
    Operon::Evaluator<DTable> childB { &problem, &dtable };
    Operon::MultiEvaluator multi { &problem };
    multi.Add(&childA);
    multi.Add(&childB);
    auto const sum = [&] {
        auto const stats = multi.Stats();
        return std::pair { std::get<0>(stats), std::get<1>(stats) };
    };

    constexpr std::uint64_t beyond32 = (std::uint64_t { 1 } << 32U) + 7;
    childA.ResidualEvaluations = beyond32;
    childB.ResidualEvaluations = beyond32;
    childA.JacobianEvaluations = 3;
    multi.ResidualEvaluations = 5;
    auto [residual, jacobian] = sum();
    CHECK(residual == 2 * beyond32 + 5); // would have been truncated to 14 + 5 with a 32 bit accumulator
    CHECK(jacobian == 3);

    childA.ResidualEvaluations = kSaturated;
    childB.ResidualEvaluations = 11; // the sum would wrap to 10
    std::tie(residual, jacobian) = sum();
    CHECK(residual == kSaturated);
    CHECK(multi.TotalEvaluations() == 5); // the composite's own counters only
    multi.SetBudget(100);
    CHECK(multi.BudgetExhausted());
    childB.ResidualEvaluations = 0;
    multi.ResidualEvaluations = 11; // the composite's own counter is added with saturation as well
    std::tie(residual, jacobian) = sum();
    CHECK(residual == kSaturated);

    childA.ResidualEvaluations = std::numeric_limits<std::size_t>::max() / 11;
    childB.ResidualEvaluations = std::numeric_limits<std::size_t>::max() / 11;
    multi.ResidualEvaluations = 0;
    std::tie(residual, jacobian) = sum();
    CHECK(residual == 2 * (std::numeric_limits<std::size_t>::max() / 11));

    // the detailed view saturates too
    EvaluationStats a;
    a.ResidualScoring = kSaturated - 1;
    a.ResidualOptimizer = 5;
    CHECK(a.ResidualTotal() == kSaturated);
    CHECK(a.Consumed() == kSaturated);
}

// The shared attempt accumulator saturates: once the sum has reached the maximum, adding more does not bring it back
// under the cap, also for a cap of the maximum value.
TEST_CASE("The attempt-cap sum saturates", "[budget][accounting]")
{
    using detail::kSaturated;
    Operon::Problem problem { gsl::not_null<Dataset*>(&Poly10()) };
    DTable dtable;
    Operon::Evaluator<DTable> evaluator { &problem, &dtable };
    OptimizerAccounting optimizer;
    detail::BudgetLedger ledger { &evaluator, &optimizer, Unlimited };
    ledger.Begin();
    ledger.CommitInitialization();

    // the primitive the ledger accumulates with: a sum that would wrap stays at the maximum
    std::atomic<std::uint64_t> sum { kSaturated - 1 };
    detail::SaturatingFetchAdd(sum, 1);
    CHECK(sum == kSaturated);
    detail::SaturatingFetchAdd(sum, 11); // would wrap to 10
    CHECK(sum == kSaturated);

    // a concurrent accumulation does not lose anything: 8 workers add 8000 attempts
    for (auto const [cap, exceeded] : { std::pair<std::size_t, bool> { 8000, false }, { 7999, true } }) {
        ledger.BeginGeneration(cap);
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) {
            workers.emplace_back([&] {
                for (int i = 0; i < 1000; ++i) {
                    ledger.AddAttempt(false, true);
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }
        CHECK(ledger.AttemptCapExceeded() == exceeded);
    }

    // a cap of the maximum value is not exceeded by an ordinary count, and an unlimited generation counts nothing
    ledger.BeginGeneration(std::numeric_limits<std::size_t>::max());
    ledger.AddAttempt(false, true);
    CHECK_FALSE(ledger.AttemptCapExceeded());
    ledger.BeginGeneration(std::nullopt);
    ledger.AddAttempt(false, true);
    CHECK_FALSE(ledger.AttemptCapExceeded());
}

TEST_CASE("The algorithms no longer consult Terminate() or the evaluator's own budget", "[budget][config]")
{
    Scenario sc;
    sc.Algo = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    sc.Generator = "capped";
    sc.Terminate = true;
    sc.Generations = 3;
    sc.Population = 16;
    sc.Pool = 16;
    CAPTURE(Label(sc));

    Fixture f { Poly10(), sc };
    OptimizerAccounting accounting;
    REQUIRE(f.Generator().Terminate(accounting));
    f.ActiveEvaluator().SetBudget(0); // telemetry only: it would stop every run at once
    REQUIRE(f.ActiveEvaluator().BudgetExhausted());
    f.With(f.Config(), [&](auto& algo) {
        Operon::RandomGenerator random { sc.Seed };
        algo.Run(random, nullptr, 2);
        CHECK(algo.Generation() == 3);
        CHECK(algo.BudgetReport().GenerationsCommitted == 3);
        CHECK_FALSE(algo.IncompleteGeneration().has_value());
        return 0;
    });
}

TEST_CASE("OffspringSelectionGenerator::MaxAttempts is the selection pressure times the parents", "[budget][os]")
{
    Scenario sc;
    sc.Generator = "os";
    Fixture f { Poly10(), sc };
    auto& os = dynamic_cast<OffspringSelectionGenerator&>(f.Generator());
    os.MaxSelectionPressure(3);
    CHECK(os.MaxAttempts(40) == std::optional<std::size_t> { 120 });
    CHECK(os.MaxAttempts(0) == std::optional<std::size_t> { 0 });
    os.MaxSelectionPressure(Unlimited / 2 + 1);
    CHECK(os.MaxAttempts(2) == std::optional<std::size_t> { Unlimited }); // saturates instead of wrapping

    for (auto const* name : { "basic", "brood", "poly" }) {
        sc.Generator = name;
        Fixture other { Poly10(), sc };
        CHECK_FALSE(other.Generator().MaxAttempts(40).has_value()); // no pressure: unaffected
    }
}

namespace {
    // Every initial individual has the same structure (only its coefficients differ).
    class SingleStructureInitializer final : public TreeInitializerBase {
    public:
        explicit SingleStructureInitializer(Tree tree) : tree_(std::move(tree)) {}
        auto operator()(Operon::RandomGenerator& /*random*/) const -> Tree override { return tree_; }

    private:
        Tree tree_;
    };

    // The parent with the best (or worst) first objective; no random draw, so the choice is the same on every platform.
    class ExtremeSelector final : public SelectorBase {
    public:
        explicit ExtremeSelector(bool worst) : worst_(worst) {}
        auto operator()(Operon::RandomGenerator& /*random*/) const -> std::size_t override
        {
            auto const pop = Population();
            std::size_t extreme = 0;
            for (std::size_t i = 1; i < pop.size(); ++i) {
                if (worst_ ? pop[i].Fitness[0] > pop[extreme].Fitness[0]
                           : pop[i].Fitness[0] < pop[extreme].Fitness[0]) {
                    extreme = i;
                }
            }
            return extreme;
        }

    private:
        bool worst_;
    };

    // A uniformly random parent, drawn from the raw engine output (not a standard library distribution).
    class RawRandomSelector final : public SelectorBase {
    public:
        auto operator()(Operon::RandomGenerator& random) const -> std::size_t override
        {
            return static_cast<std::size_t>(random() % Population().size());
        }
    };
} // namespace

// Liveness. One structure, no crossover or mutation, a child is a copy of its female parent, fitness depends on the
// coefficients (no linear scaling) and the committed transposition cache returns the fitness of the first copy of the
// structure for every later one: a cache hit is a worse model than the best male parent, so offspring selection
// (pressure 1, comparison factor 1) rejects every candidate for ever. A cache hit makes no evaluator call, so a cap
// on evaluator calls never rose; the cap on attempts does, and ends the generation. The run also has a time limit, so
// that a regression fails the test (End == TimeLimit) instead of hanging it.
TEST_CASE("Attempt cap: os with a cache cannot retry for ever on rejected cache hits", "[budget][os][cache][liveness]")
{
    auto const algorithm = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    auto const threads = GENERATE(std::size_t { 1 }, std::size_t { 4 });
    auto const worstFemale = GENERATE(true, false); // false: a random female parent, as in the CLI reproducer
    CAPTURE(algorithm == Algorithm::GP ? "gp" : "nsga2", threads, worstFemale);

    constexpr std::size_t population = 40;
    constexpr std::size_t pressure = 1;
    constexpr std::size_t timeLimitSeconds = 30;

    Dataset& ds = Poly10();
    Operon::Problem problem { gsl::not_null<Dataset*>(&ds) };
    problem.SetTrainingRange(0, static_cast<int>(TrainRows));
    problem.SetTestRange(static_cast<int>(TrainRows), static_cast<int>(2 * TrainRows));
    problem.SetTarget("Y");
    problem.SetLinearScalingEnabled(false);
    problem.ConfigurePrimitiveSet(NodeType::Variable | BuiltinOp::Add);
    std::vector<Operon::Hash> const inputs { ds.GetVariable("X1")->Hash };

    Operon::GrowTreeCreator creator { &problem.GetPrimitiveSet(), inputs, MaxLength };
    Operon::RandomGenerator structureRng { 1 };
    SingleStructureInitializer initializer { creator(structureRng, 5, 1, MaxDepth) };
    Operon::NormalCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });

    Operon::SubtreeCrossover crossover { 0.9, MaxDepth, MaxLength };
    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
    onePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
    Operon::MultiMutation mutator;
    mutator.Add(&onePoint, 1.0);

    DTable dtable;
    Operon::GeneticAlgorithmConfig config;
    config.Generations = 10;
    config.Iterations = 0;
    config.PopulationSize = population;
    config.PoolSize = population;
    config.Seed = 4242;
    config.TimeLimit = timeLimitSeconds;
    config.CrossoverProbability = 0.0;
    config.MutationProbability = 0.0;
    config.LocalSearchProbability = 0.0;
    config.LamarckianProbability = 0.0;

    Operon::RandomGenerator cacheRng { config.Seed };
    Operon::Zobrist cache { cacheRng, static_cast<int>(MaxLength), problem.GetInputs(),
        ZobristOptions { .MaxAge = 5, .Visibility = CacheVisibility::Committed } };
    config.Cache = &cache;

    Operon::Evaluator<DTable> errorEvaluator { &problem, &dtable };
    Operon::TreePropertyEvaluator lengthEvaluator { &problem,
        [](Operon::Tree const& tree) { return static_cast<Operon::Scalar>(tree.Length()); },
        static_cast<Operon::Scalar>(MaxLength) };
    Operon::MultiEvaluator multiEvaluator { &problem };
    multiEvaluator.Add(&errorEvaluator);
    multiEvaluator.Add(&lengthEvaluator);

    ExtremeSelector male { /*worst=*/false };
    ExtremeSelector worst { /*worst=*/true };
    RawRandomSelector random;
    SelectorBase const& female
        = worstFemale ? static_cast<SelectorBase const&>(worst) : static_cast<SelectorBase const&>(random);

    auto const check = [&](auto& algo, OffspringSelectionGenerator const& generator) {
        Operon::RandomGenerator rng { config.Seed };
        algo.Run(rng, nullptr, threads);
        auto const& accounting = algo.BudgetReport();
        CHECK(accounting.End != RunEnd::TimeLimit); // a regression runs into the time limit
        CHECK(accounting.End != RunEnd::StopRequested);
        REQUIRE(generator.Cache() != nullptr);
        CHECK(algo.DetailedStats().CacheHits.value_or(0) > 0); // non-vacuous: the candidates were cache hits
        if (worstFemale) {
            // every candidate is rejected: the cap, counted in attempts, discards the first generation
            auto const cap = pressure * population;
            CHECK(algo.Generation() == 0);
            CHECK(accounting.End == RunEnd::ResourceLimit);
            REQUIRE(algo.IncompleteGeneration().has_value());
            auto const& progress = *algo.IncompleteGeneration();
            CHECK(progress.Reason == DiscardReason::ResourceLimit);
            CHECK(progress.AttemptCapExceeded);
            CHECK(progress.Completed < progress.PoolSize);
            CHECK(progress.Attempted > cap);
            CHECK(progress.Attempted <= cap + threads);
        }
    };

    auto const makeGenerator = [&](EvaluatorBase& evaluator, SelectorBase const& fem, SelectorBase const& mal) {
        auto g = std::make_unique<OffspringSelectionGenerator>(&evaluator, &crossover, &mutator, &fem, &mal, nullptr);
        g->MaxSelectionPressure(pressure);
        g->ComparisonFactor(1.0);
        return g;
    };

    if (algorithm == Algorithm::GP) {
        auto generator = makeGenerator(errorEvaluator, female, male);
        Operon::ComparisonCallback const comp = Operon::SingleObjectiveComparison { 0 };
        Operon::KeepBestReinserter reinserter { comp, /*eliteCount=*/1 };
        Operon::GeneticProgrammingAlgorithm gp { config, &problem, &initializer, &coeffInit, generator.get(),
            &reinserter };
        check(gp, *generator);
    } else {
        auto generator = makeGenerator(multiEvaluator, female, male);
        Operon::CrowdedComparison const comp;
        Operon::KeepBestReinserter reinserter { comp };
        Operon::RankIntersectSorter sorter;
        Operon::NSGA2 nsga { config, &problem, &initializer, &coeffInit, generator.get(), &reinserter, &sorter };
        check(nsga, *generator);
    }
}

} // namespace Operon::Test
