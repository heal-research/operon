// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "checkpoint_helpers.hpp"
#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/tree.hpp"
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

// With the transposition cache on and CacheVisibility::Committed, a run does not depend on the number of threads (no
// time limit or stop request cuts a generation here): the final population, its fitness, the worker RNG states, the generation
// count and the evaluation and cache counters are identical for 1, 4 and 16 threads and between repeated runs. A
// generation that is cut short leaves no trace in the cache (Zobrist::AbortGeneration).
namespace Operon::Test {
namespace {

    using DTable = DispatchTable<Operon::Scalar>;

    enum class Algorithm { GP, NSGA2 };

    constexpr std::size_t PopulationSize = 40;
    constexpr std::size_t Generations = 8;
    constexpr std::size_t TrainRows = 60;
    constexpr std::size_t MaxLength = 14;
    constexpr std::size_t MaxDepth = 6;
    constexpr std::size_t Seed = 4242;

    struct Setup {
        Algorithm Algo;
        std::string Generator;
        bool LocalSearch;
        CacheVisibility Visibility;
        std::size_t MaxAge;
        std::size_t Threads;
        // Every individual has one of a handful of structures and neither crossover nor mutation changes a structure
        // (mutation only moves coefficients, and the structures differ in length, so fitness varies between
        // individuals and between structures, which offspring selection needs to accept children). From the second
        // generation on every offspring is a structure the cache holds: cache hits are guaranteed by construction,
        // not by the random sequence of the standard library.
        bool Duplicates { false };
        std::size_t GenerationLimit { Generations };
        bool ExtraObjective {
            false
        }; // NSGA2: a third objective (tree depth), for resuming with another objective count
    };

    // Returns a copy of one of a few fixed trees, chosen by the slot's random engine (raw output, not a standard
    // library distribution, so the choice is the same on every platform). Every initial individual then has one of a
    // handful of structures; they differ in length, so the objectives vary between structures.
    class FixedStructureInitializer final : public TreeInitializerBase {
        std::vector<Tree> trees_;

    public:
        explicit FixedStructureInitializer(std::vector<Tree> trees) : trees_(std::move(trees)) {}
        auto operator()(Operon::RandomGenerator& random) const -> Tree override
        {
            return trees_[static_cast<std::size_t>(random() % trees_.size())];
        }
    };

    struct IdentityCrossover final : CrossoverBase {
        auto operator()(Operon::RandomGenerator& /*random*/, Tree const& lhs, Tree const& /*rhs*/) const
            -> Tree override
        {
            return lhs;
        }
    };

    // Everything that must not depend on the thread count. Optimizer wall time is deliberately absent.
    struct Outcome {
        std::size_t Generation {};
        std::vector<std::uint64_t> Population; // genotype, coefficients, fitness, rank and distance of every parent
        std::vector<std::uint64_t> Rngs; // state of every worker RNG
        std::array<std::size_t, 7> Counters {}; // calls, residual and Jacobian work by phase, cache hits and lookups
        std::string Cache; // the transposition cache: every entry with its stamp, and the clock (as a checkpoint)

        [[nodiscard]] auto operator==(Outcome const&) const -> bool = default;
    };

    template <typename Algo> auto Collect(Algo const& algo) -> Outcome
    {
        auto const stats = algo.DetailedStats();
        Serialization::Checkpoint cacheOnly;
        if (auto const* cache = algo.GetGenerator()->Cache()) {
            cacheOnly.Cache = Serialization::ToCacheCheckpoint(cache->Export());
        }
        return { algo.Generation(), Signature(Serialized(algo.Parents())), Signature(algo.WorkerRngs()),
            { stats.Calls, stats.ResidualScoring, stats.ResidualOptimizer, stats.JacobianScoring,
                stats.JacobianOptimizer, stats.CacheHits.value_or(0), stats.CacheLookups.value_or(0) },
            Serialization::ToBeve(cacheOnly) };
    }

    // `duplicates`: the structures are few and fixed (see Setup::Duplicates). Offspring selection then accepts a child
    // whenever its parents differ in fitness (a huge negative comparison factor makes the threshold exceed any
    // child), so a cache hit that is worse than the parents cannot make it retry forever; parents of one structure
    // with equal fitness still need a child that is no worse, and the population holds parents of several
    // structures with different fitness.
    auto MakeGenerator(std::string const& name, EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut,
        SelectorBase& fem, SelectorBase& male, CoefficientOptimizer const* opt, bool duplicates)
        -> std::unique_ptr<OffspringGeneratorBase>
    {
        if (name == "basic") {
            return std::make_unique<BasicOffspringGenerator>(&eval, &cx, &mut, &fem, &male, opt);
        }
        if (name == "os") {
            auto g = std::make_unique<OffspringSelectionGenerator>(&eval, &cx, &mut, &fem, &male, opt);
            g->MaxSelectionPressure(10);
            g->ComparisonFactor(duplicates ? -1e30 : 0);
            return g;
        }
        if (name == "brood") {
            auto g = std::make_unique<BroodOffspringGenerator>(&eval, &cx, &mut, &fem, &male, opt);
            g->BroodSize(3);
            return g;
        }
        auto g = std::make_unique<PolygenicOffspringGenerator>(&eval, &cx, &mut, &fem, &male, opt);
        g->PolygenicSize(3);
        return g;
    }

    // Builds everything a run needs from `setup` and hands the algorithm, its master RNG and the transposition cache to
    // `fn`, which runs it (possibly warm-started from a checkpoint) and returns what it observed.
    template <typename Fn> auto WithAlgorithm(Dataset& ds, Setup const& setup, Fn&& fn)
    {
        Operon::Problem problem { gsl::not_null<Dataset*>(&ds) };
        problem.SetTrainingRange(0, static_cast<int>(TrainRows));
        problem.SetTestRange(static_cast<int>(TrainRows), static_cast<int>(2 * TrainRows));
        problem.SetTarget("Y");
        problem.ConfigurePrimitiveSet(PrimitiveSet::Arithmetic);
        auto const inputs = std::vector<Operon::Hash>(problem.GetInputs().begin(), problem.GetInputs().end());

        Operon::BalancedTreeCreator creator { &problem.GetPrimitiveSet(), inputs, 0.0, MaxLength };
        Operon::UniformTreeInitializer treeInit { &creator };
        treeInit.ParameterizeDistribution(std::size_t { 1 }, MaxLength);
        treeInit.SetMinDepth(1);
        treeInit.SetMaxDepth(MaxDepth);
        Operon::NormalCoefficientInitializer coeffInit;
        coeffInit.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });

        Operon::SubtreeCrossover crossover { 0.9, MaxDepth, MaxLength };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
        onePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        Operon::ChangeVariableMutation changeVar { problem.GetInputs() };
        Operon::ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
        Operon::ReplaceSubtreeMutation replaceSubtree { &creator, &coeffInit, MaxDepth, MaxLength };
        Operon::InsertSubtreeMutation insertSubtree { &creator, &coeffInit, MaxDepth, MaxLength };
        Operon::RemoveSubtreeMutation removeSubtree { &creator, &coeffInit, MaxDepth };
        Operon::MultiMutation mutator;
        for (Operon::MutatorBase* m : std::initializer_list<Operon::MutatorBase*> {
                 &onePoint, &changeVar, &changeFunc, &replaceSubtree, &insertSubtree, &removeSubtree }) {
            mutator.Add(m, 1.0);
        }

        // Duplicates: one structure for everybody, preserved by the operators (see Setup::Duplicates)
        Operon::RandomGenerator structureRng { Seed };
        std::vector<Tree> structures;
        for (std::size_t length = 3; length <= MaxLength; length += 3) {
            structures.push_back(creator(structureRng, length, 1, MaxDepth));
        }
        FixedStructureInitializer fixedInit { std::move(structures) };
        IdentityCrossover identityCrossover;
        Operon::TreeInitializerBase const& initializer
            = setup.Duplicates ? static_cast<Operon::TreeInitializerBase const&>(fixedInit) : treeInit;
        Operon::CrossoverBase& cx
            = setup.Duplicates ? static_cast<Operon::CrossoverBase&>(identityCrossover) : crossover;
        Operon::MultiMutation coefficientMutator; // one-point mutation changes a coefficient, never the structure
        coefficientMutator.Add(&onePoint, 1.0);
        Operon::MutatorBase& mut = setup.Duplicates ? static_cast<Operon::MutatorBase&>(coefficientMutator) : mutator;

        DTable dtable;
        Operon::GeneticAlgorithmConfig config;
        config.Generations = setup.GenerationLimit;
        config.Evaluations = std::size_t { 1 } << 40U; // never binds
        config.Iterations = setup.LocalSearch ? 3 : 0;
        config.PopulationSize = PopulationSize;
        config.PoolSize = PopulationSize;
        config.Seed = Seed;
        config.LocalSearchProbability = 1.0;
        config.LamarckianProbability = 1.0;

        Operon::RandomGenerator cacheRng { Seed };
        Operon::Zobrist cache { cacheRng, static_cast<int>(MaxLength), problem.GetInputs(),
            { .MaxAge = setup.MaxAge, .Visibility = setup.Visibility } };
        config.Cache = &cache;

        Operon::LevenbergMarquardtOptimizer<DTable, Operon::OptimizerType::Eigen> optimizer { &dtable, &problem };
        optimizer.SetIterations(config.Iterations);
        Operon::CoefficientOptimizer const coeffOptimizer { &optimizer };

        Operon::Evaluator<DTable> errorEvaluator { &problem, &dtable };
        Operon::RandomGenerator random { Seed };

        if (setup.Algo == Algorithm::GP) {
            Operon::ComparisonCallback const comp = Operon::SingleObjectiveComparison { 0 };
            Operon::TournamentSelector fem { comp };
            Operon::TournamentSelector male { comp };
            auto generator
                = MakeGenerator(setup.Generator, errorEvaluator, cx, mut, fem, male, &coeffOptimizer, setup.Duplicates);
            Operon::KeepBestReinserter reinserter { comp, /*eliteCount=*/1 };
            Operon::GeneticProgrammingAlgorithm gp { config, &problem, &initializer, &coeffInit, generator.get(),
                &reinserter };
            return fn(gp, random);
        }

        Operon::TreePropertyEvaluator lengthEvaluator { &problem,
            [](Operon::Tree const& tree) { return static_cast<Operon::Scalar>(tree.Length()); },
            static_cast<Operon::Scalar>(MaxLength) };
        Operon::MultiEvaluator evaluator { &problem };
        evaluator.Add(&errorEvaluator);
        evaluator.Add(&lengthEvaluator);
        Operon::TreePropertyEvaluator depthEvaluator { &problem,
            [](Operon::Tree const& tree) { return static_cast<Operon::Scalar>(tree.Depth()); },
            static_cast<Operon::Scalar>(MaxDepth) };
        if (setup.ExtraObjective) {
            evaluator.Add(&depthEvaluator);
        }
        Operon::CrowdedComparison const comp;
        Operon::TournamentSelector fem { comp };
        Operon::TournamentSelector male { comp };
        auto generator
            = MakeGenerator(setup.Generator, evaluator, cx, mut, fem, male, &coeffOptimizer, setup.Duplicates);
        Operon::KeepBestReinserter reinserter { comp };
        Operon::RankIntersectSorter sorter;
        Operon::NSGA2 nsga { config, &problem, &initializer, &coeffInit, generator.get(), &reinserter, &sorter };
        return fn(nsga, random);
    }

    auto Run(Dataset& ds, Setup const& setup) -> Outcome
    {
        return WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> Outcome {
            algo.Run(random, nullptr, setup.Threads);
            return Collect(algo);
        });
    }

    // Runs to `checkpointAt`, serializes a checkpoint there, and finishes the run in a second, freshly built set of
    // objects that gets nothing but the serialized bytes (what `--resume` does in a new process).
    auto RunResumed(Dataset& ds, Setup setup, std::size_t checkpointAt) -> Outcome
    {
        auto const total = setup.GenerationLimit;
        setup.GenerationLimit = checkpointAt;
        std::string bytes;
        WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> int {
            algo.Run(random, nullptr, setup.Threads);
            bytes = CaptureCheckpoint(algo, random);
            return 0;
        });
        setup.GenerationLimit = total;
        return WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> Outcome {
            RestoreCheckpoint(algo, random, bytes);
            algo.Run(random, nullptr, setup.Threads, /*warmStart=*/true);
            return Collect(algo);
        });
    }

    // ── Generations cut short ────────────────────────────────────────────────────────────────────────────────────

    // Offspring generation through the generator's own cache (OffspringGeneratorBase::Generate), with hooks:
    // OnCall(attempt, call) runs before every invocation, `attempt` being the 1-based generation attempt (counted by
    // Prepare()) and `call` the 1-based invocation within it; OnChild(attempt, hash, hit) runs after it and tells
    // whether the cache served the structure. Nothing here depends on timing.
    class HookedGenerator final : public OffspringGeneratorBase {
    public:
        using OffspringGeneratorBase::OffspringGeneratorBase;

        std::function<void(std::size_t, std::size_t)> OnCall;
        std::function<void(std::size_t, Operon::Hash, bool)> OnChild;

        auto Prepare(Operon::Span<Individual const> pop) const -> void override
        {
            OffspringGeneratorBase::Prepare(pop);
            ++attempt_;
            call_ = 0;
        }

        auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
            double pLamarck, Operon::Span<Operon::Scalar> buf, OptimizerAccounting& accounting) const
            -> std::optional<Individual> override
        {
            auto const attempt = attempt_.load(std::memory_order_relaxed);
            auto const call = call_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (OnCall) {
                OnCall(attempt, call);
            }
            auto const* cache = Cache();
            auto const hitsBefore = cache != nullptr ? cache->Hits() : 0;
            auto res = Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, accounting);
            if (OnChild && cache != nullptr && res.Child) {
                OnChild(attempt, cache->ComputeHash(res.Child->Genotype), cache->Hits() != hitsBefore);
            }
            return std::move(res.Child);
        }

    private:
        mutable std::atomic_size_t attempt_ { 0 };
        mutable std::atomic_size_t call_ { 0 };
    };

    constexpr std::size_t CutPopulation = 16;

    struct CutRun {
        Dataset Ds { [] {
            std::vector<std::vector<Operon::Scalar>> data(2, std::vector<Operon::Scalar>(10));
            std::iota(data[0].begin(), data[0].end(), Operon::Scalar { 1 });
            for (std::size_t i = 0; i < 10; ++i) {
                data[1][i] = data[0][i] * Operon::Scalar { 2 };
            }
            return Dataset({ "X1", "Y" }, data);
        }() };
        Operon::Problem Problem { gsl::not_null<Dataset*>(&Ds) };
        Operon::PrimitiveSet Pset;
        std::vector<Operon::Hash> Vars { Ds.GetVariable("X1")->Hash };

        DTable Dtable;
        Operon::Evaluator<DTable> Evaluator { &Problem, &Dtable };
        Operon::SubtreeCrossover Crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
        Operon::MultiMutation Mutator;
        Operon::TournamentSelector FemSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::TournamentSelector MaleSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::RandomGenerator CacheRng { 1234 };
        Operon::Zobrist Cache;
        HookedGenerator Generator { &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
        Operon::KeepBestReinserter KeepBest { Operon::SingleObjectiveComparison { 0 } };
        Operon::BalancedTreeCreator Creator { &Pset, Vars, 0.0, 10 };
        Operon::UniformTreeInitializer TreeInit { &Creator };
        Operon::UniformCoefficientInitializer CoeffInit;
        Operon::DeductiveSorter Sorter;
        Operon::GeneticAlgorithmConfig Config;

        explicit CutRun(CacheVisibility visibility)
            : Cache(CacheRng, /*maxLength=*/20, Vars, ZobristOptions { .Visibility = visibility })
        {
            Pset.SetConfig(PrimitiveSet::Arithmetic);
            Problem.SetTrainingRange({ 0, 10 });
            Problem.SetTestRange({ 0, 10 });
            Problem.SetTarget("Y");
            TreeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
            OnePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            Mutator.Add(&OnePoint, 1.0);
            Config.PopulationSize = CutPopulation;
            Config.PoolSize = CutPopulation;
            Config.Generations = 3;
            Config.Cache = &Cache;
        }
    };

    auto MakeGp(CutRun& f) -> Operon::GeneticProgrammingAlgorithm
    {
        return { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator, &f.KeepBest };
    }

    auto MakeNsga2(CutRun& f) -> Operon::NSGA2
    {
        return { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator, &f.KeepBest, &f.Sorter };
    }

    auto Lookup(Zobrist const& cache, Operon::Hash hash) -> std::optional<Operon::Scalar>
    {
        Operon::Vector<Operon::Scalar> out(1);
        if (cache.TryGet(hash, out)) { // outside any slot: sees every entry of the table
            return out[0];
        }
        return std::nullopt;
    }

    // Generation 1 completes, generation attempt 2 is cut by a stop request after 5 generator calls and attempt 3 (the
    // warm resume) completes generations 2 and 3. Attempt 2 plants an entry with fitness -10 (a value no real
    // fitness has), attempt 3 one with +10 for the same hash.
    template <typename Algo, typename Make>
    void CheckCutGeneration(CacheVisibility visibility, std::size_t threads, Make make)
    {
        constexpr Operon::Hash planted { 0xBADBADU };
        CutRun f { visibility };
        Algo algo = make(f);
        Operon::RandomGenerator rng { 7 };

        std::mutex mutex;
        std::vector<Operon::Hash> fresh; // structures inserted for the first time by attempt 2
        std::size_t sizeBeforeCut = 0;
        f.Generator.OnCall = [&](std::size_t attempt, std::size_t call) {
            if (attempt == 2 && call == 1) {
                sizeBeforeCut = f.Cache.Size();
                f.Cache.Insert(planted, { Operon::Scalar { -10 } });
            } else if (attempt == 2 && call == 5) {
                algo.RequestStop();
            } else if (attempt == 3 && call == 1) {
                f.Cache.Insert(planted, { Operon::Scalar { 10 } });
            }
        };
        f.Generator.OnChild = [&](std::size_t attempt, Operon::Hash hash, bool hit) {
            if (attempt == 2 && !hit) {
                std::scoped_lock const lock { mutex };
                fresh.push_back(hash);
            }
        };

        algo.Run(rng, nullptr, threads);
        REQUIRE(algo.Generation() == 1);
        REQUIRE(algo.IncompleteGeneration().has_value());

        auto const kept = Lookup(f.Cache, planted);
        if (visibility == CacheVisibility::Committed) {
            CHECK_FALSE(kept.has_value());
            if (threads == 1) { // the hit/miss bookkeeping of OnChild is exact with one worker only
                // `fresh` holds whatever structures the cut generation inserted; which ones that is depends on the
                // random sequence of the standard library, so the planted entry above is what makes the rollback
                // check non-vacuous on every platform. Size equality covers any other entry.
                for (auto const hash : fresh) {
                    CHECK_FALSE(Lookup(f.Cache, hash).has_value());
                }
                CHECK(f.Cache.Size() == sizeBeforeCut);
            }
        } else {
            REQUIRE(kept.has_value()); // Immediate: nothing is rolled back
            CHECK(*kept == Operon::Scalar { -10 });
        }

        algo.Run(rng, nullptr, threads, /*warmStart=*/true);
        CHECK(algo.Generation() == 3);
        CHECK_FALSE(algo.IncompleteGeneration().has_value());
        auto const resumed = Lookup(f.Cache, planted);
        REQUIRE(resumed.has_value());
        // Committed: the retry's +10 is the only entry of generation 2, the discarded -10 never took part in the
        // minimum. Immediate: the first writer, the discarded attempt, keeps the entry.
        CHECK(*resumed == (visibility == CacheVisibility::Committed ? Operon::Scalar { 10 } : Operon::Scalar { -10 }));
    }

    // Generation 1 completes and is reported; the first generator call of attempt 2 plants an entry (fitness -10, a
    // value no real fitness has) and throws, the way a task that has already inserted and then fails does. What is
    // guaranteed afterwards, and all this tests: the exception reaches the caller, the cache holds none of the failed
    // generation's entries (Committed; Immediate keeps them), Generation() is the last committed generation, the
    // algorithm object refuses a warm restart (its population may be half-way through a generation: it is not
    // resumable after a task exception), and a new algorithm object can reuse the cache. A retry that continues
    // exactly is NOT promised.
    template <typename Algo, typename Make>
    void CheckThrowingGeneration(CacheVisibility visibility, std::size_t threads, Make make)
    {
        constexpr Operon::Hash planted { 0xBADBADU };
        CutRun f { visibility };
        Algo algo = make(f);
        Operon::RandomGenerator rng { 7 };

        std::size_t sizeAfterGeneration1 = 0;
        f.Generator.OnCall = [&](std::size_t attempt, std::size_t call) {
            if (attempt == 2 && call == 1) {
                f.Cache.Insert(planted, { Operon::Scalar { -10.F } });
                throw std::runtime_error { "generator failed" };
            }
        };
        auto const report = [&]() -> bool {
            if (algo.Generation() == 1) {
                sizeAfterGeneration1 = f.Cache.Size();
            }
            return false;
        };

        REQUIRE_THROWS_AS(algo.Run(rng, report, threads), std::runtime_error);
        CHECK(algo.Generation() == 1);
        REQUIRE(sizeAfterGeneration1 > 0);

        auto const kept = Lookup(f.Cache, planted);
        if (visibility == CacheVisibility::Committed) {
            CHECK_FALSE(kept.has_value());
            CHECK(f.Cache.Size() == sizeAfterGeneration1); // nothing of the failed generation is left
        } else {
            REQUIRE(kept.has_value()); // Immediate: nothing is rolled back
            CHECK(*kept == Operon::Scalar { -10 });
        }

        // the object is not resumable ...
        CHECK_THROWS_AS(algo.Run(rng, nullptr, threads, /*warmStart=*/true), std::logic_error);

        // ... but the cache is usable by a new one
        f.Generator.OnCall = nullptr;
        Algo fresh = make(f);
        Operon::RandomGenerator freshRng { 7 };
        CHECK_NOTHROW(fresh.Run(freshRng, nullptr, threads));
        CHECK(fresh.Generation() == f.Config.Generations);
    }

    // An exception in the report callback after generation 1 was committed must not roll that generation back: its
    // entries (and the generation count) stay.
    template <typename Algo, typename Make> void CheckThrowingReport(std::size_t threads, Make make)
    {
        constexpr Operon::Hash planted { 0xBADBADU };
        CutRun f { CacheVisibility::Committed };
        Algo algo = make(f);
        Operon::RandomGenerator rng { 7 };

        f.Generator.OnCall = [&](std::size_t attempt, std::size_t call) {
            if (attempt == 1 && call == 1) { // the first offspring generation, which is committed and then reported
                f.Cache.Insert(planted, { Operon::Scalar { 5 } });
            }
        };
        auto const report = [&]() -> bool {
            if (algo.Generation() == 1) {
                throw std::runtime_error { "report failed" };
            }
            return false;
        };

        REQUIRE_THROWS_AS(algo.Run(rng, report, threads), std::runtime_error);
        CHECK(algo.Generation() == 1); // committed before the callback ran
        auto const kept = Lookup(f.Cache, planted);
        REQUIRE(kept.has_value());
        CHECK(*kept == Operon::Scalar { 5 });
    }

    // Fails on demand: creating a population is the first thing a cold Run() does.
    class FlakyInitializer final : public TreeInitializerBase {
        TreeInitializerBase const* inner_;

    public:
        explicit FlakyInitializer(TreeInitializerBase const* inner) : inner_(inner) {}
        mutable std::atomic_bool Fail { false };
        auto operator()(Operon::RandomGenerator& random) const -> Tree override
        {
            if (Fail.load()) {
                throw std::runtime_error { "initializer failed" };
            }
            return (*inner_)(random);
        }
    };

    auto MakeGpWith(CutRun& f, TreeInitializerBase const& init) -> Operon::GeneticProgrammingAlgorithm
    {
        return { f.Config, &f.Problem, &init, &f.CoeffInit, &f.Generator, &f.KeepBest };
    }

    auto MakeNsga2With(CutRun& f, TreeInitializerBase const& init) -> Operon::NSGA2
    {
        return { f.Config, &f.Problem, &init, &f.CoeffInit, &f.Generator, &f.KeepBest, &f.Sorter };
    }

    // What a warm start may continue: an object whose last Run() completed, or one that a checkpoint restore marked
    // fitted. Not a never-run object and not one whose last Run() threw, wherever it threw and however the Run()
    // before it ended. (Scenarios, not Catch sections: the function runs once per algorithm and thread count within
    // one test case, and Catch would skip the sections of the second call.)
    enum class Scenario {
        NeverRun,
        CompletedRun,
        InitException,
        InitExceptionAfterCompletedRun,
        ReportExceptionAtGenerationZero,
        ReportExceptionAtGenerationZeroAfterCompletedRun,
        ReportExceptionInLaterGeneration,
        ColdRunAfterException,
    };

    template <typename Algo, typename Make> void CheckResumability(Scenario scenario, std::size_t threads, Make make)
    {
        CutRun f { CacheVisibility::Committed };
        FlakyInitializer init { &f.TreeInit };
        Algo algo = make(f, init);
        Operon::RandomGenerator rng { 7 };
        auto const thrownAt = [&](std::size_t generation) {
            return [&algo, generation]() -> bool {
                if (algo.Generation() == generation) {
                    throw std::runtime_error { "report failed" };
                }
                return false;
            };
        };
        auto const warmStartRejected
            = [&] { CHECK_THROWS_AS(algo.Run(rng, nullptr, threads, /*warmStart=*/true), std::logic_error); };

        switch (scenario) {
        case Scenario::NeverRun:
            CHECK_FALSE(algo.IsFitted());
            warmStartRejected();
            CHECK_FALSE(algo.IsFitted());
            break;
        case Scenario::CompletedRun: // a completed run can be warm started, again and again
            algo.Run(rng, nullptr, threads);
            REQUIRE(algo.IsFitted());
            REQUIRE(algo.Generation() == f.Config.Generations);
            CHECK_NOTHROW(algo.Run(rng, nullptr, threads, /*warmStart=*/true));
            CHECK(algo.IsFitted());
            CHECK(algo.Generation() == f.Config.Generations);
            CHECK_NOTHROW(algo.Run(rng, nullptr, threads, /*warmStart=*/true));
            break;
        case Scenario::InitException: // while the population is created
            init.Fail = true;
            CHECK_THROWS_AS(algo.Run(rng, nullptr, threads), std::runtime_error);
            CHECK_FALSE(algo.IsFitted());
            warmStartRejected();
            break;
        case Scenario::InitExceptionAfterCompletedRun:
            algo.Run(rng, nullptr, threads);
            REQUIRE(algo.IsFitted());
            init.Fail = true;
            CHECK_THROWS_AS(algo.Run(rng, nullptr, threads), std::runtime_error);
            CHECK_FALSE(algo.IsFitted()); // the earlier completed run does not make the object resumable
            warmStartRejected();
            break;
        case Scenario::ReportExceptionAtGenerationZero:
            CHECK_THROWS_AS(algo.Run(rng, thrownAt(0), threads), std::runtime_error);
            CHECK(algo.Generation() == 0);
            CHECK_FALSE(algo.IsFitted());
            warmStartRejected();
            break;
        case Scenario::ReportExceptionAtGenerationZeroAfterCompletedRun:
            algo.Run(rng, nullptr, threads);
            REQUIRE(algo.IsFitted());
            CHECK_THROWS_AS(algo.Run(rng, thrownAt(0), threads), std::runtime_error);
            CHECK_FALSE(algo.IsFitted());
            warmStartRejected();
            break;
        case Scenario::ReportExceptionInLaterGeneration:
            CHECK_THROWS_AS(algo.Run(rng, thrownAt(2), threads), std::runtime_error);
            CHECK(algo.Generation() == 2);
            CHECK_FALSE(algo.IsFitted());
            warmStartRejected();
            break;
        case Scenario::ColdRunAfterException: // a new cold run completes and makes the object resumable again
            CHECK_THROWS_AS(algo.Run(rng, thrownAt(2), threads), std::runtime_error);
            REQUIRE_NOTHROW(algo.Run(rng, nullptr, threads));
            CHECK(algo.IsFitted());
            CHECK_NOTHROW(algo.Run(rng, nullptr, threads, /*warmStart=*/true));
            break;
        }
    }

    // The generation limit is `Generation() >= limit`: a population that a checkpoint brought past the limit runs no
    // further generation (it would never meet `==` again). The report callback ends the run if the loop wrongly
    // went on.
    template <typename Algo, typename Make> void CheckLimitBelowCheckpoint(Make make)
    {
        std::string bytes;
        {
            CutRun f { CacheVisibility::Committed };
            f.Config.Generations = 3;
            Algo algo = make(f);
            Operon::RandomGenerator rng { 7 };
            algo.Run(rng, nullptr, 1);
            REQUIRE(algo.Generation() == 3);
            bytes = CaptureCheckpoint(algo, rng);
        }
        CutRun f { CacheVisibility::Committed };
        f.Config.Generations = 2;
        Algo algo = make(f);
        Operon::RandomGenerator rng { 7 };
        RestoreCheckpoint(algo, rng, bytes);
        REQUIRE(algo.Generation() == 3);
        algo.Run(rng, [&]() -> bool { return algo.Generation() > 3; }, 1, /*warmStart=*/true);
        CHECK(algo.Generation() == 3);
        CHECK(algo.IsFitted());
    }

} // namespace

TEST_CASE("Committed cache visibility makes a run independent of the thread count", "[cache][determinism]")
{
    auto const algorithm = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    auto const generator = GENERATE(as<std::string> {}, "basic", "os", "brood", "poly");
    auto const localSearch = GENERATE(true, false);
    auto const maxAge = GENERATE(std::size_t { 0 }, std::size_t { 3 });
    auto const duplicates = GENERATE(false, true);
    CAPTURE(algorithm == Algorithm::GP ? "gp" : "nsga2", generator, localSearch, maxAge, duplicates);

    if (duplicates && generator == "os" && maxAge > 0) {
        // Offspring selection ends a generation by the evaluator calls it makes, and a cache hit makes none. Once the
        // population has converged on one structure, an expired entry is re-evaluated to a value that can be worse
        // than every parent, and the slot then retries on its own cached copies for ever: nothing raises the
        // selection pressure. (Without expiry the cached value is the parents' own.)
        SUCCEED("offspring selection can retry for ever on cache hits once entries expire");
        return;
    }

    Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
    auto const reference
        = Run(ds, { algorithm, generator, localSearch, CacheVisibility::Committed, maxAge, /*Threads=*/1, duplicates });

    REQUIRE(reference.Generation == Generations);
    REQUIRE(reference.Counters[6] > 0); // the cache was consulted
    if (duplicates) {
        // Hits are guaranteed by construction here (see Setup::Duplicates). With random operators a hit depends on
        // the random sequence of the standard library, so the random cells do not require one.
        REQUIRE(reference.Counters[5] > 0);
    }

    for (auto const threads : { std::size_t { 4 }, std::size_t { 16 }, std::size_t { 16 } }) {
        CAPTURE(threads);
        auto const outcome
            = Run(ds, { algorithm, generator, localSearch, CacheVisibility::Committed, maxAge, threads, duplicates });
        CHECK(outcome.Generation == reference.Generation);
        CHECK(Same(outcome.Counters, reference.Counters));
        CHECK(Same(outcome.Population, reference.Population));
        CHECK(Same(outcome.Rngs, reference.Rngs));
        CHECK(Same(outcome.Cache, reference.Cache)); // the final cache: entries, stamps and clock
    }
}

TEST_CASE(
    "A run resumed from a serialized checkpoint continues exactly like the uninterrupted run", "[cache][checkpoint]")
{
    // The checkpoint carries the transposition cache: without its entries a resumed run takes different cache
    // decisions and its population differs from the one an uninterrupted run reaches.
    auto const algorithm = GENERATE(Algorithm::GP, Algorithm::NSGA2);
    auto const generator = GENERATE(as<std::string> {}, "basic", "os", "brood", "poly");
    auto const maxAge = GENERATE(std::size_t { 0 }, std::size_t { 3 });
    auto const checkpointAt = GENERATE(std::size_t { 2 }, std::size_t { 3 });
    auto const mode = GENERATE(std::pair { CacheVisibility::Committed, std::size_t { 1 } },
        std::pair { CacheVisibility::Committed, std::size_t { 4 } },
        std::pair { CacheVisibility::Immediate, std::size_t { 1 } });
    auto const [visibility, threads] = mode;
    CAPTURE(algorithm == Algorithm::GP ? "gp" : "nsga2", generator, maxAge, checkpointAt,
        visibility == CacheVisibility::Committed, threads);

    Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
    Setup setup { algorithm, generator, /*LocalSearch=*/true, visibility, maxAge, threads };
    setup.GenerationLimit = 6;
    auto const uninterrupted = Run(ds, setup);
    auto const resumed = RunResumed(ds, setup, checkpointAt);

    REQUIRE(uninterrupted.Generation == 6);
    CHECK(resumed.Generation == uninterrupted.Generation);
    CHECK(Same(resumed.Population, uninterrupted.Population));
    CHECK(Same(resumed.Rngs, uninterrupted.Rngs));
    CHECK(Same(resumed.Cache, uninterrupted.Cache));
}

TEST_CASE(
    "A resumed run without the cache entries of its checkpoint takes different cache decisions", "[cache][checkpoint]")
{
    // Non-vacuity of the test above: the same scenarios with the cache stripped from the checkpoint must be able to
    // differ from the uninterrupted run. Which settings do depends on the platform's random distributions, so it is
    // enough that one of them does.
    Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
    std::size_t differing = 0;
    for (auto const* generator : { "basic", "brood", "poly" }) {
        for (auto const algorithm : { Algorithm::GP, Algorithm::NSGA2 }) {
            Setup setup { algorithm, generator, /*LocalSearch=*/true, CacheVisibility::Committed, 0, 1 };
            setup.GenerationLimit = 6;
            auto const uninterrupted = Run(ds, setup);

            setup.GenerationLimit = 3;
            std::string bytes;
            WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> int {
                algo.Run(random, nullptr, setup.Threads);
                auto cp = Serialization::CheckpointFromBeve(CaptureCheckpoint(algo, random));
                REQUIRE(cp.has_value());
                REQUIRE(cp->Cache.has_value());
                REQUIRE_FALSE(cp->Cache->Keys.empty());
                cp->Cache.reset();
                bytes = Serialization::ToBeve(*cp);
                return 0;
            });
            setup.GenerationLimit = 6;
            auto const stripped = WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> Outcome {
                RestoreCheckpoint(algo, random, bytes);
                algo.Run(random, nullptr, setup.Threads, /*warmStart=*/true);
                return Collect(algo);
            });
            differing += stripped.Population != uninterrupted.Population ? 1 : 0;
        }
    }
    CHECK(differing > 0);
}

// What a warm resume keeps of a checkpoint's parents, with the checkpoint doctored so that it shows: fitness, rank and
// distance carry sentinel values that no evaluation or sort produces.
TEST_CASE("A warm resume keeps the checkpointed fitness, rank and distance only for an unchanged objective count",
    "[cache][checkpoint]")
{
    constexpr Operon::Scalar sentinelFitness = 123456;
    constexpr std::size_t sentinelRank = 999;
    constexpr Operon::Scalar sentinelDistance = 777;
    Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };

    auto const doctored = [&](Setup setup) {
        setup.GenerationLimit = 3;
        std::string bytes;
        WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> int {
            algo.Run(random, nullptr, setup.Threads);
            auto cp = Serialization::CheckpointFromBeve(CaptureCheckpoint(algo, random));
            REQUIRE(cp.has_value());
            for (auto& ind : cp->Population) {
                for (auto& f : ind.Fitness) {
                    f = sentinelFitness;
                }
                ind.Rank = sentinelRank;
                ind.Distance = sentinelDistance;
            }
            cp->Cache.reset(); // the cache is the subject of the next test; its entries have the old objective count
            bytes = Serialization::ToBeve(*cp);
            return 0;
        });
        return bytes;
    };
    // Resumes `bytes` into a run of `setup` that is at its generation limit already, so that only the warm start's
    // re-evaluation and re-sort happen, and returns the parents.
    auto const resume = [&](Setup setup, std::string const& bytes) {
        setup.GenerationLimit = 3;
        return WithAlgorithm(ds, setup, [&](auto& algo, Operon::RandomGenerator& random) -> std::vector<Individual> {
            RestoreCheckpoint(algo, random, bytes);
            algo.Run(random, nullptr, setup.Threads, /*warmStart=*/true);
            return { algo.Parents().begin(), algo.Parents().end() };
        });
    };

    for (auto const algorithm : { Algorithm::GP, Algorithm::NSGA2 }) {
        CAPTURE(algorithm == Algorithm::GP ? "gp" : "nsga2");
        Setup setup { algorithm, "basic", /*LocalSearch=*/false, CacheVisibility::Committed, 0, 1 };
        setup.GenerationLimit = 3;
        auto const bytes = doctored(setup);

        // the same objective count: everything the checkpoint holds is kept
        for (auto const& ind : resume(setup, bytes)) {
            CHECK(ind.Fitness[0] == sentinelFitness);
            if (algorithm == Algorithm::NSGA2) {
                CHECK(ind.Rank == sentinelRank);
                CHECK(ind.Distance == sentinelDistance);
            }
        }
    }

    // NSGA2 with a third objective: the re-evaluated fitness wins, and rank and distance are those of the sort of the
    // new objective space, not the checkpointed ones of the old one
    Setup two { Algorithm::NSGA2, "basic", /*LocalSearch=*/false, CacheVisibility::Committed, 0, 1 };
    auto const bytes = doctored(two);
    Setup three = two;
    three.ExtraObjective = true;
    auto const parents = resume(three, bytes);
    REQUIRE_FALSE(parents.empty());
    for (auto const& ind : parents) {
        CHECK(ind.Fitness.size() == 3);
        CHECK(ind.Fitness[0] != sentinelFitness);
        CHECK(ind.Rank != sentinelRank);
        CHECK(ind.Distance != sentinelDistance);
    }
}

TEST_CASE("A checkpointed cache is rejected when the evaluator has another objective count", "[cache][checkpoint]")
{
    Dataset ds { "./data/Poly-10.csv", /*hasHeader=*/true };
    Setup two { Algorithm::NSGA2, "basic", /*LocalSearch=*/false, CacheVisibility::Committed, 0, 1 };
    two.GenerationLimit = 3;
    std::string bytes;
    WithAlgorithm(ds, two, [&](auto& algo, Operon::RandomGenerator& random) -> int {
        algo.Run(random, nullptr, two.Threads);
        bytes = CaptureCheckpoint(algo, random);
        return 0;
    });
    Setup three = two;
    three.ExtraObjective = true;
    WithAlgorithm(ds, three, [&](auto& algo, Operon::RandomGenerator& random) -> int {
        CHECK_THROWS_AS(RestoreCheckpoint(algo, random, bytes), std::invalid_argument);
        return 0;
    });
}

TEST_CASE("Committed: a generation cut short leaves nothing in the cache, also after a warm resume",
    "[cache][partial-generation]")
{
    for (auto const threads : { std::size_t { 1 }, std::size_t { 4 } }) {
        CAPTURE(threads);
        CheckCutGeneration<Operon::GeneticProgrammingAlgorithm>(CacheVisibility::Committed, threads, MakeGp);
        CheckCutGeneration<Operon::NSGA2>(CacheVisibility::Committed, threads, MakeNsga2);
    }
}

TEST_CASE("Committed: a generation that throws leaves nothing in the cache", "[cache][partial-generation]")
{
    for (auto const threads : { std::size_t { 1 }, std::size_t { 4 }, std::size_t { 16 } }) {
        CAPTURE(threads);
        CheckThrowingGeneration<Operon::GeneticProgrammingAlgorithm>(CacheVisibility::Committed, threads, MakeGp);
        CheckThrowingGeneration<Operon::NSGA2>(CacheVisibility::Committed, threads, MakeNsga2);
    }
}

TEST_CASE("Immediate: the entries of a generation that throws are kept", "[cache][partial-generation]")
{
    CheckThrowingGeneration<Operon::GeneticProgrammingAlgorithm>(CacheVisibility::Immediate, 4, MakeGp);
    CheckThrowingGeneration<Operon::NSGA2>(CacheVisibility::Immediate, 4, MakeNsga2);
}

TEST_CASE("Committed: an exception in the report callback keeps the generation that was just committed",
    "[cache][partial-generation]")
{
    for (auto const threads : { std::size_t { 1 }, std::size_t { 4 } }) {
        CAPTURE(threads);
        CheckThrowingReport<Operon::GeneticProgrammingAlgorithm>(threads, MakeGp);
        CheckThrowingReport<Operon::NSGA2>(threads, MakeNsga2);
    }
}

TEST_CASE("Immediate: the entries of a generation cut short are kept", "[cache][partial-generation]")
{
    CheckCutGeneration<Operon::GeneticProgrammingAlgorithm>(CacheVisibility::Immediate, 1, MakeGp);
    CheckCutGeneration<Operon::NSGA2>(CacheVisibility::Immediate, 1, MakeNsga2);
}

TEST_CASE("A warm start needs a fitted object: completed or restored, never after an exception or never run",
    "[cache][partial-generation][resume]")
{
    for (auto const scenario :
        { Scenario::NeverRun, Scenario::CompletedRun, Scenario::InitException, Scenario::InitExceptionAfterCompletedRun,
            Scenario::ReportExceptionAtGenerationZero, Scenario::ReportExceptionAtGenerationZeroAfterCompletedRun,
            Scenario::ReportExceptionInLaterGeneration, Scenario::ColdRunAfterException }) {
        for (auto const threads : { std::size_t { 1 }, std::size_t { 4 } }) {
            CAPTURE(static_cast<int>(scenario), threads);
            CheckResumability<Operon::GeneticProgrammingAlgorithm>(scenario, threads, MakeGpWith);
            CheckResumability<Operon::NSGA2>(scenario, threads, MakeNsga2With);
        }
    }
}

TEST_CASE("A checkpoint at a generation past the generation limit runs no further generation", "[cache][resume]")
{
    CheckLimitBelowCheckpoint<Operon::GeneticProgrammingAlgorithm>(MakeGp);
    CheckLimitBelowCheckpoint<Operon::NSGA2>(MakeNsga2);
}

TEST_CASE("The outcome signature covers every node field and every worker RNG word", "[cache][determinism]")
{
    auto const signatureOf = [](std::vector<Node> const& nodes) {
        Individual ind;
        ind.Genotype = Tree { Operon::Vector<Node>(nodes.begin(), nodes.end()) };
        return Signature(Operon::Span<Individual const>(&ind, 1));
    };

    Node base { NodeType::Variable, Operon::Hash { 11 } };
    base.CalculatedHashValue = 12;
    base.Value = Operon::Scalar { 0 };
    base.Arity = 2;
    base.Length = 3;
    base.Depth = 4;
    base.Level = 5;
    base.Parent = 6;
    base.IsEnabled = true;
    base.Optimize = false;
    base.RefTo = 7;
    std::vector<Node> const reference { base, base };
    auto const expected = signatureOf(reference);
    CHECK(Same(signatureOf(reference), expected));

    using Change = std::pair<char const*, void (*)(Node&)>;
    std::array<Change, 13> const changes { {
        { "HashValue", [](Node& n) { n.HashValue += 1; } },
        { "CalculatedHashValue", [](Node& n) { n.CalculatedHashValue += 1; } },
        { "Value, last bit", [](Node& n) { n.Value = std::nextafter(n.Value, Operon::Scalar { 1 }); } },
        { "Value, sign of zero", [](Node& n) { n.Value = -n.Value; } },
        { "Arity", [](Node& n) { n.Arity += 1; } },
        { "Length", [](Node& n) { n.Length += 1; } },
        { "Depth", [](Node& n) { n.Depth += 1; } },
        { "Level", [](Node& n) { n.Level += 1; } },
        { "Parent", [](Node& n) { n.Parent += 1; } },
        { "Type", [](Node& n) { n.Type = NodeType::Function; } },
        { "IsEnabled", [](Node& n) { n.IsEnabled = !n.IsEnabled; } },
        { "Optimize", [](Node& n) { n.Optimize = !n.Optimize; } },
        { "RefTo", [](Node& n) { n.RefTo += 1; } },
    } };
    for (auto const& [name, change] : changes) {
        CAPTURE(name);
        auto nodes = reference;
        change(nodes.back());
        CHECK_FALSE(Same(signatureOf(nodes), expected));
    }

    std::vector<Operon::RandomGenerator> rngs(3, Operon::RandomGenerator { 5 });
    auto const rngSignature = Signature(rngs);
    CHECK(Signature(rngs) == rngSignature);
    for (auto& rng : rngs) {
        auto const before = Signature(rngs);
        rng(); // advancing any one worker RNG must show
        CHECK(Signature(rngs) != before);
    }
}

} // namespace Operon::Test
