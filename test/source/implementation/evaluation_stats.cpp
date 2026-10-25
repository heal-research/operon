// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
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
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {
namespace {

    using DTable = DispatchTable<Operon::Scalar>;
    using ScoringStats = std::tuple<std::size_t, std::size_t, std::size_t>;

    constexpr int FunctionEvaluationsPerSearch = 3;
    constexpr int JacobianEvaluationsPerSearch = 2;
    // One prediction pass plus one search of fixed work: what a candidate costs a run that scores with an Evaluator
    // and optimizes with the FixedWorkOptimizer.
    constexpr std::size_t CostPerCandidate = 1 + FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch;

    // An optimizer that changes nothing and reports a fixed amount of work, so the expected counters are exact.
    class FixedWorkOptimizer final : public OptimizerBase {
    public:
        explicit FixedWorkOptimizer(gsl::not_null<Problem const*> problem) : OptimizerBase(problem) {}

        [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*rng*/, Tree const& tree) const -> FitOutcome override
        {
            FitResult result;
            result.InitialParameters = tree.GetCoefficients();
            result.FinalParameters = result.InitialParameters;
            result.InitialCost = static_cast<Operon::Scalar>(1);
            result.FinalCost = static_cast<Operon::Scalar>(0);
            result.FunctionEvaluations = FunctionEvaluationsPerSearch;
            result.JacobianEvaluations = JacobianEvaluationsPerSearch;
            return result;
        }
    };

    // An optimizer that returns whatever its script builds from the tree, and counts its invocations.
    class ScriptedOptimizer final : public OptimizerBase {
    public:
        using Script = std::function<FitOutcome(Tree const&)>;

        ScriptedOptimizer(gsl::not_null<Problem const*> problem, Script script)
            : OptimizerBase(problem)
            , script_(std::move(script))
        {
        }

        [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*rng*/, Tree const& tree) const -> FitOutcome override
        {
            ++Invocations;
            return script_(tree);
        }

        mutable std::atomic_size_t Invocations { 0 };

    private:
        Script script_;
    };

    // Diagnostics of a fit that doubles every coefficient: improving when `improves`, otherwise not.
    auto ScriptedDiagnostics(Tree const& tree, int functionEvaluations, int jacobianEvaluations, bool improves = true)
        -> FitDiagnostics
    {
        FitDiagnostics diag;
        diag.InitialParameters = tree.GetCoefficients();
        diag.FinalParameters = diag.InitialParameters;
        for (auto& c : diag.FinalParameters) {
            c *= 2;
        }
        diag.InitialCost = static_cast<Operon::Scalar>(1);
        diag.FinalCost = static_cast<Operon::Scalar>(improves ? 0 : 2);
        diag.FunctionEvaluations = functionEvaluations;
        diag.JacobianEvaluations = jacobianEvaluations;
        return diag;
    }

    // An evaluator whose scoring always throws, after the optimizer already did its work.
    struct ThrowingEvaluator final : Operon::EvaluatorBase {
        using EvaluatorBase::EvaluatorBase;

    protected:
        auto Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& /*ind*/,
            Operon::Span<Operon::Scalar> /*buf*/, std::optional<EvaluatedBuffer> /*evaluated*/) const
            -> ReturnType override
        {
            throw std::runtime_error("scoring failed");
        }
    };

    auto MakeDataset() -> Operon::Dataset
    {
        std::vector<std::vector<Operon::Scalar>> data(2, std::vector<Operon::Scalar>(10));
        std::iota(data[0].begin(), data[0].end(), Operon::Scalar { 1 });
        for (std::size_t i = 0; i < 10; ++i) {
            data[1][i] = data[0][i] * Operon::Scalar { 2 };
        }
        return Operon::Dataset({ "X1", "Y" }, data);
    }

    struct StatsFixture {
        Operon::Dataset Ds { MakeDataset() };
        Operon::Problem Problem { gsl::not_null<Operon::Dataset*>(&Ds) };
        DTable Dtable;
        Operon::RandomGenerator Rng { 1 };
        FixedWorkOptimizer Optimizer { &Problem };
        Operon::CoefficientOptimizer CoeffOptimizer { &Optimizer };
        Operon::OptimizerAccounting Accounting;
        std::vector<Operon::Scalar> Buf;

        StatsFixture()
        {
            Problem.SetTrainingRange({ 0, 10 });
            Problem.SetTestRange({ 0, 10 });
            Problem.SetTarget("Y");
            Buf.resize(10);
        }

        [[nodiscard]] auto MakeIndividual(std::string const& infix) const -> Operon::Individual
        {
            Operon::Individual ind;
            ind.Genotype = Operon::InfixParser::ParseOrThrow(infix, Ds);
            return ind;
        }

        // One local search (pLocal = 1) followed by one scoring evaluation; the search is charged to `Accounting`.
        void ScoreWithLocalSearch(Operon::EvaluatorBase const& evaluator, Operon::Individual& ind)
        {
            ScoreIndividual(Rng, ind, evaluator, &CoeffOptimizer, /*pLocal=*/1.0, /*pLamarck=*/1.0,
                Operon::Span<Operon::Scalar> { Buf }, Accounting);
        }
    };

    // Generator/algorithm scaffolding shared by the GP and NSGA2 tests.
    struct AlgorithmFixture : StatsFixture {
        static constexpr std::size_t PopSize = 8;
        static constexpr std::size_t PoolSize = 8;

        Operon::PrimitiveSet Pset;
        std::vector<Operon::Hash> Vars;
        Operon::SubtreeCrossover Crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
        Operon::MultiMutation Mutator;
        Operon::TournamentSelector FemSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::TournamentSelector MaleSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::KeepBestReinserter Reinserter { Operon::SingleObjectiveComparison { 0 } };
        Operon::BalancedTreeCreator Creator;
        Operon::UniformTreeInitializer TreeInit { &Creator };
        Operon::UniformCoefficientInitializer CoeffInit;
        Operon::GeneticAlgorithmConfig Config;

        AlgorithmFixture()
            : Pset([] {
                Operon::PrimitiveSet p;
                p.SetConfig(PrimitiveSet::Arithmetic);
                return p;
            }())
            , Vars { Ds.GetVariable("X1")->Hash }
            , Creator(&Pset, Vars, 0.0, 10)
        {
            TreeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
            OnePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            Mutator.Add(&OnePoint, 1.0);
            Config.PopulationSize = PopSize;
            Config.PoolSize = PoolSize;
            Config.Generations = 3;
            Config.LocalSearchProbability = 1.0;
            Config.LamarckianProbability = 1.0;
        }
    };

    // A two-objective [error, length] NSGA2 setup over the fixture.
    struct NsgaRig {
        Operon::Evaluator<DTable> Mse;
        Operon::TreePropertyEvaluator Length;
        Operon::MultiEvaluator Multi;
        Operon::CrowdedComparison Comp;
        Operon::TournamentSelector Fem { Comp };
        Operon::TournamentSelector Male { Comp };
        Operon::BasicOffspringGenerator Generator;
        Operon::DeductiveSorter Sorter;
        Operon::KeepBestReinserter Reinserter { Comp };

        explicit NsgaRig(AlgorithmFixture& fix)
            : Mse(&fix.Problem, &fix.Dtable, MSE {})
            , Length(&fix.Problem, [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); })
            , Multi(&fix.Problem)
            , Generator(&Multi, &fix.Crossover, &fix.Mutator, &Fem, &Male, &fix.CoeffOptimizer)
        {
            Multi.Add(&Mse);
            Multi.Add(&Length);
        }
    };

    // A custom generator, as downstream code writes them: it delegates to another generator and must hand it the
    // run's accounting for the optimizer work to reach the run.
    class ForwardingGenerator final : public Operon::OffspringGeneratorBase {
    public:
        ForwardingGenerator(Operon::EvaluatorBase const* eval, Operon::CrossoverBase const* cx,
            Operon::MutatorBase const* mut, Operon::SelectorBase const* femSel, Operon::SelectorBase const* maleSel,
            Operon::CoefficientOptimizer const* coeffOptimizer)
            : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel, coeffOptimizer)
            , inner_(eval, cx, mut, femSel, maleSel, coeffOptimizer)
        {
        }

        auto operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal,
            double pLamarck, Operon::Span<Operon::Scalar> buf, Operon::OptimizerAccounting& accounting) const
            -> std::optional<Operon::Individual> override
        {
            return inner_(random, pCrossover, pMutation, pLocal, pLamarck, buf, accounting);
        }

    private:
        Operon::BasicOffspringGenerator inner_;
    };

    // Every scoring call of a cold, uncached run with the fixture's fixed-work optimizer is paired with exactly one
    // search (initialization: one search and one scoring call per parent; a candidate: one of each), so whatever the
    // generator did with its candidates -- kept, rejected, lost a brood, discarded with its generation -- the
    // optimizer counters are the per-search work times the call count.
    void CheckFixedWorkAccounting(EvaluationStats const& stats)
    {
        REQUIRE(stats.Calls > 0);
        CHECK(stats.ResidualScoring == stats.Calls); // one prediction pass per scoring call
        CHECK(stats.JacobianScoring == 0);
        CHECK(stats.ResidualOptimizer == stats.Calls * FunctionEvaluationsPerSearch);
        CHECK(stats.JacobianOptimizer == stats.Calls * JacobianEvaluationsPerSearch);
        CHECK(stats.Consumed() == stats.Calls * CostPerCandidate);
    }

} // namespace

// ---- the evaluator owns scoring only ------------------------------------------------------------------------------

TEST_CASE("Evaluator counters are scoring-only; the optimizer pass lands in the run accounting", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    auto ind = fix.MakeIndividual("X1 + 0.5");

    CHECK(evaluator.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(fix.Accounting.Snapshot().Evaluations() == 0);

    fix.ScoreWithLocalSearch(evaluator, ind);

    auto const [residual, jacobian, calls] = evaluator.Stats();
    CHECK(residual == 1); // the prediction pass only
    CHECK(jacobian == 0);
    CHECK(calls == 1);
    CHECK(evaluator.TotalEvaluations() == 1);

    auto const work = fix.Accounting.Snapshot();
    CHECK(work.Residual == FunctionEvaluationsPerSearch);
    CHECK(work.Jacobian == JacobianEvaluationsPerSearch);
    CHECK(work.Evaluations() == FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch);
}

TEST_CASE("A Jacobian taken by a likelihood scorer is a scoring Jacobian", "[evaluator][stats]")
{
    StatsFixture fix;
    MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const evaluator { &fix.Problem,
        &fix.Dtable };
    auto const ind = fix.MakeIndividual("X1 * 0.7 + 0.5");

    static_cast<void>(evaluator(fix.Rng, ind, fix.Buf));

    auto const [residual, jacobian, calls] = evaluator.Stats();
    CHECK(residual == 1);
    CHECK(jacobian == 1);
    CHECK(calls == 1);
    CHECK(fix.Accounting.Snapshot().Evaluations() == 0); // no optimizer was involved
}

TEST_CASE("MultiEvaluator aggregates its children's scoring work, never optimizer work", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const mse { &fix.Problem, &fix.Dtable, MSE {} };
    Evaluator<DTable> const nmse { &fix.Problem, &fix.Dtable, NMSE {} };
    Operon::TreePropertyEvaluator const length { &fix.Problem,
        [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); } };
    MultiEvaluator inner { &fix.Problem };
    inner.Add(&nmse);
    inner.Add(&length);
    MultiEvaluator multi { &fix.Problem };
    multi.Add(&mse);
    multi.Add(&inner);
    auto ind = fix.MakeIndividual("X1 + 0.5");

    // The fit is recorded once, by the orchestration that ran it; the scoring evaluations land on the children that
    // run the prediction pass.
    fix.ScoreWithLocalSearch(multi, ind);

    auto const [residual, jacobian, calls] = multi.Stats();
    CHECK(residual == 2); // mse + nmse
    CHECK(jacobian == 0);
    CHECK(calls == 1); // the composite's own calls
    CHECK(fix.Accounting.Snapshot().Residual == FunctionEvaluationsPerSearch);
    CHECK(fix.Accounting.Snapshot().Jacobian == JacobianEvaluationsPerSearch);

    // The composite's own budget check is scoring-only: children's scoring plus its own.
    multi.SetBudget(2);
    CHECK(multi.BudgetExhausted());
    multi.SetBudget(3);
    CHECK_FALSE(multi.BudgetExhausted());
}

TEST_CASE("MultiEvaluator::Reset zeroes every child, however deeply nested", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const mse { &fix.Problem, &fix.Dtable, MSE {} };
    Evaluator<DTable> const nmse { &fix.Problem, &fix.Dtable, NMSE {} };
    MultiEvaluator inner { &fix.Problem };
    inner.Add(&nmse);
    MultiEvaluator multi { &fix.Problem };
    multi.Add(&mse);
    multi.Add(&inner);
    auto ind = fix.MakeIndividual("X1 + 0.5");
    fix.ScoreWithLocalSearch(multi, ind);
    REQUIRE(std::get<2>(mse.Stats()) == 1);
    REQUIRE(std::get<2>(nmse.Stats()) == 1);
    REQUIRE(std::get<0>(inner.Stats()) == 1);

    multi.Reset();

    CHECK(multi.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(inner.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(mse.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(nmse.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(multi.TotalEvaluations() == 0);
}

TEST_CASE("A child added twice to a MultiEvaluator is counted once per slot, in Stats() and the budget alike",
    "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const mse { &fix.Problem, &fix.Dtable, MSE {} };
    // MultiEvaluator::Add does not reject duplicates, and neither Stats() nor BudgetExhausted() deduplicates: every
    // slot contributes the child's own counters. The child runs once per slot, so a shared child is counted
    // (slots x its own work) times by the composite.
    MultiEvaluator inner { &fix.Problem };
    inner.Add(&mse);
    MultiEvaluator multi { &fix.Problem };
    multi.Add(&mse);
    multi.Add(&inner); // the same instance again, through a nested composite
    auto ind = fix.MakeIndividual("X1 + 0.5");

    fix.ScoreWithLocalSearch(multi, ind);

    CHECK(mse.TotalEvaluations() == 2); // the child really ran twice
    auto const [residual, jacobian, calls] = multi.Stats();
    CHECK(residual == 4); // 2 runs, each counted at both slots
    CHECK(jacobian == 0);
    CHECK(calls == 1);
    CHECK(fix.Accounting.Snapshot().Residual == FunctionEvaluationsPerSearch); // recorded once, not per slot

    // The budget consumes exactly what Stats() reports.
    multi.SetBudget(residual + jacobian);
    CHECK(multi.BudgetExhausted());
    multi.SetBudget(residual + jacobian + 1);
    CHECK_FALSE(multi.BudgetExhausted());

    multi.Reset();
    CHECK(multi.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(mse.Stats() == ScoringStats { 0, 0, 0 });
}

TEST_CASE("Scoring counters stay exact under concurrent scoring and recording", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    constexpr std::size_t threads = 4;
    constexpr std::size_t perThread = 250;

    std::vector<std::jthread> workers;
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            Operon::RandomGenerator rng { t };
            std::vector<Operon::Scalar> buf(10);
            auto ind = fix.MakeIndividual("X1 + 0.5");
            for (std::size_t i = 0; i < perThread; ++i) {
                ScoreIndividual(rng, ind, evaluator, &fix.CoeffOptimizer, 1.0, 1.0, Operon::Span<Operon::Scalar> { buf },
                    fix.Accounting);
            }
        });
    }
    workers.clear(); // joins

    constexpr auto n = threads * perThread;
    CHECK(evaluator.Stats() == ScoringStats { n, 0, n });
    auto const work = fix.Accounting.Snapshot();
    CHECK(work.Residual == n * FunctionEvaluationsPerSearch);
    CHECK(work.Jacobian == n * JacobianEvaluationsPerSearch);
}

TEST_CASE("The evaluator's budget check is scoring-only; the run's stop condition adds the optimizer work",
    "[evaluator][stats]")
{
    AlgorithmFixture fix;
    Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    auto ind = fix.MakeIndividual("X1 + 0.5");
    fix.ScoreWithLocalSearch(evaluator, ind);
    constexpr std::size_t consumed = CostPerCandidate;

    evaluator.SetBudget(consumed + 1);
    CHECK_FALSE(generator.Terminate(fix.Accounting));
    evaluator.SetBudget(consumed);
    CHECK(generator.Terminate(fix.Accounting));

    // Without the run's accounting the same evaluator has only scored once.
    Operon::OptimizerAccounting empty;
    CHECK_FALSE(generator.Terminate(empty));
    CHECK_FALSE(evaluator.BudgetExhausted()); // scoring work alone: 1 < 6
    evaluator.SetBudget(1);
    CHECK(evaluator.BudgetExhausted());
    CHECK(generator.Terminate(empty));
}

// ---- OptimizerAccounting ------------------------------------------------------------------------------------------

TEST_CASE("OptimizerAccounting records diagnostics, saturates, resets and copies", "[stats][accounting]")
{
    Operon::OptimizerAccounting accounting;
    CHECK(accounting.Snapshot().Residual == 0);

    FitDiagnostics diag;
    diag.FunctionEvaluations = 7;
    diag.JacobianEvaluations = 4;
    accounting.Record(diag, 11);
    accounting.Record(diag, 5);
    auto work = accounting.Snapshot();
    CHECK(work.Residual == 14);
    CHECK(work.Jacobian == 8);
    CHECK(work.TimeUs == 16);
    CHECK(work.Evaluations() == 22);
    CHECK(accounting.Evaluations() == 22);

    SECTION("an invocation that performed no work contributes zero work")
    {
        accounting.Record(FitDiagnostics {}, 0);
        CHECK(accounting.Snapshot().Residual == 14);
        CHECK(accounting.Snapshot().Jacobian == 8);
        CHECK(accounting.Snapshot().TimeUs == 16);
    }

    SECTION("negative signed fit counters count as zero")
    {
        FitDiagnostics sentinel;
        sentinel.FunctionEvaluations = -1;
        sentinel.JacobianEvaluations = -1;
        accounting.Record(sentinel, 0);
        CHECK(accounting.Snapshot().Residual == 14);
        CHECK(accounting.Snapshot().Jacobian == 8);
    }

    SECTION("counters saturate instead of wrapping")
    {
        constexpr auto max = std::numeric_limits<std::uint64_t>::max();
        accounting.Record(FitDiagnostics {}, max);
        accounting.Record(FitDiagnostics {}, 5);
        CHECK(accounting.Snapshot().TimeUs == max);
        // the combined count saturates too
        CHECK(detail::SaturatingAdd(max, std::uint64_t { 1 }) == max);
        OptimizerWork const full { .Residual = max, .Jacobian = 1, .TimeUs = 0 };
        CHECK(full.Evaluations() == max);
    }

    SECTION("a copy holds the same quiescent snapshot, independently")
    {
        Operon::OptimizerAccounting copy { accounting };
        CHECK(copy.Snapshot().Residual == 14);
        CHECK(copy.Snapshot().TimeUs == 16);
        copy.Record(diag, 1);
        CHECK(copy.Snapshot().Residual == 21);
        CHECK(accounting.Snapshot().Residual == 14);

        Operon::OptimizerAccounting assigned;
        assigned = accounting;
        CHECK(assigned.Snapshot().Jacobian == 8);
    }

    SECTION("Reset clears every counter")
    {
        accounting.Reset();
        CHECK(accounting.Snapshot().Residual == 0);
        CHECK(accounting.Snapshot().Jacobian == 0);
        CHECK(accounting.Snapshot().TimeUs == 0);
    }
}

TEST_CASE("OptimizerAccounting stays exact under concurrent recording", "[stats][accounting]")
{
    Operon::OptimizerAccounting accounting;
    constexpr std::size_t threads = 8;
    constexpr std::size_t perThread = 1000;

    std::vector<std::jthread> workers;
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
            FitDiagnostics diag;
            diag.FunctionEvaluations = FunctionEvaluationsPerSearch;
            diag.JacobianEvaluations = JacobianEvaluationsPerSearch;
            for (std::size_t i = 0; i < perThread; ++i) {
                accounting.Record(diag, 2);
            }
        });
    }
    workers.clear(); // joins

    constexpr auto n = threads * perThread;
    auto const work = accounting.Snapshot();
    CHECK(work.Residual == n * FunctionEvaluationsPerSearch);
    CHECK(work.Jacobian == n * JacobianEvaluationsPerSearch);
    CHECK(work.TimeUs == n * 2);
}

TEST_CASE("EvaluationStats totals are disjoint sums and saturate", "[stats][accounting]")
{
    EvaluationStats stats;
    stats.ResidualScoring = 10;
    stats.ResidualOptimizer = 3;
    stats.JacobianScoring = 1;
    stats.JacobianOptimizer = 2;
    stats.Calls = 99; // calls and time never consume the budget
    stats.OptimizerTimeUs = 99;
    CHECK(stats.ResidualTotal() == 13);
    CHECK(stats.JacobianTotal() == 3);
    CHECK(stats.Consumed() == 16);

    stats.ResidualScoring = std::numeric_limits<std::size_t>::max();
    CHECK(stats.ResidualTotal() == std::numeric_limits<std::size_t>::max());
    CHECK(stats.Consumed() == std::numeric_limits<std::size_t>::max());
}

// ---- LocalSearch ---------------------------------------------------------------------------------------------------

TEST_CASE("LocalSearch performs no work and charges nothing when no optimizer is invoked", "[stats][local-search]")
{
    StatsFixture fix;
    auto ind = fix.MakeIndividual("X1 + 0.5");
    auto const before = ind.Genotype.GetCoefficients();
    Operon::RandomGenerator rng { 5 };
    auto probe = rng;

    SECTION("no optimizer")
    {
        auto const result = LocalSearch(rng, ind, nullptr, 1.0, 1.0);
        CHECK_FALSE(result.Outcome.has_value());
        CHECK_FALSE(result.OriginalCoefficients.has_value());
        CHECK(result.OptimizerTimeUs == 0);
        CHECK(rng() == probe()); // no random number consumed
    }

    SECTION("pLocal is not positive")
    {
        auto const result = LocalSearch(rng, ind, &fix.CoeffOptimizer, 0.0, 1.0);
        CHECK_FALSE(result.Outcome.has_value());
        CHECK(rng() == probe());
    }

    SECTION("the Bernoulli trial skips the search")
    {
        // pLocal just above zero: the trial draw is consumed and (for this seed) fails.
        REQUIRE_FALSE(std::bernoulli_distribution { std::numeric_limits<double>::min() }(probe));
        auto const result = LocalSearch(rng, ind, &fix.CoeffOptimizer, std::numeric_limits<double>::min(), 1.0);
        CHECK_FALSE(result.Outcome.has_value());
        CHECK_FALSE(result.OriginalCoefficients.has_value());
        CHECK(result.OptimizerTimeUs == 0);
        CHECK(rng() == probe()); // exactly the one trial draw
    }

    CHECK(ind.Genotype.GetCoefficients() == before);
    Operon::OptimizerAccounting accounting;
    accounting.Record(LocalSearchResult {});
    CHECK(accounting.Snapshot().Evaluations() == 0);
    CHECK(accounting.Snapshot().TimeUs == 0);
}

TEST_CASE("LocalSearch with a zero-iteration optimizer reports a failure without work", "[stats][local-search]")
{
    StatsFixture fix;
    ScriptedOptimizer optimizer { &fix.Problem, [](Tree const& t) -> FitOutcome {
                                     return detail::MakeFitOutcome(ScriptedDiagnostics(t, 9, 9));
                                 } };
    optimizer.SetIterations(0);
    Operon::CoefficientOptimizer const coeffOptimizer { &optimizer };
    auto ind = fix.MakeIndividual("X1 + 0.5");
    Operon::RandomGenerator rng { 5 };

    auto const result = LocalSearch(rng, ind, &coeffOptimizer, 1.0, 1.0);

    REQUIRE(result.Outcome.has_value());
    CHECK_FALSE(result.Outcome->has_value()); // the wrapper's "unsuccessful" contract
    CHECK(optimizer.Invocations.load() == 0); // the backend was never reached
    CHECK(Diagnostics(*result.Outcome).FunctionEvaluations == 0);
    Operon::OptimizerAccounting accounting;
    accounting.Record(result);
    CHECK(accounting.Snapshot().Evaluations() == 0);
}

TEST_CASE("LocalSearch publishes the complete outcome of every invoked optimizer", "[stats][local-search]")
{
    StatsFixture fix;
    Operon::OptimizerAccounting accounting;
    Operon::RandomGenerator rng { 5 };
    auto ind = fix.MakeIndividual("X1 + 0.5");
    auto const original = ind.Genotype.GetCoefficients();
    REQUIRE(original.size() == 2);

    auto run = [&](ScriptedOptimizer::Script script, double pLamarck = 1.0) {
        ScriptedOptimizer optimizer { &fix.Problem, std::move(script) };
        Operon::CoefficientOptimizer const coeffOptimizer { &optimizer };
        auto result = LocalSearch(rng, ind, &coeffOptimizer, 1.0, pLamarck);
        REQUIRE(optimizer.Invocations.load() == 1);
        REQUIRE(result.Outcome.has_value());
        accounting.Record(result);
        return result;
    };

    SECTION("a successful fit")
    {
        auto const result = run([](Tree const& t) -> FitOutcome {
            return detail::MakeFitOutcome(ScriptedDiagnostics(t, 4, 3));
        });
        CHECK(result.Outcome->has_value());
        CHECK_FALSE(result.OriginalCoefficients.has_value()); // Lamarckian: the genotype keeps the update
        auto const updated = ind.Genotype.GetCoefficients();
        for (std::size_t i = 0; i < original.size(); ++i) {
            CHECK(updated[i] == original[i] * 2);
        }
        CHECK(accounting.Snapshot().Residual == 4);
        CHECK(accounting.Snapshot().Jacobian == 3);
    }

    SECTION("a successful non-Lamarckian fit hands back the original coefficients")
    {
        auto const result = run(
            [](Tree const& t) -> FitOutcome { return detail::MakeFitOutcome(ScriptedDiagnostics(t, 4, 3)); }, 0.0);
        REQUIRE(result.OriginalCoefficients.has_value());
        CHECK(*result.OriginalCoefficients == original);
        CHECK(ind.Genotype.GetCoefficients()[0] == original[0] * 2); // the genotype is the optimized one
        CHECK(accounting.Snapshot().Residual == 4);
    }

    SECTION("a fit that does not improve is charged and leaves the coefficients alone")
    {
        auto const result = run([](Tree const& t) -> FitOutcome {
            return detail::MakeFitOutcome(ScriptedDiagnostics(t, 5, 1, /*improves=*/false));
        });
        CHECK_FALSE(result.Outcome->has_value());
        CHECK(Diagnostics(*result.Outcome).FunctionEvaluations == 5);
        CHECK(ind.Genotype.GetCoefficients() == original);
        CHECK(accounting.Snapshot().Residual == 5);
        CHECK(accounting.Snapshot().Jacobian == 1);
    }

    SECTION("an evaluation failure after work began is charged")
    {
        auto const result = run([](Tree const& t) -> FitOutcome {
            return detail::MakeFitEvaluationError(InterpreterError {}, ScriptedDiagnostics(t, 2, 1));
        });
        CHECK(EvaluationError(*result.Outcome) != nullptr);
        CHECK(accounting.Snapshot().Residual == 2);
        CHECK(accounting.Snapshot().Jacobian == 1);
    }

    SECTION("a configuration failure after work began is charged")
    {
        auto const result = run([](Tree const& t) -> FitOutcome {
            return detail::MakeFitConfigurationError(WeightError {}, ScriptedDiagnostics(t, 1, 0));
        });
        CHECK(ConfigurationError(*result.Outcome) != nullptr);
        CHECK(accounting.Snapshot().Residual == 1);
        CHECK(accounting.Snapshot().Jacobian == 0);
    }

    SECTION("an error raised before anything was evaluated contributes zero work")
    {
        auto const result = run([](Tree const& t) -> FitOutcome {
            return detail::MakeUnevaluatedFitEvaluationError(GradientError {}, ScriptedDiagnostics(t, 0, 0));
        });
        CHECK(EvaluationError(*result.Outcome) != nullptr);
        CHECK(accounting.Snapshot().Evaluations() == 0);
    }
}

TEST_CASE("LocalSearch consumes random numbers in a fixed order", "[stats][local-search]")
{
    StatsFixture fix;
    auto ind = fix.MakeIndividual("X1 + 0.5");
    Operon::RandomGenerator rng { 9 };
    auto probe = rng;
    constexpr double pLocal = 0.999999;
    constexpr double pLamarck = 0.5;

    // local-search trial, (the optimizer ignores the generator), Lamarckian trial
    REQUIRE(std::bernoulli_distribution { pLocal }(probe));
    auto const lamarckian = std::bernoulli_distribution { pLamarck }(probe);

    auto const result = LocalSearch(rng, ind, &fix.CoeffOptimizer, pLocal, pLamarck);

    CHECK(result.OriginalCoefficients.has_value() == !lamarckian);
    CHECK(rng() == probe());
}

TEST_CASE("ScoreIndividual charges a completed fit even when scoring then throws", "[stats][local-search]")
{
    StatsFixture fix;
    ThrowingEvaluator const evaluator { &fix.Problem };
    auto ind = fix.MakeIndividual("X1 + 0.5");

    CHECK_THROWS_AS(fix.ScoreWithLocalSearch(evaluator, ind), std::runtime_error);

    auto const work = fix.Accounting.Snapshot();
    CHECK(work.Residual == FunctionEvaluationsPerSearch);
    CHECK(work.Jacobian == JacobianEvaluationsPerSearch);
}

TEST_CASE("ScoreIndividual with a skipped search charges no optimizer work", "[stats][local-search]")
{
    StatsFixture fix;
    Evaluator<DTable> const evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    auto ind = fix.MakeIndividual("X1 + 0.5");

    ScoreIndividual(fix.Rng, ind, evaluator, &fix.CoeffOptimizer, /*pLocal=*/0.0, /*pLamarck=*/1.0,
        Operon::Span<Operon::Scalar> { fix.Buf }, fix.Accounting);

    CHECK(evaluator.Stats() == ScoringStats { 1, 0, 1 });
    CHECK(fix.Accounting.Snapshot().Evaluations() == 0);
}

// ---- GeneticAlgorithmBase ------------------------------------------------------------------------------------------

TEST_CASE("GeneticAlgorithmBase::DetailedStats after a GP run", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };

    SECTION("no cache: scoring and optimizer counters come from their own owners, cache counters are null")
    {
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);

        auto const stats = gp.DetailedStats();
        auto const [residual, jacobian, calls] = evaluator.Stats();
        auto const work = gp.Accounting().Snapshot();
        CHECK(stats.Calls == calls);
        CHECK(stats.ResidualScoring == residual); // the evaluator's scoring counters, verbatim
        CHECK(stats.JacobianScoring == jacobian);
        CHECK(stats.ResidualOptimizer == work.Residual); // the run's accounting, verbatim
        CHECK(stats.JacobianOptimizer == work.Jacobian);
        CHECK(stats.OptimizerTimeUs == work.TimeUs);
        CHECK(stats.Consumed() == evaluator.TotalEvaluations() + work.Evaluations());
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (fix.Config.Generations * AlgorithmFixture::PoolSize));
        CHECK_FALSE(stats.CacheHits.has_value());
        CHECK_FALSE(stats.CacheLookups.has_value());
    }

    SECTION("a configured cache reports hits and lookups, and cache hits cost no optimizer work")
    {
        Operon::RandomGenerator cacheRng { 3 };
        Operon::Zobrist cache { cacheRng, /*maxLength=*/20, fix.Vars };
        auto config = fix.Config;
        config.Cache = &cache;
        Operon::GeneticProgrammingAlgorithm cachedGp { config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
            &fix.Reinserter };

        Operon::RandomGenerator rng { 7 };
        cachedGp.Run(rng, nullptr, /*threads=*/1);

        auto const stats = cachedGp.DetailedStats();
        REQUIRE(stats.CacheHits.has_value());
        REQUIRE(stats.CacheLookups.has_value());
        CHECK(*stats.CacheLookups > 0);
        CHECK(*stats.CacheHits <= *stats.CacheLookups);
        CHECK(*stats.CacheHits == cache.Hits());
        CHECK(*stats.CacheLookups == cache.Lookups());
        // every lookup is a hit (no scoring, no search) or a miss (one scoring call, one search); the initial
        // population is scored without the cache
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (*stats.CacheLookups - *stats.CacheHits));
        CheckFixedWorkAccounting(stats);

        cachedGp.Reset();
        auto const cleared = cachedGp.DetailedStats();
        CHECK(cleared.Calls == 0);
        CHECK(cleared.Consumed() == 0);
        CHECK(cleared.OptimizerTimeUs == 0);
        CHECK(cleared.CacheHits == std::optional<std::size_t> { 0 });
        CHECK(cleared.CacheLookups == std::optional<std::size_t> { 0 });
        CHECK(cachedGp.Accounting().Snapshot().Evaluations() == 0);
    }
}

TEST_CASE("Optimizer work is accounted for every generator and every candidate path", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    fix.Config.Generations = 2;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    constexpr std::size_t broodSize = 3;
    Operon::RandomGenerator rng { 7 };

    auto run = [&](Operon::OffspringGeneratorBase& generator) {
        Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
            &fix.Reinserter };
        gp.Run(rng, nullptr, /*threads=*/1);
        return std::pair { gp.DetailedStats(), gp.Generation() };
    };

    SECTION("basic")
    {
        Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel,
            &fix.MaleSel, &fix.CoeffOptimizer };
        auto const [stats, generation] = run(generator);
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (generation * AlgorithmFixture::PoolSize));
    }

    SECTION("brood: every candidate of the brood is charged, not only the survivor")
    {
        Operon::BroodOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel,
            &fix.MaleSel, &fix.CoeffOptimizer };
        generator.BroodSize(broodSize);
        auto const [stats, generation] = run(generator);
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (generation * AlgorithmFixture::PoolSize * broodSize));
    }

    SECTION("polygenic: every candidate is charged, not only the survivor")
    {
        Operon::PolygenicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel,
            &fix.MaleSel, &fix.CoeffOptimizer };
        generator.PolygenicSize(broodSize);
        auto const [stats, generation] = run(generator);
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (generation * AlgorithmFixture::PoolSize * broodSize));
    }

    SECTION("offspring selection: rejected candidates are charged")
    {
        Operon::OffspringSelectionGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel,
            &fix.MaleSel, &fix.CoeffOptimizer };
        auto const [stats, generation] = run(generator);
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls >= AlgorithmFixture::PopSize + (generation * AlgorithmFixture::PoolSize));
    }

    SECTION("a custom generator that forwards the run's accounting")
    {
        ForwardingGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
            &fix.CoeffOptimizer };
        auto const [stats, generation] = run(generator);
        CheckFixedWorkAccounting(stats);
        CHECK(stats.Calls == AlgorithmFixture::PopSize + (generation * AlgorithmFixture::PoolSize));
    }
}

TEST_CASE("NSGA2 accounts for the optimizer work of initialization and generations", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    NsgaRig rig { fix };
    Operon::NSGA2 nsga { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &rig.Generator, &rig.Reinserter,
        &rig.Sorter };

    Operon::RandomGenerator rng { 11 };
    nsga.Run(rng, nullptr, /*threads=*/2);

    auto const stats = nsga.DetailedStats();
    CheckFixedWorkAccounting(stats);
    CHECK(stats.Calls == AlgorithmFixture::PopSize + (fix.Config.Generations * AlgorithmFixture::PoolSize));
    // the composite's own scoring counters; the optimizer pass is not charged to it nor to any child
    auto const [residual, jacobian, calls] = rig.Multi.Stats();
    CHECK(stats.ResidualScoring == residual);
    CHECK(stats.JacobianScoring == jacobian);
    CHECK(stats.Calls == calls);
    CHECK(std::get<0>(rig.Mse.Stats()) == residual);
}

// ---- budget --------------------------------------------------------------------------------------------------------

namespace {
    constexpr std::size_t InitCost = AlgorithmFixture::PopSize * CostPerCandidate;
    constexpr std::size_t GenerationCost = AlgorithmFixture::PoolSize * CostPerCandidate;

    // The expectations are independent of the implementation: a candidate costs CostPerCandidate units in total
    // (one prediction pass plus the fixed search work), the budget is checked before every offspring attempt, and a
    // single worker makes the stop point exact. Initialization always completes.
    template <typename Algo> void CheckBudgetedRun(Algo& algo, EvaluatorBase& budgeted)
    {
        auto run = [&](std::size_t budget) {
            budgeted.SetBudget(budget);
            Operon::RandomGenerator rng { 7 };
            algo.Run(rng, nullptr, /*threads=*/1);
            return algo.DetailedStats();
        };

        SECTION("initialization alone reaches the budget: it commits and no generation starts")
        {
            auto const stats = run(InitCost);
            CHECK(stats.Consumed() == InitCost);
            CHECK(stats.ResidualOptimizer == AlgorithmFixture::PopSize * FunctionEvaluationsPerSearch);
            CHECK(stats.JacobianOptimizer == AlgorithmFixture::PopSize * JacobianEvaluationsPerSearch);
            CHECK(algo.Generation() == 0);
            CHECK_FALSE(algo.IncompleteGeneration().has_value());
        }

        SECTION("a budget below the initialization cost: initialization still completes in full")
        {
            auto const stats = run(1);
            CHECK(stats.Consumed() == InitCost);
            CHECK(algo.Generation() == 0);
            CHECK_FALSE(algo.IncompleteGeneration().has_value());
        }

        SECTION("a unit over the initialization: one candidate is generated, the generation is incomplete")
        {
            auto const stats = run(InitCost + 1);
            CHECK(stats.Consumed() == InitCost + CostPerCandidate); // the discarded generation's work is executed work
            CHECK(algo.Generation() == 0);
            REQUIRE(algo.IncompleteGeneration().has_value());
            CHECK(algo.IncompleteGeneration()->Completed == 1);
        }

        SECTION("optimizer work alone moves the stop point by whole candidates")
        {
            auto const stats = run(InitCost + (3 * CostPerCandidate));
            CHECK(stats.Consumed() == InitCost + (3 * CostPerCandidate));
            REQUIRE(algo.IncompleteGeneration().has_value());
            CHECK(algo.IncompleteGeneration()->Completed == 3);
        }

        SECTION("exactly one generation fits")
        {
            auto const stats = run(InitCost + GenerationCost);
            CHECK(stats.Consumed() == InitCost + GenerationCost);
            CHECK(algo.Generation() == 1);
            CHECK_FALSE(algo.IncompleteGeneration().has_value());
        }

        SECTION("one unit short of a generation still completes it; one unit over starts the next")
        {
            auto const stats = run(InitCost + GenerationCost - 1);
            CHECK(stats.Consumed() == InitCost + GenerationCost);
            CHECK(algo.Generation() == 1);
            CHECK_FALSE(algo.IncompleteGeneration().has_value());

            auto const over = run(InitCost + GenerationCost + 1);
            CHECK(over.Consumed() == InitCost + GenerationCost + CostPerCandidate);
            CHECK(algo.Generation() == 1);
            REQUIRE(algo.IncompleteGeneration().has_value());
        }

        SECTION("the accounting of a run that discarded a generation is the executed work, and the next run resets it")
        {
            auto const first = run(InitCost + 1);
            REQUIRE(algo.IncompleteGeneration().has_value());
            auto const executed = InitCost + CostPerCandidate;
            CHECK(first.Consumed() == executed);
            CHECK(algo.Accounting().Snapshot().Evaluations()
                == (AlgorithmFixture::PopSize + 1) * (FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch));

            auto const second = run(InitCost + 1);
            CHECK(second.Consumed() == executed); // not double
            algo.Reset();
            CHECK(algo.Accounting().Snapshot().Evaluations() == 0);
            CHECK(algo.DetailedStats().Consumed() == 0);
        }
    }
} // namespace

TEST_CASE("GP budget consumption includes the optimizer work of initialization and offspring", "[algorithms][budget]")
{
    AlgorithmFixture fix;
    fix.Config.Generations = 100;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };
    CheckBudgetedRun(gp, evaluator);
}

TEST_CASE("NSGA2 budget consumption includes the optimizer work of initialization and offspring",
    "[algorithms][budget]")
{
    AlgorithmFixture fix;
    fix.Config.Generations = 100;
    NsgaRig rig { fix };
    Operon::NSGA2 nsga { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &rig.Generator, &rig.Reinserter,
        &rig.Sorter };
    CheckBudgetedRun(nsga, rig.Multi);
}

TEST_CASE("Optimizer work is budgeted even when scoring is free", "[algorithms][budget]")
{
    AlgorithmFixture fix;
    fix.Config.Generations = 100;
    // A genotype-only objective: scoring costs nothing, local search on its individuals does.
    Operon::TreePropertyEvaluator length { &fix.Problem,
        [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); } };
    Operon::BasicOffspringGenerator generator { &length, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };
    constexpr std::size_t searchCost = FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch;

    length.SetBudget((AlgorithmFixture::PopSize + 2) * searchCost);
    Operon::RandomGenerator rng { 7 };
    gp.Run(rng, nullptr, /*threads=*/1);

    auto const stats = gp.DetailedStats();
    CHECK(stats.ResidualScoring == 0);
    CHECK(stats.JacobianScoring == 0);
    CHECK(stats.Consumed() == (AlgorithmFixture::PopSize + 2) * searchCost);
    REQUIRE(gp.IncompleteGeneration().has_value());
    CHECK(gp.IncompleteGeneration()->Completed == 2);
}

TEST_CASE("A warm resume re-evaluates without local search, so it spends no optimizer work", "[algorithms][budget]")
{
    AlgorithmFixture fix;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };

    Operon::RandomGenerator rng { 7 };
    gp.Run(rng, nullptr, /*threads=*/1);
    REQUIRE(gp.IsFitted());
    REQUIRE(gp.Generation() == fix.Config.Generations);
    REQUIRE(gp.Accounting().Snapshot().Evaluations() > 0);

    Operon::RandomGenerator rng2 { 8 };
    gp.Run(rng2, nullptr, /*threads=*/1, /*warmStart=*/true);

    auto const stats = gp.DetailedStats();
    CHECK(stats.Calls == AlgorithmFixture::PopSize); // the parents, re-evaluated
    CHECK(stats.ResidualScoring == AlgorithmFixture::PopSize);
    CHECK(stats.ResidualOptimizer == 0);
    CHECK(stats.JacobianOptimizer == 0);
    CHECK(gp.Accounting().Snapshot().Evaluations() == 0);
}

TEST_CASE("Copying a quiescent algorithm preserves its run accounting", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };
    Operon::RandomGenerator rng { 7 };
    gp.Run(rng, nullptr, /*threads=*/1);
    auto const work = gp.Accounting().Snapshot();
    REQUIRE(work.Evaluations() > 0);

    Operon::GeneticProgrammingAlgorithm copy { gp };
    CHECK(copy.Accounting().Snapshot().Residual == work.Residual);
    CHECK(copy.Accounting().Snapshot().Jacobian == work.Jacobian);
    CHECK(copy.Accounting().Snapshot().TimeUs == work.TimeUs);
    CHECK(copy.DetailedStats() == gp.DetailedStats());

    Operon::GeneticProgrammingAlgorithm assigned { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit,
        &generator, &fix.Reinserter };
    assigned = gp;
    CHECK(assigned.Accounting().Snapshot().Residual == work.Residual);

    copy.Reset(); // the copies are independent
    CHECK(copy.Accounting().Snapshot().Evaluations() == 0);
    CHECK(gp.Accounting().Snapshot().Residual == work.Residual);
}

// ---- cache clock, reset and repeated runs -------------------------------------------------------------------------

TEST_CASE("GeneticAlgorithmBase cache counters follow the generator's active cache", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::RandomGenerator cacheRng { 3 };
    Operon::Zobrist configured { cacheRng, /*maxLength=*/20, fix.Vars };
    Operon::Zobrist replacement { cacheRng, /*maxLength=*/20, fix.Vars };
    auto config = fix.Config;
    config.Cache = &configured;
    Operon::GeneticProgrammingAlgorithm gp { config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };
    REQUIRE(generator.Cache() == &configured);

    SECTION("replaced by another cache")
    {
        generator.SetCache(&replacement);
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);

        auto const stats = gp.DetailedStats();
        REQUIRE(stats.CacheLookups.has_value());
        CHECK(replacement.Lookups() > 0);
        CHECK(configured.Lookups() == 0);
        CHECK(stats.CacheHits == std::optional<std::size_t> { replacement.Hits() });
        CHECK(stats.CacheLookups == std::optional<std::size_t> { replacement.Lookups() });

        // Reset clears the active cache's counters, not the configured one's.
        auto const hash = configured.ComputeHash(fix.MakeIndividual("X1").Genotype);
        Operon::Vector<Operon::Scalar> out(1);
        CHECK_FALSE(configured.TryGet(hash, out)); // a miss still counts as a lookup
        REQUIRE(configured.Lookups() > 0);
        auto const configuredLookups = configured.Lookups();
        gp.Reset();
        CHECK(replacement.Hits() == 0);
        CHECK(replacement.Lookups() == 0);
        CHECK(configured.Lookups() == configuredLookups);
    }

    SECTION("the replacement's generation clock is advanced, so a nonzero max age expires old entries")
    {
        Operon::ZobristOptions const immediate { .MaxAge = 1, .Visibility = Operon::CacheVisibility::Immediate };
        Operon::Zobrist aging { cacheRng, /*maxLength=*/20, fix.Vars, immediate };
        Operon::Zobrist keeping { cacheRng, /*maxLength=*/20, fix.Vars }; // max age 0: never expires
        auto const hash = Operon::Hash { 0xDEADBEEF };
        Operon::Vector<Operon::Scalar> out;
        aging.Insert(hash, { Operon::Scalar { 1 } }); // stamped with generation 0, before the run
        keeping.Insert(hash, { Operon::Scalar { 1 } });
        REQUIRE(aging.TryGet(hash, out)); // fresh while the clock is still at 0

        generator.SetCache(&aging);
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);
        REQUIRE(gp.Generation() >= 2);
        CHECK_FALSE(aging.TryGet(hash, out)); // the clock moved past maxAge on the generator's cache
        CHECK(configured.Lookups() == 0);

        generator.SetCache(&keeping);
        Operon::RandomGenerator rng2 { 7 };
        gp.Run(rng2, nullptr, /*threads=*/2);
        CHECK(keeping.TryGet(hash, out)); // unchanged behaviour for the default max age
    }

    SECTION("committed: the clock advance expires old entries at the generation boundary")
    {
        Operon::Zobrist aging { cacheRng, /*maxLength=*/20, fix.Vars, { .MaxAge = 1 } };
        auto const hash = Operon::Hash { 0xDEADBEEF };
        aging.Insert(hash, { Operon::Scalar { 1 } }); // stamped with generation 0, before the run
        REQUIRE(aging.Size() == 1);

        generator.SetCache(&aging);
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);
        REQUIRE(gp.Generation() >= 2);
        Operon::Vector<Operon::Scalar> out;
        CHECK_FALSE(aging.TryGet(hash, out));
        CHECK(configured.Lookups() == 0);
    }

    SECTION("removed")
    {
        generator.SetCache(nullptr);
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);

        auto const stats = gp.DetailedStats();
        CHECK_FALSE(stats.CacheHits.has_value());
        CHECK_FALSE(stats.CacheLookups.has_value());
        CHECK(configured.Lookups() == 0);
    }
}

TEST_CASE("GeneticAlgorithmBase::Reset and a second Run start from zero on every child", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    NsgaRig rig { fix };
    Operon::NSGA2 nsga { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &rig.Generator, &rig.Reinserter,
        &rig.Sorter };

    Operon::RandomGenerator rng { 11 };
    nsga.Run(rng, nullptr, /*threads=*/2);
    auto const first = nsga.DetailedStats();
    REQUIRE(first.Calls > 0);
    REQUIRE(std::get<2>(rig.Mse.Stats()) > 0);
    REQUIRE(nsga.Accounting().Snapshot().Evaluations() > 0);

    nsga.Reset();
    CHECK(nsga.DetailedStats() == EvaluationStats {});
    CHECK(rig.Mse.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(rig.Length.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(rig.Multi.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(rig.Multi.TotalEvaluations() == 0);
    CHECK(nsga.Accounting().Snapshot().Evaluations() == 0);
}

TEST_CASE("Run() clears the counters at start", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    NsgaRig rig { fix };
    Operon::RandomGenerator cacheRng { 3 };
    Operon::Zobrist cache { cacheRng, /*maxLength=*/20, fix.Vars };
    auto config = fix.Config;
    config.Cache = &cache;
    Operon::NSGA2 nsga { config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &rig.Generator, &rig.Reinserter,
        &rig.Sorter };

    SECTION("a second Run without an intervening Reset reports only its own counts")
    {
        // Without a cache every run scores the same number of individuals (the initial population plus one pool per
        // generation), so accumulated counters would show up as exactly double.
        rig.Generator.SetCache(nullptr);
        Operon::RandomGenerator rng1 { 11 };
        nsga.Run(rng1, nullptr, /*threads=*/1);
        auto const first = nsga.DetailedStats();
        auto const expected = fix.PopSize + (fix.Config.Generations * fix.PoolSize);
        REQUIRE(first.Calls == expected);
        REQUIRE(first.ResidualOptimizer > 0);
        REQUIRE(std::get<2>(rig.Mse.Stats()) == expected);
        REQUIRE_FALSE(first.CacheLookups.has_value());

        Operon::RandomGenerator rng2 { 12 };
        nsga.Run(rng2, nullptr, /*threads=*/1); // no Reset() in between
        auto const second = nsga.DetailedStats();
        CHECK(second.Calls == expected);
        CHECK(second.ResidualOptimizer == first.ResidualOptimizer); // the run's accounting restarted too
        CHECK(second.JacobianOptimizer == first.JacobianOptimizer);
        CHECK(std::get<2>(rig.Mse.Stats()) == expected);
    }

    SECTION("preloaded counters are cleared by the run-start reset")
    {
        // Leave nonzero evaluator, accounting and cache counters behind, as a previous run (or other users of the
        // objects) would.
        Operon::Vector<Operon::Scalar> out;
        for (auto i = 0; i < 50; ++i) {
            std::ignore = cache.TryGet(Operon::Hash { 1000U + static_cast<std::uint64_t>(i) }, out);
        }
        rig.Mse.ResidualEvaluations = 1'000'000;
        rig.Mse.CallCount = 1'000'000;
        rig.Multi.ResidualEvaluations = 1'000'000;
        FitDiagnostics preloaded;
        preloaded.FunctionEvaluations = 1'000'000;
        nsga.Accounting().Record(preloaded, 1);
        REQUIRE(cache.Lookups() == 50);
        REQUIRE(nsga.DetailedStats().Consumed() >= 3'000'000);

        Operon::RandomGenerator rng { 11 };
        nsga.Run(rng, nullptr, /*threads=*/1);

        auto const stats = nsga.DetailedStats();
        REQUIRE(stats.Calls > 0);
        CHECK(stats.Consumed() < 1'000'000);
        CHECK(rig.Mse.CallCount.load() < 1'000'000);
        CHECK(nsga.Accounting().Snapshot().Residual < 1'000'000);
        REQUIRE(stats.CacheLookups.has_value());

        // The same run on an empty cache: the preloaded lookups must not show up in the preloaded run's count.
        cache.Clear();
        Operon::RandomGenerator again { 11 };
        nsga.Run(again, nullptr, /*threads=*/1);
        CHECK(nsga.DetailedStats().CacheLookups == stats.CacheLookups);
    }
}

namespace {
    // A downstream evaluator that declares its own `Reset()`: it hides the base one rather than overriding it.
    struct HidingEvaluator final : Operon::EvaluatorBase {
        using EvaluatorBase::EvaluatorBase;
        mutable bool ResetChildrenCalled { false };
        mutable bool OwnResetCalled { false };
        void Reset() const { OwnResetCalled = true; }

    protected:
        void ResetChildren() const override { ResetChildrenCalled = true; }
        auto Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& /*ind*/,
            Operon::Span<Operon::Scalar> /*buf*/, std::optional<EvaluatedBuffer> /*evaluated*/) const
            -> ReturnType override
        {
            return ReturnType {};
        }
    };
} // namespace

TEST_CASE(
    "EvaluatorBase::Reset is not virtual and clears the base counters through the base pointer", "[evaluator][stats]")
{
    StatsFixture fix;
    HidingEvaluator hiding { &fix.Problem };
    Operon::EvaluatorBase const* base = &hiding;
    hiding.ResidualEvaluations = 5;
    hiding.JacobianEvaluations = 6;
    hiding.CallCount = 7;

    base->Reset();
    CHECK_FALSE(hiding.OwnResetCalled);
    CHECK(hiding.ResetChildrenCalled);
    CHECK(hiding.Stats() == ScoringStats { 0, 0, 0 });
    CHECK(hiding.TotalEvaluations() == 0);
}

TEST_CASE("NSGA2 advances the generator's active cache clock", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    NsgaRig rig { fix };
    Operon::NSGA2 nsga { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &rig.Generator, &rig.Reinserter,
        &rig.Sorter };

    Operon::RandomGenerator cacheRng { 3 };
    Operon::Zobrist aging { cacheRng, /*maxLength=*/20, fix.Vars,
        { .MaxAge = 1, .Visibility = Operon::CacheVisibility::Immediate } };
    auto const hash = Operon::Hash { 0xDEADBEEF };
    Operon::Vector<Operon::Scalar> out;
    aging.Insert(hash, { Operon::Scalar { 1 } });
    REQUIRE(aging.TryGet(hash, out));

    rig.Generator.SetCache(&aging);
    Operon::RandomGenerator rng { 7 };
    nsga.Run(rng, nullptr, /*threads=*/2);
    REQUIRE(nsga.Generation() >= 2);
    CHECK_FALSE(aging.TryGet(hash, out));
}


} // namespace Operon::Test
