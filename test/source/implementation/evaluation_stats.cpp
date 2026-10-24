// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "operon/algorithms/config.hpp"
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

    constexpr int FunctionEvaluationsPerSearch = 3;
    constexpr int JacobianEvaluationsPerSearch = 2;

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

        // One local search (pLocal = 1) followed by one scoring evaluation.
        void ScoreWithLocalSearch(Operon::EvaluatorBase const& evaluator, Operon::Individual& ind)
        {
            ScoreIndividual(Rng, ind, evaluator, &CoeffOptimizer, /*pLocal=*/1.0, /*pLamarck=*/1.0,
                Operon::Span<Operon::Scalar> { Buf });
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

} // namespace

TEST_CASE("DetailedStats separates a scoring evaluation from an optimizer pass", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    auto ind = fix.MakeIndividual("X1 + 0.5");

    CHECK(evaluator.DetailedStats() == EvaluationStats {});

    fix.ScoreWithLocalSearch(evaluator, ind);

    auto const stats = evaluator.DetailedStats();
    CHECK(stats.Calls == 1);
    CHECK(stats.ResidualScoring == 1);
    CHECK(stats.ResidualOptimizer == FunctionEvaluationsPerSearch);
    CHECK(stats.JacobianScoring == 0);
    CHECK(stats.JacobianOptimizer == JacobianEvaluationsPerSearch);
    CHECK_FALSE(stats.CacheHits.has_value());
    CHECK_FALSE(stats.CacheLookups.has_value());

    // The aggregate view is unchanged: Stats() and the budget see the same totals as before the split.
    auto const [residual, jacobian, calls, time] = evaluator.Stats();
    CHECK(residual == stats.ResidualTotal());
    CHECK(jacobian == stats.JacobianTotal());
    CHECK(calls == stats.Calls);
    CHECK(time == stats.OptimizerTimeUs);
    CHECK(evaluator.TotalEvaluations() == stats.Consumed());
    CHECK(stats.Consumed() == 1 + FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch);
}

TEST_CASE("The evaluation budget counts scoring and optimizer evaluations alike", "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    auto ind = fix.MakeIndividual("X1 + 0.5");
    constexpr std::size_t consumed = 1 + FunctionEvaluationsPerSearch + JacobianEvaluationsPerSearch;

    fix.ScoreWithLocalSearch(evaluator, ind);

    evaluator.SetBudget(consumed + 1);
    CHECK_FALSE(evaluator.BudgetExhausted());
    evaluator.SetBudget(consumed);
    CHECK(evaluator.BudgetExhausted());
}

TEST_CASE("A Jacobian taken by a likelihood scorer is a scoring Jacobian", "[evaluator][stats]")
{
    StatsFixture fix;
    MinimumDescriptionLengthEvaluator<DTable, GaussianLikelihood<Operon::Scalar>> const evaluator { &fix.Problem,
        &fix.Dtable };
    auto const ind = fix.MakeIndividual("X1 * 0.7 + 0.5");

    static_cast<void>(evaluator(fix.Rng, ind, fix.Buf));

    auto const stats = evaluator.DetailedStats();
    CHECK(stats.ResidualScoring == 1);
    CHECK(stats.JacobianScoring == 1);
    CHECK(stats.ResidualOptimizer == 0);
    CHECK(stats.JacobianOptimizer == 0);
}

TEST_CASE("MultiEvaluator aggregates its children and its own optimizer charges", "[evaluator][stats]")
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

    // The optimizer's work is charged to the composite the algorithm hands to LocalSearch; the scoring
    // evaluations land on the children that run the prediction pass.
    fix.ScoreWithLocalSearch(multi, ind);

    auto const stats = multi.DetailedStats();
    CHECK(stats.Calls == 1); // the composite's own calls, like Stats()
    CHECK(stats.ResidualScoring == 2); // mse + nmse
    CHECK(stats.ResidualOptimizer == FunctionEvaluationsPerSearch);
    CHECK(stats.JacobianScoring == 0);
    CHECK(stats.JacobianOptimizer == JacobianEvaluationsPerSearch);

    auto const [residual, jacobian, calls, time] = multi.Stats();
    CHECK(residual == stats.ResidualTotal());
    CHECK(jacobian == stats.JacobianTotal());
    CHECK(calls == stats.Calls);
    CHECK(time == stats.OptimizerTimeUs);

    // Budget semantics: the composite's budget is its children's plus its own, exactly as before.
    multi.SetBudget(stats.Consumed());
    CHECK(multi.BudgetExhausted());
    multi.SetBudget(stats.Consumed() + 1);
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
    REQUIRE(mse.DetailedStats().Calls == 1);
    REQUIRE(nmse.DetailedStats().Calls == 1);
    REQUIRE(inner.DetailedStats().ResidualScoring == 1);

    multi.Reset();

    CHECK(multi.DetailedStats() == EvaluationStats {});
    CHECK(inner.DetailedStats() == EvaluationStats {});
    CHECK(mse.DetailedStats() == EvaluationStats {});
    CHECK(nmse.DetailedStats() == EvaluationStats {});
    CHECK(multi.OptimizerResidualEvaluations.load() == 0);
    CHECK(multi.OptimizerJacobianEvaluations.load() == 0);
    CHECK(multi.TotalEvaluations() == 0);
}

TEST_CASE("A child added twice to a MultiEvaluator is counted once per slot, in Stats() and DetailedStats() alike",
    "[evaluator][stats]")
{
    StatsFixture fix;
    Evaluator<DTable> const mse { &fix.Problem, &fix.Dtable, MSE {} };
    // MultiEvaluator::Add does not reject duplicates, and neither Stats() nor BudgetExhausted() deduplicates: every
    // slot contributes the child's own counters. The child runs once per slot, so a shared child is counted
    // (slots x its own work) times by the composite. DetailedStats() sums the same way and agrees exactly.
    MultiEvaluator inner { &fix.Problem };
    inner.Add(&mse);
    MultiEvaluator multi { &fix.Problem };
    multi.Add(&mse);
    multi.Add(&inner); // the same instance again, through a nested composite
    auto ind = fix.MakeIndividual("X1 + 0.5");

    fix.ScoreWithLocalSearch(multi, ind);

    CHECK(mse.TotalEvaluations() == 2); // the child really ran twice
    auto const stats = multi.DetailedStats();
    auto const [residual, jacobian, calls, time] = multi.Stats();
    CHECK(stats.ResidualScoring == 4); // 2 runs, each counted at both slots
    CHECK(stats.ResidualOptimizer == FunctionEvaluationsPerSearch);
    CHECK(residual == stats.ResidualTotal());
    CHECK(jacobian == stats.JacobianTotal());
    CHECK(calls == stats.Calls);
    CHECK(time == stats.OptimizerTimeUs);

    // The budget consumes exactly what the stats report.
    multi.SetBudget(stats.Consumed());
    CHECK(multi.BudgetExhausted());
    multi.SetBudget(stats.Consumed() + 1);
    CHECK_FALSE(multi.BudgetExhausted());

    multi.Reset();
    CHECK(multi.DetailedStats() == EvaluationStats {});
    CHECK(mse.DetailedStats() == EvaluationStats {});
}

TEST_CASE("Counters stay exact under concurrent scoring", "[evaluator][stats]")
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
                ScoreIndividual(
                    rng, ind, evaluator, &fix.CoeffOptimizer, 1.0, 1.0, Operon::Span<Operon::Scalar> { buf });
            }
        });
    }
    workers.clear(); // joins

    auto const stats = evaluator.DetailedStats();
    constexpr auto n = threads * perThread;
    CHECK(stats.Calls == n);
    CHECK(stats.ResidualScoring == n);
    CHECK(stats.ResidualOptimizer == n * FunctionEvaluationsPerSearch);
    CHECK(stats.JacobianOptimizer == n * JacobianEvaluationsPerSearch);
    CHECK(stats.JacobianScoring == 0);
}

TEST_CASE("GeneticAlgorithmBase::DetailedStats after a GP run", "[algorithms][stats]")
{
    AlgorithmFixture fix;
    Operon::Evaluator<DTable> evaluator { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::BasicOffspringGenerator generator { &evaluator, &fix.Crossover, &fix.Mutator, &fix.FemSel, &fix.MaleSel,
        &fix.CoeffOptimizer };
    Operon::GeneticProgrammingAlgorithm gp { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
        &fix.Reinserter };

    SECTION("no cache: counters split, cache counters null")
    {
        Operon::RandomGenerator rng { 7 };
        gp.Run(rng, nullptr, /*threads=*/2);

        auto const stats = gp.DetailedStats();
        auto const [residual, jacobian, calls, time] = evaluator.Stats();
        CHECK(stats.Calls == calls);
        CHECK(stats.ResidualTotal() == residual);
        CHECK(stats.JacobianTotal() == jacobian);
        CHECK(stats.Consumed() == evaluator.TotalEvaluations());
        CHECK(stats.OptimizerTimeUs == time);
        CHECK(stats.ResidualScoring == stats.Calls); // every scoring call runs exactly one prediction pass
        CHECK(stats.ResidualOptimizer > 0);
        // each search charges (FunctionEvaluationsPerSearch, JacobianEvaluationsPerSearch) in fixed proportion
        CHECK(stats.ResidualOptimizer * JacobianEvaluationsPerSearch
            == stats.JacobianOptimizer * FunctionEvaluationsPerSearch);
        CHECK_FALSE(stats.CacheHits.has_value());
        CHECK_FALSE(stats.CacheLookups.has_value());
    }

    SECTION("a configured cache reports hits and lookups; Reset zeroes everything")
    {
        Operon::RandomGenerator cacheRng { 3 };
        Operon::Zobrist cache { cacheRng, /*maxLength=*/20, fix.Vars };
        auto config = fix.Config;
        config.Cache = &cache;
        Operon::GeneticProgrammingAlgorithm cachedGp { config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &generator,
            &fix.Reinserter };

        Operon::RandomGenerator rng { 7 };
        cachedGp.Run(rng, nullptr, /*threads=*/2);

        auto const stats = cachedGp.DetailedStats();
        REQUIRE(stats.CacheHits.has_value());
        REQUIRE(stats.CacheLookups.has_value());
        CHECK(*stats.CacheLookups > 0);
        CHECK(*stats.CacheHits <= *stats.CacheLookups);
        CHECK(*stats.CacheHits == cache.Hits());
        CHECK(*stats.CacheLookups == cache.Lookups());

        cachedGp.Reset();
        auto const cleared = cachedGp.DetailedStats();
        CHECK(cleared.Calls == 0);
        CHECK(cleared.Consumed() == 0);
        CHECK(cleared.CacheHits == std::optional<std::size_t> { 0 });
        CHECK(cleared.CacheLookups == std::optional<std::size_t> { 0 });
    }
}

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
    Operon::Evaluator<DTable> mse { &fix.Problem, &fix.Dtable, MSE {} };
    Operon::TreePropertyEvaluator length { &fix.Problem,
        [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); } };
    Operon::MultiEvaluator multi { &fix.Problem };
    multi.Add(&mse);
    multi.Add(&length);
    Operon::CrowdedComparison comp;
    Operon::TournamentSelector fem { comp };
    Operon::TournamentSelector male { comp };
    Operon::BasicOffspringGenerator nsgaGenerator { &multi, &fix.Crossover, &fix.Mutator, &fem, &male,
        &fix.CoeffOptimizer };
    Operon::DeductiveSorter sorter;
    Operon::KeepBestReinserter reinserter { comp };
    Operon::NSGA2 nsga { fix.Config, &fix.Problem, &fix.TreeInit, &fix.CoeffInit, &nsgaGenerator, &reinserter,
        &sorter };

    Operon::RandomGenerator rng { 11 };
    nsga.Run(rng, nullptr, /*threads=*/2);
    auto const first = nsga.DetailedStats();
    REQUIRE(first.Calls > 0);
    REQUIRE(mse.DetailedStats().Calls > 0);
    auto const [residual, jacobian, calls, time] = multi.Stats();
    CHECK(first.ResidualTotal() == residual);
    CHECK(first.JacobianTotal() == jacobian);
    CHECK(first.Calls == calls);
    // the optimizer pass is charged to the composite, the prediction passes to its first child
    CHECK(multi.OptimizerResidualEvaluations.load() > 0);
    CHECK(mse.OptimizerResidualEvaluations.load() == 0);

    nsga.Reset();
    CHECK(nsga.DetailedStats() == EvaluationStats {});
    CHECK(mse.DetailedStats() == EvaluationStats {});
    CHECK(length.DetailedStats() == EvaluationStats {});
    CHECK(multi.TotalEvaluations() == 0);

    // Run() resets at start: a second run's counters do not include the first run's.
    nsga.Run(rng, nullptr, /*threads=*/2);
    auto const second = nsga.DetailedStats();
    CHECK(second.Calls > 0);
    CHECK(second.Calls < 2 * first.Calls);
}

} // namespace Operon::Test
