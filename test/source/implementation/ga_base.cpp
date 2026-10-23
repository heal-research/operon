// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
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
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

namespace Operon::Test {
namespace {

    using DTable = DispatchTable<Operon::Scalar>;

    auto MakeDataset() -> Operon::Dataset
    {
        std::vector<std::vector<Operon::Scalar>> data(2, std::vector<Operon::Scalar>(10));
        std::iota(data[0].begin(), data[0].end(), Operon::Scalar { 1 });
        for (std::size_t i = 0; i < 10; ++i) {
            data[1][i] = data[0][i] * Operon::Scalar { 2 };
        }
        return Operon::Dataset({ "X1", "Y" }, data);
    }

    auto MakePset() -> Operon::PrimitiveSet
    {
        Operon::PrimitiveSet p;
        p.SetConfig(PrimitiveSet::Arithmetic);
        return p;
    }

    auto MakeConfig(std::size_t popSize, std::size_t poolSize) -> Operon::GeneticAlgorithmConfig
    {
        Operon::GeneticAlgorithmConfig c;
        c.PopulationSize = popSize;
        c.PoolSize = poolSize;
        c.Generations = 1;
        return c;
    }

    struct GaBaseFixture {
        static constexpr std::size_t PopSize = 6;
        static constexpr std::size_t PoolSize = 4;

        Operon::Dataset Ds { MakeDataset() };
        Operon::Problem Problem { gsl::not_null<Operon::Dataset*>(&Ds) };
        Operon::PrimitiveSet Pset { MakePset() };
        std::vector<Operon::Hash> Vars { Ds.GetVariable("X1")->Hash };

        DTable Dtable;
        Operon::Evaluator<DTable> Evaluator { &Problem, &Dtable };
        Operon::SubtreeCrossover Crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
        Operon::MultiMutation Mutator;
        Operon::TournamentSelector FemSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::TournamentSelector MaleSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::BasicOffspringGenerator Generator { &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
        Operon::KeepBestReinserter Reinserter { Operon::SingleObjectiveComparison { 0 } };
        Operon::BalancedTreeCreator Creator { &Pset, Vars, 0.0, 10 };
        Operon::UniformTreeInitializer TreeInit { &Creator };
        Operon::UniformCoefficientInitializer CoeffInit;
        Operon::GeneticAlgorithmConfig Config { MakeConfig(PopSize, PoolSize) };
        Operon::GeneticProgrammingAlgorithm Gp { Config, &Problem, &TreeInit, &CoeffInit, &Generator, &Reinserter };
        Operon::DeductiveSorter Sorter;
        Operon::NSGA2 Nsga { Config, &Problem, &TreeInit, &CoeffInit, &Generator, &Reinserter, &Sorter };

        GaBaseFixture()
        {
            Problem.SetTrainingRange({ 0, 10 });
            Problem.SetTestRange({ 0, 10 });
            Problem.SetTarget("Y");
            // Unparameterized, TreeInit's distribution defaults to an unbounded
            // param_type, so it can request an arbitrarily large tree length.
            TreeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
            OnePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            // MultiMutation needs at least one operator or it indexes an empty vector.
            Mutator.Add(&OnePoint, 1.0);
        }
    };

} // namespace

TEST_CASE("RestoreIndividuals maps parents and offspring spans correctly", "[algorithms]")
{
    GaBaseFixture f;
    constexpr auto popSize = GaBaseFixture::PopSize;
    constexpr auto poolSize = GaBaseFixture::PoolSize;
    constexpr auto total = popSize + poolSize;

    // Give each individual a distinct fitness so we can verify which span they land in.
    std::vector<Operon::Individual> inds(total);
    for (std::size_t i = 0; i < total; ++i) {
        inds[i].Fitness.resize(1);
        inds[i].Fitness[0] = static_cast<Operon::Scalar>(i);
    }

    f.Gp.RestoreIndividuals(inds);

    auto parents = f.Gp.Parents();
    auto offspring = f.Gp.Offspring();

    REQUIRE(parents.size() == popSize);
    REQUIRE(offspring.size() == poolSize);

    for (std::size_t i = 0; i < popSize; ++i) {
        CHECK(parents[i].Fitness[0] == static_cast<Operon::Scalar>(i));
    }
    for (std::size_t i = 0; i < poolSize; ++i) {
        CHECK(offspring[i].Fitness[0] == static_cast<Operon::Scalar>(popSize + i));
    }
}

TEST_CASE(
    "Copying a GeneticAlgorithmBase-derived object rebinds Parents()/Offspring() to its own storage", "[algorithms]")
{
    GaBaseFixture f;
    constexpr auto popSize = GaBaseFixture::PopSize;
    constexpr auto poolSize = GaBaseFixture::PoolSize;
    constexpr auto total = popSize + poolSize;

    std::vector<Operon::Individual> inds(total);
    for (std::size_t i = 0; i < total; ++i) {
        inds[i].Fitness.resize(1);
        inds[i].Fitness[0] = static_cast<Operon::Scalar>(i);
    }
    f.Gp.RestoreIndividuals(inds);
    f.Gp.RequestStop(); // exercise StopRequested() being copied too

    Operon::GeneticProgrammingAlgorithm copy { f.Gp }; // NOLINT(performance-unnecessary-copy-initialization)

    // Parents()/Offspring() must point into the *copy*'s own Individuals(),
    // not the original's - otherwise they dangle once the original that
    // still owns that storage goes out of scope, or alias it while both are
    // alive. std::addressof on the first element is the simplest way to
    // check "same underlying storage" without relying on span internals.
    CHECK(std::addressof(copy.Parents()[0]) == std::addressof(copy.Individuals()[0]));
    CHECK(std::addressof(copy.Offspring()[0]) == std::addressof(copy.Individuals()[popSize]));
    CHECK(std::addressof(copy.Parents()[0]) != std::addressof(f.Gp.Parents()[0]));

    // Content and stop-flag state are still copied correctly.
    REQUIRE(copy.Parents().size() == popSize);
    REQUIRE(copy.Offspring().size() == poolSize);
    for (std::size_t i = 0; i < popSize; ++i) {
        CHECK(copy.Parents()[i].Fitness[0] == static_cast<Operon::Scalar>(i));
    }
    for (std::size_t i = 0; i < poolSize; ++i) {
        CHECK(copy.Offspring()[i].Fitness[0] == static_cast<Operon::Scalar>(popSize + i));
    }
    CHECK(copy.StopRequested());
}

TEST_CASE("ReportCallback returning true stops the run before the next generation", "[algorithms]")
{
    GaBaseFixture f; // Config.Generations == 1
    Operon::RandomGenerator rng { 42 };

    std::size_t calls = 0;
    f.Gp.Run(
        rng,
        [&]() -> bool {
            ++calls;
            return true;
        },
        /*threads=*/1);

    // The report fired during init, before the loop body ever ran, so the
    // generation counter (only incremented inside the body) never advanced.
    CHECK(calls == 1);
    CHECK(f.Gp.Generation() == 0);
}

#if defined(__cpp_lib_move_only_function)
// ReportCallback is Operon::MoveOnlyFunction<bool()>, which resolves to
// std::move_only_function on toolchains that implement it - only there can
// this test's move-only capture actually compile (the std::function
// fallback used elsewhere cannot hold a non-copyable closure).
TEST_CASE("ReportCallback accepts a move-only capture (unique_ptr progress sink)", "[algorithms]")
{
    GaBaseFixture f; // Config.Generations == 1
    Operon::RandomGenerator rng { 42 };

    // `calls` is a plain local counter, alive on this TEST_CASE's own stack
    // frame independent of the callback's lifetime, so reading it after
    // Run() returns is safe. `sink` is the move-only capture under test -
    // Run() consumes it (and destroys it, freeing its heap allocation) as
    // part of tearing down the callback, so nothing below reads through it.
    std::size_t calls = 0;
    auto sink = std::make_unique<std::size_t>(0);
    Operon::ReportCallback report = [&calls, sink = std::move(sink)]() mutable -> bool {
        ++calls;
        ++*sink;
        return true;
    };
    f.Gp.Run(rng, std::move(report), /*threads=*/1);

    CHECK(calls == 1); // the moved-in sink was actually reached, not just moved and dropped
    CHECK(f.Gp.Generation() == 0);
}
#endif

TEST_CASE("ReportCallback returning false lets the run reach the configured generations", "[algorithms]")
{
    GaBaseFixture f; // Config.Generations == 1
    Operon::RandomGenerator rng { 42 };

    std::size_t calls = 0;
    f.Gp.Run(
        rng,
        [&]() -> bool {
            ++calls;
            return false;
        },
        /*threads=*/1);

    CHECK(f.Gp.Generation() == f.Config.Generations);
    CHECK(calls == 2); // one report from init, one from the single generation's body
}

TEST_CASE("NSGA2: ReportCallback returning true stops the run before the next generation", "[algorithms]")
{
    GaBaseFixture f; // Config.Generations == 1
    Operon::RandomGenerator rng { 42 };

    std::size_t calls = 0;
    f.Nsga.Run(
        rng,
        [&]() -> bool {
            ++calls;
            return true;
        },
        /*threads=*/1);

    CHECK(calls == 1);
    CHECK(f.Nsga.Generation() == 0);
}

TEST_CASE("NSGA2: ReportCallback returning false lets the run reach the configured generations", "[algorithms]")
{
    GaBaseFixture f; // Config.Generations == 1
    Operon::RandomGenerator rng { 42 };

    std::size_t calls = 0;
    f.Nsga.Run(
        rng,
        [&]() -> bool {
            ++calls;
            return false;
        },
        /*threads=*/1);

    CHECK(f.Nsga.Generation() == f.Config.Generations);
    CHECK(calls == 2); // one report from init, one from the single generation's body
}

TEST_CASE("NSGA2: keep-best and replace-worst reinserters produce different results", "[algorithms][nsga2]")
{
    // Regression test for the reinserter-delegation fix: before it, NSGA2
    // called only ReinserterBase::Sort (ignoring the concrete reinserter's
    // operator()), so --reinserter keep-best vs replace-worst was silently
    // inert. This runs identical NSGA2 setups (same seed, same everything
    // except the reinserter) and asserts the final populations differ.
    Operon::Dataset ds { MakeDataset() };
    Operon::Problem problem { gsl::not_null<Operon::Dataset*>(&ds) };
    problem.SetTrainingRange({ 0, 10 });
    problem.SetTestRange({ 0, 10 });
    problem.SetTarget("Y");
    Operon::PrimitiveSet pset { MakePset() };
    std::vector<Operon::Hash> vars { ds.GetVariable("X1")->Hash };

    DTable dtable;
    Operon::Evaluator<DTable> rmseEval { &problem, &dtable };
    Operon::TreePropertyEvaluator lenEval { &problem,
        [](Operon::Tree const& tree) { return static_cast<Operon::Scalar>(tree.Length()); }, /*normalizer=*/20 };
    Operon::MultiEvaluator multiEval { &problem };
    multiEval.Add(&rmseEval);
    multiEval.Add(&lenEval);

    Operon::SubtreeCrossover crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
    onePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
    Operon::MultiMutation mutator;
    mutator.Add(&onePoint, 1.0);

    Operon::CrowdedComparison comp;
    Operon::TournamentSelector femSel { comp };
    Operon::TournamentSelector maleSel { comp };
    Operon::BasicOffspringGenerator generator { &multiEval, &crossover, &mutator, &femSel, &maleSel };

    Operon::BalancedTreeCreator creator { &pset, vars, 0.0, 10 };
    Operon::UniformTreeInitializer treeInit { &creator };
    treeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
    Operon::UniformCoefficientInitializer coeffInit;
    Operon::DeductiveSorter sorter;

    auto config = MakeConfig(/*popSize=*/20, /*poolSize=*/20);
    config.Generations = 3;

    Operon::KeepBestReinserter keepBest { comp };
    Operon::NSGA2 nsgaKeepBest { config, &problem, &treeInit, &coeffInit, &generator, &keepBest, &sorter };
    Operon::RandomGenerator rngKeepBest { 42 };
    nsgaKeepBest.Run(rngKeepBest, nullptr, /*threads=*/1);

    Operon::ReplaceWorstReinserter replaceWorst { comp };
    Operon::NSGA2 nsgaReplaceWorst { config, &problem, &treeInit, &coeffInit, &generator, &replaceWorst, &sorter };
    Operon::RandomGenerator rngReplaceWorst { 42 };
    nsgaReplaceWorst.Run(rngReplaceWorst, nullptr, /*threads=*/1);

    auto keepBestFitness = std::vector<Operon::Scalar> {};
    for (auto const& ind : nsgaKeepBest.Individuals()) {
        keepBestFitness.insert(keepBestFitness.end(), ind.Fitness.begin(), ind.Fitness.end());
    }
    auto replaceWorstFitness = std::vector<Operon::Scalar> {};
    for (auto const& ind : nsgaReplaceWorst.Individuals()) {
        replaceWorstFitness.insert(replaceWorstFitness.end(), ind.Fitness.begin(), ind.Fitness.end());
    }

    REQUIRE(keepBestFitness.size() == replaceWorstFitness.size());
    CHECK(keepBestFitness != replaceWorstFitness);
}

TEST_CASE("Generation/Elapsed/IsFitted return by value for const objects, by reference for mutable, regardless of "
          "value category",
    "[algorithms]")
{
    // The pre-deducing-this overloads were never ref-qualified, so the const
    // overload returned by value for both lvalues and rvalues, and the
    // non-const overload returned a reference for both. A naive `auto&&`
    // deducing-this replacement gets the const-rvalue case wrong (a
    // reference into a temporary instead of a safe copy) - this locks in
    // that all four Self x value-category combinations still match the
    // original behavior exactly.
    using Algo = Operon::GeneticProgrammingAlgorithm;

    static_assert(std::is_same_v<decltype(std::declval<Algo&>().Generation()), std::size_t&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&>().Generation()), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Algo&&>().Generation()), std::size_t&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&&>().Generation()), std::size_t>);

    static_assert(std::is_same_v<decltype(std::declval<Algo&>().Elapsed()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&>().Elapsed()), double>);
    static_assert(std::is_same_v<decltype(std::declval<Algo&&>().Elapsed()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&&>().Elapsed()), double>);

    static_assert(std::is_same_v<decltype(std::declval<Algo&>().IsFitted()), bool&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&>().IsFitted()), bool>);
    static_assert(std::is_same_v<decltype(std::declval<Algo&&>().IsFitted()), bool&>);
    static_assert(std::is_same_v<decltype(std::declval<Algo const&&>().IsFitted()), bool>);

    SUCCEED("all four Self x value-category combinations verified at compile time");
}

namespace {

    // Self-contained GP/NSGA2 setup with a configurable population/pool size and an evaluation budget, used to drive
    // runs into (or away from) a stop condition in the middle of a generation.
    struct BudgetedRun {
        Operon::Dataset Ds { MakeDataset() };
        Operon::Problem Problem { gsl::not_null<Operon::Dataset*>(&Ds) };
        Operon::PrimitiveSet Pset { MakePset() };
        std::vector<Operon::Hash> Vars { Ds.GetVariable("X1")->Hash };

        DTable Dtable;
        Operon::Evaluator<DTable> Evaluator { &Problem, &Dtable };
        Operon::SubtreeCrossover Crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
        Operon::MultiMutation Mutator;
        Operon::TournamentSelector FemSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::TournamentSelector MaleSel { Operon::SingleObjectiveComparison { 0 } };
        Operon::BasicOffspringGenerator Generator { &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
        Operon::KeepBestReinserter KeepBest { Operon::SingleObjectiveComparison { 0 } };
        Operon::ReplaceWorstReinserter ReplaceWorst { Operon::SingleObjectiveComparison { 0 } };
        Operon::BalancedTreeCreator Creator { &Pset, Vars, 0.0, 10 };
        Operon::UniformTreeInitializer TreeInit { &Creator };
        Operon::UniformCoefficientInitializer CoeffInit;
        Operon::DeductiveSorter Sorter;
        Operon::GeneticAlgorithmConfig Config {};

        BudgetedRun(std::size_t popSize, std::size_t poolSize, std::size_t generations, std::size_t budget)
        {
            Config.PopulationSize = popSize;
            Config.PoolSize = poolSize;
            Config.Generations = generations;
            Problem.SetTrainingRange({ 0, 10 });
            Problem.SetTestRange({ 0, 10 });
            Problem.SetTarget("Y");
            TreeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
            OnePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            Mutator.Add(&OnePoint, 1.0);
            Evaluator.SetBudget(budget);
        }
    };

    // What a report callback can observe about the population.
    struct Snapshot {
        std::size_t Generation {};
        std::vector<Operon::Scalar> Fitness;
        std::vector<std::size_t> Lengths;
        std::vector<Operon::Hash> Hashes;
        auto operator==(Snapshot const&) const -> bool = default;
    };

    auto Capture(Operon::GeneticAlgorithmBase const& algo) -> Snapshot
    {
        Snapshot s;
        s.Generation = algo.Generation();
        for (auto const& ind : algo.Parents()) {
            s.Fitness.push_back(ind[0]);
            s.Lengths.push_back(ind.Genotype.Length());
            s.Hashes.push_back(ind.Genotype.HashValue());
        }
        return s;
    }

    // pop = pool = 16, budget 24: the 16 initial evaluations leave room for 8 offspring, so generation 1 is cut short.
    constexpr std::size_t PartialPop = 16;
    constexpr std::size_t PartialBudget = PartialPop + 8;

} // namespace

TEST_CASE("GP: a generation cut short by the evaluation budget is neither reinserted, counted nor reported",
    "[algorithms][partial-generation]")
{
    BudgetedRun f { PartialPop, PartialPop, /*generations=*/100, PartialBudget };
    Operon::GeneticProgrammingAlgorithm gp { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator,
        &f.KeepBest };
    Operon::RandomGenerator rng { 7 };

    std::vector<Snapshot> reports;
    gp.Run(
        rng,
        [&]() -> bool {
            reports.push_back(Capture(gp));
            return false;
        },
        /*threads=*/1);

    REQUIRE(reports.size() == 1); // only the initial population: generation 1 never completed
    CHECK(reports.front().Generation == 0);
    CHECK(gp.Generation() == 0);
    CHECK(Capture(gp) == reports.front()); // the population is the last reported one, untouched

    REQUIRE(gp.IncompleteGeneration().has_value());
    auto const progress = *gp.IncompleteGeneration();
    CHECK(progress.PoolSize == PartialPop);
    CHECK(progress.Completed == PartialBudget - PartialPop);
    CHECK(progress.Attempted >= progress.Completed);
    CHECK(f.Evaluator.TotalEvaluations() == PartialBudget);

    SECTION("the incomplete state is copied with the algorithm and cleared by Reset()")
    {
        Operon::GeneticProgrammingAlgorithm copy { gp };
        REQUIRE(copy.IncompleteGeneration().has_value());
        CHECK(copy.IncompleteGeneration()->Completed == progress.Completed);
        copy.Reset();
        CHECK_FALSE(copy.IncompleteGeneration().has_value());
    }
}

TEST_CASE(
    "GP: replace-worst does not inject unfilled pool slots into the population", "[algorithms][partial-generation]")
{
    // Before the generation guard, the 8 pool slots that were never filled (default-constructed individuals: empty
    // genotype, maximal fitness) were swapped into the population by ReplaceWorstReinserter.
    for (std::size_t const threads : { 1UL, 4UL }) {
        BudgetedRun f { PartialPop, PartialPop, /*generations=*/100, PartialBudget };
        Operon::GeneticProgrammingAlgorithm gp { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator,
            &f.ReplaceWorst };
        Operon::RandomGenerator rng { 7 };

        std::optional<Snapshot> initial;
        gp.Run(
            rng,
            [&]() -> bool {
                if (!initial) {
                    initial = Capture(gp);
                }
                return false;
            },
            threads);

        INFO("threads = " << threads);
        REQUIRE(initial.has_value());
        CHECK(Capture(gp) == *initial);
        CHECK(gp.Generation() == 0);
        CHECK(gp.IncompleteGeneration().has_value());
        for (auto const& ind : gp.Parents()) {
            CHECK(ind.Genotype.Length() > 0);
            CHECK(ind[0] < Operon::EvaluatorBase::ErrMax);
        }
    }
}

TEST_CASE("GP: generations that fill their pool before the budget is reached are reinserted, counted and reported",
    "[algorithms][partial-generation]")
{
    // init (16) + 3 complete generations (3 * 16) + 8 evaluations: generation 4 is cut short.
    BudgetedRun f { PartialPop, PartialPop, /*generations=*/100, 4 * PartialPop + 8 };
    Operon::GeneticProgrammingAlgorithm gp { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator,
        &f.KeepBest };
    Operon::RandomGenerator rng { 7 };

    std::vector<std::size_t> reported;
    gp.Run(
        rng,
        [&]() -> bool {
            reported.push_back(gp.Generation());
            return false;
        },
        /*threads=*/1);

    CHECK(reported == std::vector<std::size_t> { 0, 1, 2, 3 });
    CHECK(gp.Generation() == 3);
    REQUIRE(gp.IncompleteGeneration().has_value());
    CHECK(gp.IncompleteGeneration()->Completed == 8);
}

TEST_CASE("NSGA2: IncompleteGeneration() reports the cut-short generation", "[algorithms][partial-generation]")
{
    // Evaluations in flight when the budget is reached can overshoot it by less than the thread count, which cannot
    // fill the 16 - 8 = 8 missing pool slots, so the generation is discarded at every thread count.
    for (std::size_t const threads : { 1UL, 4UL }) {
        BudgetedRun f { PartialPop, PartialPop, /*generations=*/100, PartialBudget };
        Operon::NSGA2 nsga { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator, &f.KeepBest, &f.Sorter };
        Operon::RandomGenerator rng { 7 };

        std::optional<Snapshot> initial;
        std::size_t reports = 0;
        nsga.Run(
            rng,
            [&]() -> bool {
                ++reports;
                if (!initial) {
                    initial = Capture(nsga);
                }
                return false;
            },
            threads);

        INFO("threads = " << threads);
        CHECK(reports == 1);
        CHECK(nsga.Generation() == 0);
        REQUIRE(initial.has_value());
        CHECK(Capture(nsga) == *initial);
        REQUIRE(nsga.IncompleteGeneration().has_value());
        CHECK(nsga.IncompleteGeneration()->PoolSize == PartialPop);
        CHECK(nsga.IncompleteGeneration()->Completed >= PartialBudget - PartialPop);
        CHECK(nsga.IncompleteGeneration()->Completed < PartialPop);
        if (threads == 1) {
            CHECK(nsga.IncompleteGeneration()->Completed == PartialBudget - PartialPop);
        }
    }
}

TEST_CASE("GP: runs that stop at the generation limit are complete and identical across seeds and thread counts",
    "[algorithms][partial-generation]")
{
    // No budget or time limit is reached, so every generation is complete: the trajectory (all reports) and the final
    // population must not depend on the thread count, and no incomplete generation may be reported.
    constexpr std::size_t generations = 6;
    auto trajectory = [&](std::uint64_t seed, std::size_t threads) {
        BudgetedRun f { 24, 24, generations, /*budget=*/std::numeric_limits<std::size_t>::max() };
        Operon::GeneticProgrammingAlgorithm gp { f.Config, &f.Problem, &f.TreeInit, &f.CoeffInit, &f.Generator,
            &f.ReplaceWorst };
        Operon::RandomGenerator rng { seed };
        std::vector<Snapshot> reports;
        gp.Run(
            rng,
            [&]() -> bool {
                reports.push_back(Capture(gp));
                return false;
            },
            threads);
        CHECK(gp.Generation() == generations);
        CHECK_FALSE(gp.IncompleteGeneration().has_value());
        reports.push_back(Capture(gp));
        return reports;
    };

    for (std::uint64_t const seed : { 1ULL, 2ULL, 3ULL, 42ULL }) {
        INFO("seed = " << seed);
        auto const reference = trajectory(seed, 1);
        REQUIRE(reference.size() == generations + 2); // init + 6 generations + final snapshot
        for (std::size_t g = 0; g <= generations; ++g) {
            CHECK(reference[g].Generation == g);
        }
        CHECK(trajectory(seed, 4) == reference);
        CHECK(trajectory(seed, 1) == reference); // repeatable
    }
}

} // namespace Operon::Test
