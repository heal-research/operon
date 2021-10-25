// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <core/individual.hpp>
#include <doctest/doctest.h>
#include <thread>

#include "algorithms/nsga2.hpp"
#include "core/dataset.hpp"
#include "core/pset.hpp"
#include "interpreter/interpreter.hpp"
#include "interpreter/dispatch_table.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/reinserter/keepbest.hpp"
#include "operators/selection.hpp"
#include "operators/non_dominated_sorter.hpp"


#include "nanobench.h"

#include "taskflow/taskflow.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

template<typename Dist> 
std::vector<Individual> InitializePop(Operon::RandomGenerator& random, Dist& dist, size_t n, size_t m) {
    std::vector<Individual> individuals(n);
    for (size_t i = 0; i < individuals.size(); ++i) {
        individuals[i].Fitness.resize(m);

        for (size_t j = 0; j < m; ++j) {
            individuals[i][j] = static_cast<Operon::Scalar>(dist(random));
        }
        ENSURE(individuals[i].Fitness.size() == m);
    }

    return individuals;
};

template<typename Sorter, typename Dist>
void Test(std::string const& str, nb::Bench& bench, Sorter& sorter, Operon::RandomGenerator& rd, Dist& dist, std::vector<size_t> const& ns, std::vector<size_t> const& ms)
{
    for (auto m : ms) {
        for (auto n : ns) {
            auto pop = InitializePop(rd, dist, n, m);
            bench.run(fmt::format("{}: n = {}, m = {}", str, n, m), [&]{
                std::stable_sort(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs.LexicographicalCompare(rhs); });
                std::vector<Individual> dup; dup.reserve(pop.size());
                auto r = std::unique(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) {
                        auto res = lhs == rhs;
                        if (res) { dup.push_back(rhs); }
                        return res;
                        });
                std::copy_n(std::make_move_iterator(dup.begin()), dup.size(), r);
                Operon::Span<Individual const> s(pop.begin(), r);
                return sorter(s).size();
            });
        }
    }
    //bench.render(ankerl::nanobench::templates::csv(), std::cout);
}

TEST_CASE("non-dominated sort" * doctest::test_suite("[performance]"))
{
    auto initializePop = [](Operon::RandomGenerator& random, auto& dist, size_t n, size_t m) {
        std::vector<Individual> individuals(n);
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Fitness.resize(m);

            for (size_t j = 0; j < m; ++j) {
                individuals[i][j] = dist(random);
            }
            ENSURE(individuals[i].Fitness.size() == m);
        }

        return individuals;
    };

    Operon::RandomGenerator rd(1234);
    //std::vector<size_t> ms;
    //for (size_t i = 2; i <= 20; ++i) { ms.push_back(i); }
    //std::vector<size_t> ns { 100, 500 };
    //for (size_t i = 1000; i <= 20000; i += 1000) { ns.push_back(i); }
    std::vector<size_t> ms{ 20 };
    std::vector<size_t> ns { 20000 };
    std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
    std::uniform_int_distribution<size_t> integerDist(0, 10);

    nb::Bench bench;
    bench.relative(true).performanceCounters(true).minEpochIterations(50);

    // A: dominate on equal
    // B: no dominate on equal
    DeductiveSorter dsA;
    HierarchicalSorter hsA;
    EfficientSorter<true> ensBsA;
    EfficientSorter<false> ensSsA;
    RankSorter rsA;
    MergeNondominatedSorter msA;


    SUBCASE("point cloud all")
    {
        bench.minEpochIterations(10);
        //Test("DS", bench, dsA, rd, dist, ns, ms);
        //Test("HS", bench, hsA, rd, dist, ns, ms);
        //Test("ENS-BS", bench, ensBsA, rd, dist, ns, ms);
        //Test("ENS-SS", bench, ensSsA, rd, dist, ns, ms);
        //Test("MNDS", bench, msA, rd, dist, ns, ms);
        Test("RS", bench, rsA, rd, dist, ns, ms);
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("point cloud threaded RS") {
        tf::Executor ex;
        tf::Taskflow tf;

        nb::Bench b;
        for (size_t i = 0; i < 16; ++i) {
            tf.emplace([&]{ 
                b = bench;
                b.minEpochIterations(10);
                Test("RS", b, rsA, rd, dist, ns, ms);
            });
        }
        ex.run(tf).wait();
        b.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("statics vs dynamic M")
    {
        auto pop = InitializePop(rd, dist, 5000, 2);
        bench.run("static M", [&]() { return RankSorter{}(pop).size(); });
        bench.run("dynamic M", [&]() { return RankSorter{}(pop).size(); });
    }


    SUBCASE("point cloud RS") { Test("RS", bench, rsA, rd, dist, ns, ms); }
    SUBCASE("point cloud DS") { Test("DS", bench, dsA, rd, dist, ns, ms); }
    SUBCASE("point cloud HS") { Test("HS", bench, hsA, rd, dist, ns, ms); }
    SUBCASE("point cloud ENS-BS") { Test("ENS-BS", bench, ensBsA, rd, dist, ns, ms); }
    SUBCASE("point cloud ENS-SS") { Test("ENS-SS", bench, ensSsA, rd, dist, ns, ms); }
    SUBCASE("point cloud MS") { Test("MNDS", bench, msA, rd, dist, ns, ms); }

    //using F = std::function<std::vector<std::vector<size_t>>(Operon::Span<Operon::Individual const>)>;
    auto check_complexity = [&](size_t m, auto&& sorter) {
        std::vector<size_t> sizes{ 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
        bench.minEpochIterations(10);
        //Operon::RandomGenerator rd(1234);

        for (auto s : sizes) {
            auto pop = initializePop(rd, dist, s, m);
            bench.complexityN(s).run(fmt::format("n = {}", s), [&]() { return sorter(pop).size(); });
        }
        std::cout << bench.complexityBigO() << "\n";
    };

    SUBCASE("M=2")
    {
        check_complexity(2, DeductiveSorter{});
        check_complexity(2, HierarchicalSorter{});
        check_complexity(2, RankSorter{});
    }

    SUBCASE("M=3")
    {
        check_complexity(3, DeductiveSorter{});
        check_complexity(3, HierarchicalSorter{});
        check_complexity(3, RankSorter{});
    }

    auto runNSGA2 = [&](Operon::RandomGenerator& rng, auto sorter, size_t n, size_t m) {
        GeneticAlgorithmConfig config;
        config.Generations          = 1000;
        config.PopulationSize       = n;
        config.PoolSize             = n;
        config.Evaluations          = 1000000;
        config.Iterations           = 0;
        config.CrossoverProbability = 1.0;
        config.MutationProbability  = 0.25;
        config.Seed                 = 42;

        Dataset ds("../data/Poly-10.csv", /* csv has header */ true);
        const std::string target = "Y";

        Range trainingRange { 0, ds.Rows() / 2 };
        Range testRange     { ds.Rows() / 2, ds.Rows() };
        auto problem = Operon::Problem(ds).Inputs(ds.Variables()).Target(target).TrainingRange(trainingRange).TestRange(testRange);

        problem.GetPrimitiveSet().SetConfig(PrimitiveSet::Arithmetic);

        size_t maxDepth{10};
        size_t maxLength{50};

        BalancedTreeCreator creator(problem.GetPrimitiveSet(), problem.InputVariables(), 0.0);
        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        std::uniform_int_distribution<size_t> sizeDistribution(amin + 1, maxLength);
        auto initializer = Initializer { creator, sizeDistribution };
        initializer.MinDepth(1);
        initializer.MaxDepth(1000);

        double crossoverInternalProbability{0.9};
        auto crossover = SubtreeCrossover { crossoverInternalProbability, maxDepth, maxLength };
        auto mutator = MultiMutation {};
        auto onePoint = OnePointMutation {};
        auto changeVar = ChangeVariableMutation { problem.InputVariables() };
        auto changeFunc = ChangeFunctionMutation { problem.GetPrimitiveSet() };
        auto replaceSubtree = ReplaceSubtreeMutation { creator, maxDepth, maxLength };
        auto insertSubtree = InsertSubtreeMutation { creator, maxDepth, maxLength, problem.GetPrimitiveSet() };
        auto removeSubtree = RemoveSubtreeMutation { problem.GetPrimitiveSet() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(replaceSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        DispatchTable ft;
        Interpreter interpreter(ft);

        // fitness evaluators
        RSquaredEvaluator r2eval(problem, interpreter);
        r2eval.SetLocalOptimizationIterations(0);

        // parsimony evaluators
        UserDefinedEvaluator length_eval(problem, [](Operon::RandomGenerator&, Individual& ind) { return Operon::Vector<Operon::Scalar>{ static_cast<Operon::Scalar>(ind.Genotype.Length()) }; });
        UserDefinedEvaluator shape_eval(problem, [](Operon::RandomGenerator&, Individual& ind) { return Operon::Vector<Operon::Scalar>{ static_cast<Operon::Scalar>(ind.Genotype.VisitationLength()) }; });
        UserDefinedEvaluator depth_eval(problem, [](Operon::RandomGenerator&, Individual& ind) { return Operon::Vector<Operon::Scalar>{ static_cast<Operon::Scalar>(ind.Genotype.Depth()) }; });
        UserDefinedEvaluator complexity_eval(problem, [](Operon::RandomGenerator&, Individual& ind) {
                auto const& nodes = ind.Genotype.Nodes();
                auto complexity = ind.Genotype.Length() + 2 * std::count_if(nodes.begin(), nodes.end(), [](auto const& v) { return v.IsVariable(); });
                return Operon::Vector<Operon::Scalar>{ static_cast<Operon::Scalar>(complexity) };
        });

        // random evaluator
        //UserDefinedEvaluator series_eval(problem, [&](Operon::RandomGenerator&, Individual& ind) {
        //    auto a = ind[0];
        //    auto b = ind[1];

        //    Operon::Vector<Operon::Scalar> y {a, b};
        //    for (size_t i = 2; i < m; ++i) {
        //        bool bit = true;
        //        Operon::Scalar p{0};
        //        for (size_t j = 0; j < i; ++j) {
        //            p += bit ? std::sin(a) : std::cos(a);
        //            p += bit ? std::cos(b) : std::sin(b);
        //            bit = !bit; 
        //        }
        //        y.push_back(p);
        //    }
        //    return y;
        //});

        MultiEvaluator eval(problem);
        eval.SetBudget(config.Evaluations);
        eval.Add(r2eval);
        //eval.Add(randomEval);
        if (m > 1) {
            eval.Add(length_eval);
        }
        if (m > 2) {
            eval.Add(shape_eval);
        }
        if (m > 3) {
            eval.Add(depth_eval);
        }
        if (m > 4) {
            eval.Add(complexity_eval);
        }
        ENSURE(m <= 5);

        CrowdedComparison comp;
        TournamentSelector selector(comp);
        BasicOffspringGenerator generator(eval, crossover, mutator, selector, selector);
        KeepBestReinserter reinserter(comp);

        NSGA2 nsga2{ problem, config, initializer, generator, reinserter, sorter };

        tf::Executor executor(1); // use one thread
        nsga2.Run(executor, rng, nullptr);
    };

    auto run_m = [&](std::string const& name, auto& sorter, size_t m) {
        bench = nb::Bench{};
        bench.relative(true).performanceCounters(true).minEpochIterations(5);
        
        Operon::RandomGenerator rng(1234);

        //std::vector<size_t> ns { 100 };
        //for (size_t n = 500; n <= 5000; n += 500) { ns.push_back(n); }

        ns = std::vector<size_t> { 1000 };
        for (size_t n : ns) {
            bench.epochIterations(10).run(fmt::format("name: nsga2 {} n = {} m = {}", name, n, m), [&]() { runNSGA2(rng, sorter, n, m); });
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    };

    SUBCASE("nsga2 m=2 DS") { DeductiveSorter sorter; run_m("DS", sorter, 2); }
    SUBCASE("nsga2 m=2 HS") { HierarchicalSorter sorter; run_m("HS", sorter, 2); }
    SUBCASE("nsga2 m=2 RS1") { RankSorter sorter; run_m("RS", sorter, 2); }
    SUBCASE("nsga2 m=2 ENS-SS") { EfficientSorter<0> sorter; run_m("ENS-SS", sorter, 2); }
    SUBCASE("nsga2 m=2 ENS-BS") { EfficientSorter<1> sorter; run_m("ENS-BS", sorter, 2); }
    SUBCASE("nsga2 m=2 MNDS") { MergeNondominatedSorter sorter; run_m("MNDS", sorter, 2); }

    SUBCASE("nsga2 m=3 DS") { DeductiveSorter sorter; run_m("DS", sorter, 3); }
    SUBCASE("nsga2 m=3 HS") { HierarchicalSorter sorter; run_m("HS", sorter, 3); }
    SUBCASE("nsga2 m=3 RS") { RankSorter sorter; run_m("RS", sorter, 3); }
    SUBCASE("nsga2 m=3 ENS-SS") { EfficientSorter<0> sorter; run_m("ENS-SS", sorter, 3); }
    SUBCASE("nsga2 m=3 ENS-BS") { EfficientSorter<1> sorter; run_m("ENS-BS", sorter, 3); }
    SUBCASE("nsga2 m=3 MNDS1") { MergeNondominatedSorter sorter; run_m("MNDS", sorter, 3); }
}

} // namespace Operon::Test
