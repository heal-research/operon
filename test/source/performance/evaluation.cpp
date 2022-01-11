// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>
#include <thread>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"

#include "nanobench.h"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter/rank_sort.hpp"
#include "operon/operators/reinserter.hpp"

namespace Operon::Test {
    auto TotalNodes(const std::vector<Tree>& trees) -> std::size_t {
#ifdef _MSC_VER
        auto totalNodes = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto totalNodes = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        return totalNodes;
    }

    namespace nb = ankerl::nanobench;

    template <typename T>
    void Evaluate(tf::Executor& executor, Interpreter const& interpreter, std::vector<Tree> const& trees, Dataset const& ds, Range range)
    {
        tf::Taskflow taskflow;
        taskflow.for_each(trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
        executor.run(taskflow).wait();
    }

    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Evaluation performance")
    {
        constexpr size_t n = 1000;
        constexpr size_t maxLength = 100;
        constexpr size_t maxDepth = 1000;
        constexpr size_t nrow = 10000;
        constexpr size_t ncol = 10;

        constexpr size_t minEpochIterations = 5;

        Eigen::Matrix<Operon::Scalar, -1, -1> data = decltype(data)::Random(nrow, ncol);
        Operon::RandomGenerator rd(1234);
        auto ds = Dataset(data);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        //Range range = { 0, ds.Rows() };
        Range range = { 0, nrow };

        PrimitiveSet pset;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { pset, inputs };

        std::vector<Tree> trees(n);

        Interpreter interpreter;

        auto test = [&](tf::Executor& executor, nb::Bench& b, PrimitiveSetConfig cfg, const std::string& name) {
            pset.SetConfig(cfg);
            for (auto t : { NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul }) {
                pset.SetMinMaxArity(Node(t).HashValue, 2, 2);
            }
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

            auto totalOps = TotalNodes(trees) * range.Size();
            b.batch(totalOps);
            b.run(name, [&]() { Evaluate<Operon::Scalar>(executor, interpreter, trees, ds, range); });
        };

        SUBCASE("arithmetic") {
            // single-thread
            nb::Bench b;
            b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + exp") {
            nb::Bench b;
            b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Exp, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + log") {
            nb::Bench b;
            b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Log, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sin") {
            nb::Bench b;
            b.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sin, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cos") {
            nb::Bench b;
            b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cos, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + tan") {
            nb::Bench b;
            b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Tan, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sqrt") {
            nb::Bench b;
            b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sqrt, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cbrt") {
            nb::Bench b;
            b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cbrt, fmt::format("N = {}", i));
            }
        }
    }

    TEST_CASE("Evaluator performance")
    {
        const size_t n         = 1000;
        const size_t maxLength = 100;
        const size_t maxDepth  = 1000;

        constexpr size_t nrow = 10000;
        constexpr size_t ncol = 10;

        Operon::RandomGenerator rd(1234);
        Eigen::Matrix<Operon::Scalar, -1, -1> data = decltype(data)::Random(nrow, ncol);
        auto ds = Dataset(data);

        auto variables = ds.Variables();
        auto target = variables.back().Name;
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& v) { return v.Name != target; });
        Range range = { 0, ds.Rows() };

        auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(range).TestRange(range);
        problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };

        std::vector<Tree> trees(n);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        std::vector<Individual> individuals(n);
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Genotype = trees[i];
        }

        nb::Bench b;
        b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);

        Operon::Vector<Operon::Scalar> buf(range.Size());

        auto test = [&](std::string const& name, EvaluatorBase&& evaluator) {
            evaluator.SetLocalOptimizationIterations(0);
            evaluator.SetBudget(std::numeric_limits<size_t>::max());
            tf::Executor executor(std::thread::hardware_concurrency());
            tf::Taskflow taskflow;

            std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());
            double sum{0};
            taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{}, [&](Operon::Individual& ind) {
                auto id = executor.this_worker_id();
                if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); } 
                return evaluator(rd, ind, slots[id]).front();
            });

            b.batch(totalNodes * range.Size()).epochs(10).epochIterations(100).run(name, [&]() {
                sum = 0;
                executor.run(taskflow).wait();
                return sum;
            });
        };

        Interpreter interpreter;
        test("c2",        Operon::Evaluator<Operon::C2, false>(problem, interpreter));
        test("c2 + ls",   Operon::Evaluator<Operon::C2, true>(problem, interpreter));
        test("r2",        Operon::Evaluator<Operon::R2, false>(problem, interpreter));
        test("r2 + ls",   Operon::Evaluator<Operon::R2, true>(problem, interpreter));
        test("nmse",      Operon::Evaluator<Operon::NMSE, false>(problem, interpreter));
        test("nmse + ls", Operon::Evaluator<Operon::NMSE, true>(problem, interpreter));
        test("mae",       Operon::Evaluator<Operon::MAE, false>(problem, interpreter));
        test("mae + ls",  Operon::Evaluator<Operon::MAE, true>(problem, interpreter));
        test("mse",       Operon::Evaluator<Operon::MSE, false>(problem, interpreter));
        test("mse + ls",  Operon::Evaluator<Operon::MSE, true>(problem, interpreter));
    }

    TEST_CASE("NSGA2")
    {
        auto ds = Dataset("/home/bogdb/projects/operon-archive/data/Friedman-I.csv", /*hasHeader=*/true);

        std::vector<Variable> inputs;
        const auto *target = "Y";
        auto variables = ds.Variables();
        std::copy_if(variables.begin(),
                variables.end(),
                std::back_inserter(inputs),
                [&](auto const& var) { return var.Name != target; });

        Range trainingRange(0, ds.Rows() / 2);
        Range testRange(ds.Rows() / 2, ds.Rows());

        auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(trainingRange).TestRange(testRange);
        problem.GetPrimitiveSet().SetConfig(PrimitiveSet::Arithmetic);
        BalancedTreeCreator creator(problem.GetPrimitiveSet(), problem.InputVariables(), 0.0);

        const size_t maxDepth = 1000;
        const size_t maxLength = 50;
        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        UniformTreeInitializer initializer(creator);
        initializer.ParameterizeDistribution(amin + 1, maxLength);
        initializer.SetMinDepth(1);
        initializer.SetMaxDepth(maxDepth);

        UniformCoefficientInitializer coeffInit;
        initializer.ParameterizeDistribution(0UL, 1UL);

        const double crossoverInternalProbability = 0.9;

        auto crossover = SubtreeCrossover { crossoverInternalProbability, maxDepth, maxLength };
        auto mutator = MultiMutation {};
        auto onePoint = OnePointMutation<std::uniform_real_distribution<Operon::Scalar>> {};
        onePoint.ParameterizeDistribution(Operon::Scalar{-2}, Operon::Scalar{+2});
        auto changeVar = ChangeVariableMutation { problem.InputVariables() };
        auto changeFunc = ChangeFunctionMutation { problem.GetPrimitiveSet() };
        auto replaceSubtree = ReplaceSubtreeMutation { creator, maxDepth, maxLength };
        auto insertSubtree = InsertSubtreeMutation { creator, maxDepth, maxLength, problem.GetPrimitiveSet() };
        auto removeSubtree = RemoveSubtreeMutation { problem.GetPrimitiveSet() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(removeSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        Interpreter interpreter;
        SquaredCorrelationEvaluator c2eval(problem, interpreter);
        LengthEvaluator lenEval(problem);

        MultiEvaluator evaluator(problem);
        evaluator.Add(c2eval);
        evaluator.Add(lenEval);

        CrowdedComparison comp;
        TournamentSelector selector(comp);
        KeepBestReinserter reinserter(comp);

        BasicOffspringGenerator generator(evaluator, crossover, mutator, selector, selector);
        Operon::RandomGenerator random(1234);

        RankSorter sorter;

        GeneticAlgorithmConfig config{};
        config.Generations = 100;
        config.PopulationSize = 1000; 
        config.PoolSize = 1000;
        config.Evaluations = 1000000; 
        config.Iterations = 0; 
        config.CrossoverProbability = 1.0; 
        config.MutationProbability = 0.25; 
        config.TimeLimit = ~size_t{0}; 
        config.Seed = random();

        Operon::NSGA2 gp { problem, config, initializer, coeffInit, generator, reinserter, sorter };
        //tf::Executor executor;

        auto report = [&]() {
            fmt::print("generation {}\n", gp.Generation());
        };

        gp.Run(random, report, 10);
    }
} // namespace Operon

