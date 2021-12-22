// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>
#include <interpreter/dispatch_table.hpp>
#include <thread>
#include <chrono>

#include "core/dataset.hpp"
#include "interpreter/interpreter.hpp"
#include "core/pset.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"

#include "nanobench.h"

#include "taskflow/taskflow.hpp"

namespace Operon {
namespace Test {
    std::size_t TotalNodes(const std::vector<Tree>& trees) {
#ifdef _MSC_VER
        auto totalNodes = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto totalNodes = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        return totalNodes;
    }

    namespace nb = ankerl::nanobench;

    template <typename T>
    void Evaluate(tf::Executor& executor, std::vector<Tree> const& trees, Dataset const& ds, Range range)
    {
        Interpreter interpreter;
        tf::Taskflow taskflow;
        std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());
        taskflow.for_each(trees.begin(), trees.end(), [&](auto const& tree) {
            auto id = executor.this_worker_id();
            if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); }
            interpreter.Evaluate<T>(tree, ds, range, slots[id]);
        });
        executor.run(taskflow).wait();
    }

    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Evaluation performance")
    {
        size_t n = 1000;
        size_t maxLength = 100;
        size_t maxDepth = 1000;

        Operon::RandomGenerator rd(1234);
        auto ds = Dataset("../data/Friedman-I.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        //Range range = { 0, ds.Rows() };
        Range range = { 0, 10000 };

        PrimitiveSet pset;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { pset, inputs };

        std::vector<Tree> trees(n);

        auto test = [&](tf::Executor& executor, nb::Bench& b, PrimitiveSetConfig cfg, const std::string& name) {
            pset.SetConfig(cfg);
            for (auto t : { NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul }) {
                pset.SetMinMaxArity(Node(t).HashValue, 2, 2);
            }
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });
            auto totalOps = TotalNodes(trees) * range.Size();
            b.batch(totalOps);
            auto start = std::chrono::high_resolution_clock::now();
            // we fix the epochs and epoch iterations here so we can use them to approximate the elapsed time manually 
            b.epochs(10).epochIterations(100).run(name, [&]() { Evaluate<Operon::Scalar>(executor, trees, ds, range); });
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) / 1e6;
            double node_evals_psec = b.batch() * static_cast<double>(b.epochs() * b.epochIterations()) / duration;
            fmt::print("node evals / s: {:L}\n", node_evals_psec);
        };

        SUBCASE("arithmetic") {
            // single-thread
            nb::Bench b;
            b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + exp") {
            nb::Bench b;
            b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Exp, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + log") {
            nb::Bench b;
            b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Log, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sin") {
            nb::Bench b;
            b.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sin, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cos") {
            nb::Bench b;
            b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cos, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + tan") {
            nb::Bench b;
            b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Tan, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sqrt") {
            nb::Bench b;
            b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sqrt, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cbrt") {
            nb::Bench b;
            b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(5);
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

        Operon::RandomGenerator rd(1234);
        auto ds = Dataset("../data/Friedman-I.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& v) { return v.Name != target; });
        Range range = { 0, ds.Rows() };

        auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(range).TestRange(range);
        problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };

        std::vector<Tree> trees(n);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        std::vector<Individual> individuals;
        individuals.reserve(n);
        std::transform(trees.begin(), trees.end(), std::back_inserter(individuals), [](auto const& tree) {
            Operon::Individual ind;
            ind.Genotype = tree;
            return ind;
        });

        nb::Bench b;
        b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);

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

            auto start = std::chrono::high_resolution_clock::now();
            b.batch(totalNodes * range.Size()).epochs(10).epochIterations(100).run(name, [&]() {
                sum = 0;
                executor.run(taskflow).wait();
                return sum;
            });
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) / 1e6;
            double node_evals_psec = b.batch() * static_cast<double>(b.epochs() * b.epochIterations()) / duration;
            fmt::print("node evals / s: {:L}\n", node_evals_psec);
        };

        Interpreter interpreter;
        test("r-squared",      Operon::Evaluator<Operon::R2, false>(problem, interpreter));
        test("r-squared + ls", Operon::Evaluator<Operon::R2, true>(problem, interpreter));
        test("nmse",           Operon::Evaluator<Operon::NMSE, false>(problem, interpreter));
        test("nmse + ls",      Operon::Evaluator<Operon::NMSE, true>(problem, interpreter));
        test("mae",            Operon::Evaluator<Operon::MAE, false>(problem, interpreter));
        test("mae + ls",       Operon::Evaluator<Operon::MAE, true>(problem, interpreter));
        test("mse",            Operon::Evaluator<Operon::MSE, false>(problem, interpreter));
        test("mse + ls",       Operon::Evaluator<Operon::MSE, true>(problem, interpreter));
    }
} // namespace Test
} // namespace Operon

