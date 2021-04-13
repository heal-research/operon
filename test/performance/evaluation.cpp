/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include <doctest/doctest.h>
#include <execution>
#include <interpreter/dispatch_table.hpp>
#include <tbb/global_control.h>
#include <thread>

#include "core/common.hpp"
#include "core/dataset.hpp"
#include "interpreter/interpreter.hpp"
#include "core/pset.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"

#include "nanobench.h"

namespace Operon {
namespace Test {
    std::size_t TotalNodes(const std::vector<Tree>& trees) {
#ifdef _MSC_VER
        auto totalNodes = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        return totalNodes;
    }

    enum ExecutionPolicy {
        Sequenced,
        Unsequenced,
        ParallelSequenced,
        ParallelUnsequenced
    };

    namespace nb = ankerl::nanobench;

    template <typename T>
    void Evaluate(std::vector<Tree> const& trees, Dataset const& ds, Range range, ExecutionPolicy ep = ExecutionPolicy::ParallelUnsequenced) 
    {
        DispatchTable ft;
        Interpreter interpreter(ft);
        switch(ep) {
            case ExecutionPolicy::Sequenced:
                std::for_each(std::execution::seq, trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
                break;
            case ExecutionPolicy::Unsequenced:
                // use seq because unseq is not yet supported by MSVC
                std::for_each(std::execution::seq, trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
                break;
            case ExecutionPolicy::ParallelSequenced:
                std::for_each(std::execution::par, trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
                break;
            case ExecutionPolicy::ParallelUnsequenced:
                std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
                break;
        }
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
        Range range = { 0, 5000 };

        PrimitiveSet pset;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { pset, inputs };

        std::vector<Tree> trees(n);

        auto test = [&](nb::Bench& b, PrimitiveSetConfig cfg, ExecutionPolicy pol, const std::string& name) {  
            pset.SetConfig(cfg);
            for (auto t : { NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul }) {
                pset.SetMinMaxArity(t, 2, 2);
            }
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

            auto totalOps = TotalNodes(trees) * range.Size();
            b.batch(totalOps);
            b.run(name, [&]() { Evaluate<Operon::Scalar>(trees, ds, range, pol); });
        };

        SUBCASE("arithmetic") {
            // single-thread
            nb::Bench b;
            b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + exp") {
            nb::Bench b;
            b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Exp, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + log") {
            nb::Bench b;
            b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Log, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + sin") {
            nb::Bench b;
            b.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Sin, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + cos") {
            nb::Bench b;
            b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Cos, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + tan") {
            nb::Bench b;
            b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Tan, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + sqrt") {
            nb::Bench b;
            b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Sqrt, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }

        SUBCASE("arithmetic + cbrt") {
            nb::Bench b;
            b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) { 
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, i);
                test(b, PrimitiveSet::Arithmetic | NodeType::Cbrt, i == 1 ? ExecutionPolicy::Unsequenced : ExecutionPolicy::ParallelUnsequenced, fmt::format("{} {}", i, i == 1 ? "thread" : "threads"));
            }
        }
    }

    TEST_CASE("Evaluator performance")
    {
        const size_t n         = 100;
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

        std::vector<Individual> individuals(n);
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Genotype = trees[i];
        }

        DispatchTable dt;
        Interpreter interpreter(dt);

        nb::Bench b;
        b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);

        auto test = [&](std::string const& name, EvaluatorBase&& evaluator) {
            evaluator.SetLocalOptimizationIterations(0);
            evaluator.SetBudget(std::numeric_limits<size_t>::max());
            b.batch(totalNodes * range.Size()).run(name, [&]() {
                return std::transform_reduce(individuals.begin(), individuals.end(), 0.0, std::plus<>{}, [&](auto& ind) { return evaluator(rd, ind); });
            });
        };
        
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

