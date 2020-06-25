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

#include "core/common.hpp"
#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/grammar.hpp"

#include "operators/creator.hpp"
#include "nanobench.h"

#include <tbb/global_control.h>

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

    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Evaluation performance")
    {
        size_t n = 1000;
        size_t maxLength = 100;
        size_t maxDepth = 1000;

        Operon::Random rd(std::random_device{}()); 
        auto ds = Dataset("../data/Friedman-I.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        //Range range = { 0, ds.Rows() };
        Range range = { 0, 5000 };

        Grammar grammar;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { grammar, inputs };

        std::vector<Tree> trees(n);
        std::vector<Operon::Scalar> fit(n);

        auto evaluate = [&](auto& tree) {
            auto estimated = Evaluate<Operon::Scalar>(tree, ds, range);
            nb::doNotOptimizeAway(estimated.size());
        };

        nb::Bench b;
        b.title("evaluation | grammar").relative(true).performanceCounters(false).minEpochIterations(10);

        auto test = [&](GrammarConfig cfg, ExecutionPolicy pol, const std::string& name) {  
            grammar.SetConfig(cfg);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });
            auto totalOps = TotalNodes(trees) * range.Size();

            b.batch(totalOps);

            switch(pol) {
                case ExecutionPolicy::Sequenced:
                    b.run(name, [&]() { std::for_each(std::execution::seq, trees.begin(), trees.end(), evaluate); });
                    break;
                case ExecutionPolicy::Unsequenced:
                    b.run(name, [&]() { std::for_each(std::execution::unseq, trees.begin(), trees.end(), evaluate); });
                    break;
                case ExecutionPolicy::ParallelSequenced:
                    b.run(name, [&]() { std::for_each(std::execution::par, trees.begin(), trees.end(), evaluate); });
                    break;
                case ExecutionPolicy::ParallelUnsequenced:
                    b.run(name, [&]() { std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), evaluate); });
                    break;
            }

        };

        // single-thread
        test(Grammar::Arithmetic, ExecutionPolicy::Unsequenced, "unseq : arithmetic");
        test(Grammar::Arithmetic | NodeType::Exp, ExecutionPolicy::Unsequenced, "unseq : arithmetic + exp");
        test(Grammar::Arithmetic | NodeType::Log, ExecutionPolicy::Unsequenced, "unseq : arithmetic + log");
        test(Grammar::Arithmetic | NodeType::Sin, ExecutionPolicy::Unsequenced, "unseq : arithmetic + sin");
        test(Grammar::Arithmetic | NodeType::Cos, ExecutionPolicy::Unsequenced, "unseq : arithmetic + cos");
        test(Grammar::Arithmetic | NodeType::Tan, ExecutionPolicy::Unsequenced, "unseq : arithmetic + tan");
        test(Grammar::Arithmetic | NodeType::Sqrt, ExecutionPolicy::Unsequenced, "unseq : arithmetic + sqrt");
        test(Grammar::Arithmetic | NodeType::Cbrt, ExecutionPolicy::Unsequenced, "unseq : arithmetic + cbrt");
        test(Grammar::Arithmetic | NodeType::Exp | NodeType::Log, ExecutionPolicy::Unsequenced, "unseq : arithmetic + exp + log");
        test(Grammar::Arithmetic | NodeType::Sin | NodeType::Cos, ExecutionPolicy::Unsequenced, "unseq : arithmetic + sin + cos");

        //// multi-thread
        //test(Grammar::Arithmetic, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic");
        //test(Grammar::Arithmetic | NodeType::Exp, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + exp");
        //test(Grammar::Arithmetic | NodeType::Log, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + log");
        //test(Grammar::Arithmetic | NodeType::Sin, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + sin");
        //test(Grammar::Arithmetic | NodeType::Cos, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + cos");
        //test(Grammar::Arithmetic | NodeType::Tan, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + tan");
        //test(Grammar::Arithmetic | NodeType::Sqrt, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + sqrt");
        //test(Grammar::Arithmetic | NodeType::Cbrt, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + cbrt");
        //test(Grammar::Arithmetic | NodeType::Exp | NodeType::Log, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + exp + log");
        //test(Grammar::Arithmetic | NodeType::Sin | NodeType::Cos, ExecutionPolicy::ParallelUnsequenced, "par_unseq : arithmetic + sin + cos");
    }
} // namespace Test
} // namespace Operon

