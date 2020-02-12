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

#include <catch2/catch.hpp>
#include <execution>

#include "core/common.hpp"
#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/grammar.hpp"

#include "operators/creator.hpp"

#include <tbb/task_scheduler_init.h>

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

    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Sextic GPops", "[performance]")
    {
        Operon::Random random(1234);
        auto ds = Dataset("../data/Sextic.csv", true);
        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        size_t n = 10'000;
        std::vector<size_t> numRows { 1000, 5000 };
        std::vector<size_t> avgLen { 50, 100 };
        std::vector<Operon::Scalar> results;

        size_t maxDepth = 10000;
        Grammar grammar;

        for (auto len : avgLen) {
            // generate trees of a fixed length 
            std::uniform_int_distribution<size_t> sizeDistribution(len, len);
            auto creator = BalancedTreeCreator { sizeDistribution, maxDepth, len };
            std::vector<Tree> trees(n);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(random, grammar, inputs); });

            auto totalNodes = TotalNodes(trees);

            // test different number of rows
            for (auto nRows : numRows) {
                Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> chronometer;
                Range range { 0, nRows };
                auto totalOps = totalNodes * range.Size();

                MeanVarianceCalculator calc;
                BENCHMARK("Parallel")
                {
                    chronometer.start();
                    std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<float>(tree, ds, range).size(); });
                    chronometer.finish();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(chronometer.elapsed()).count() / 1000.0; // ms to s
                    auto gpops = totalOps / elapsed;
                    calc.Add(gpops);
                };
                fmt::print("\nfloat,{},{},{:.3e} ± {:.3e}\n", len, nRows, calc.Mean(), calc.StandardDeviation());

                calc.Reset();
                BENCHMARK("Parallel")
                {
                    chronometer.start();
                    std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<double>(tree, ds, range).size(); });
                    chronometer.finish();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(chronometer.elapsed()).count() / 1000.0; // ms to s
                    auto gpops = totalOps / elapsed;
                    calc.Add(gpops);
                };
                fmt::print("\ndouble,{},{},{:.3e} ± {:.3e}\n", len, nRows, calc.Mean(), calc.StandardDeviation());
            }
        }
    }

    TEST_CASE("Evaluation performance", "[performance]")
    {
        size_t n = 1'000;
        size_t maxLength = 50;
        size_t maxDepth = 1000;

        auto rd = Operon::Random();
        auto ds = Dataset("../data/Friedman-I.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        //Range range = { 0, ds.Rows() };
        Range range = { 0, 5000 };

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };

        std::vector<Tree> trees(n);
        std::vector<Operon::Scalar> fit(n);

        auto evaluate = [&](auto& tree) -> size_t {
            auto estimated = Evaluate<Operon::Scalar>(tree, ds, range);
            return estimated.size();
        };

        Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> chronometer;

        Grammar grammar;
        MeanVarianceCalculator calc;

        auto measurePerformance = [&]()
        {
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });
            auto totalNodes = TotalNodes(trees);
            fmt::print("total nodes: {}\n", totalNodes);
            auto totalOps = totalNodes * range.Size();
            // [+, -, *, /]
            BENCHMARK("Sequential")
            {
                chronometer.start();
                std::transform(std::execution::seq, trees.begin(), trees.end(), fit.begin(), evaluate);
                chronometer.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(chronometer.elapsed()).count() / 1000.0;
                auto gpops = totalOps / elapsed;
                calc.Add(gpops);
            };
            fmt::print("\nGPops/second: {:.3e} ± {:.3e}\n", calc.Mean(), calc.StandardDeviation());

            auto singlePerf = calc.Mean();

            calc.Reset();
            BENCHMARK("Parallel")
            {
                chronometer.start();
                std::transform(std::execution::par_unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
                chronometer.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(chronometer.elapsed()).count() / 1000.0;
                auto gpops = totalOps / elapsed;
                calc.Add(gpops);
            };
            fmt::print("\nGPops/second: {:.3e} ± {:.3e} (MP ratio {:.2f})\n", calc.Mean(), calc.StandardDeviation(), calc.Mean() / singlePerf);
        };

        SECTION("Arithmetic")
        {
            grammar.SetConfig(Grammar::Arithmetic);
            measurePerformance();
        }

        SECTION("Arithmetic + Exp + Log")
        {
            // [+, -, *, /, exp, log]
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log);
            measurePerformance();
        }

        SECTION("Arithmetic + Sin + Cos")
        {
            // [+, -, *, /, exp, log]
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Sin | NodeType::Cos);
            measurePerformance();
        }

        SECTION("Arithmetic + Exp + Log + Sin + Cos")
        {
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos);
            measurePerformance();
        }

        SECTION("Arithmetic + Sqrt + Cbrt + Square")
        {
            // [+, -, *, /, exp, log]
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Sqrt | NodeType::Cbrt | NodeType::Square);
            measurePerformance();
        }

        SECTION("Arithmetic + Exp + Log + Sin + Cos + Tan + Sqrt + Cbrt + Square")
        {
            // [+, -, *, /, exp, log]
            grammar.SetConfig(Grammar::Full);
            measurePerformance();
        }
    }
} // namespace Test
} // namespace Operon

