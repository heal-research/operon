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

namespace Operon {
namespace Test {
    TEST_CASE("Tree creation performance")
    {
        size_t n = 5000;
        size_t maxLength = 100;
        size_t maxDepth = 100;

        thread_local Operon::Random rd; 
        auto ds = Dataset("../data/Poly-10.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

        std::vector<Tree> trees(n);

        Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

        Grammar grammar;
        grammar.SetConfig(Grammar::Arithmetic);
        auto btc = BalancedTreeCreator { grammar, inputs };
        MeanVarianceCalculator calc;
        SECTION("Balanced tree creator")
        {
            calc.Reset();
            BENCHMARK("Sequential")
            {
                model.start();
                std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());

            calc.Reset();
            BENCHMARK("Parallel")
            {
                model.start();
                std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());
        }

        auto utc = UniformTreeCreator{ grammar, inputs };
        SECTION("Uniform tree creator")
        {
            calc.Reset();
            BENCHMARK("Sequential")
            {
                model.start();
                std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return utc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());

            calc.Reset();
            BENCHMARK("Parallel")
            {
                model.start();
                std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return utc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());
        }

        auto ptc = ProbabilisticTreeCreator{ grammar, inputs };
        SECTION("Probabilistic tree creator")
        {
            calc.Reset();
            BENCHMARK("Sequential")
            {
                model.start();
                std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return ptc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());

            calc.Reset();
            BENCHMARK("Parallel")
            {
                model.start();
                std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return ptc(rd, sizeDistribution(rd), maxDepth); });
                model.finish();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(model.elapsed()).count() / 1e6; // ms to s
                calc.Add(trees.size() / elapsed);
            };
            fmt::print("\nTrees/second: {:.1f} ± {:.1f}\n", calc.Mean(), calc.StandardDeviation());
        }
    }

} // namespace Test
} // namespace Operon

