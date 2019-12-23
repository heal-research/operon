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
            auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus{}, [](auto& tree) { return tree.Length(); });

            // test different number of rows
            for (auto nRows : numRows) {
                Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;
                Range range { 0, nRows };
                auto totalOps = totalNodes * range.Size() * 1000.0;

                MeanVarianceCalculator calc;
                BENCHMARK("Parallel")
                {
                    model.start();
                    std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<float>(tree, ds, range).size(); });
                    model.finish();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
                    auto gpops = totalOps / elapsed;
                    calc.Add(gpops);
                };
                fmt::print("\nfloat,{},{},{:.3e} ± {:.3e}\n", len, nRows, calc.Mean(), calc.StandardDeviation());

                calc.Reset();
                BENCHMARK("Parallel")
                {
                    model.start();
                    std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<double>(tree, ds, range).size(); });
                    model.finish();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
                    auto gpops = totalOps / elapsed;
                    calc.Add(gpops);
                };
                fmt::print("\ndouble,{},{},{:.3e} ± {:.3e}\n", len, nRows, calc.Mean(), calc.StandardDeviation());
            }
        }
    }

}
}
