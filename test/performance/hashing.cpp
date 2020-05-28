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

TEST_CASE("Tree hashing performance") {
    size_t n = 100000;
    size_t maxLength = 200;
    size_t maxDepth = 100;

    auto rd = Operon::Random();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { grammar, inputs };

    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;
    MeanVarianceCalculator calc;

    BENCHMARK("Tree sort/hash") {
        std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); });
        auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });

        model.start();
        std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [](auto& t) {t.Sort(Operon::HashMode::Strict); });
        model.finish();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count() / 1000.0;
        calc.Add(totalNodes / elapsed);
    };
    fmt::print("\nNodes/second: {:.3e} ± {:.3e}\n", calc.Mean(), calc.StandardDeviation());
}

} // namespace Test 
} // namespace Operon

