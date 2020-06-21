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
#include "analyzers/diversity.hpp"

#include "nanobench.h"

namespace Operon {
namespace Test {

TEST_CASE("Hashing performance") {
    size_t n = 1000;
    size_t maxLength = 100;
    size_t maxDepth = 1000;

    auto rd = Operon::Random();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    std::uniform_int_distribution<size_t> dist(0, n-1);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { grammar, inputs };

    ankerl::nanobench::Bench b;
    b.relative(true).performanceCounters(true);
    std::generate(std::execution::unseq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); });

    auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });

    SUBCASE("hashing performance") {
        b.batch(totalNodes).run("strict hashing", [&]() { 
            std::for_each(trees.begin(), trees.end(), [](auto t) { t.Sort(Operon::HashMode::Strict); });
        });

        b.batch(totalNodes).run("relaxed hashing", [&]() { 
            std::for_each(trees.begin(), trees.end(), [](auto t) { t.Sort(Operon::HashMode::Relaxed); });
        });
    }

    SUBCASE("strict hashing complexity") {
        ankerl::nanobench::Bench b;
        b.relative(true).performanceCounters(true);

        for (size_t i = 1; i <= maxLength; ++i) {
            std::generate(std::execution::unseq, trees.begin(), trees.end(), [&]() { return btc(rd, i, maxDepth); });
            totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
            b.complexityN(i).batch(totalNodes).run("strict", [&]() { 
                ankerl::nanobench::doNotOptimizeAway(std::for_each(trees.begin(), trees.end(), [](auto t) { t.Sort(Operon::HashMode::Strict); }));
            });
        }

        std::cout << b.complexityBigO() << "\n";
    }

    SUBCASE("relaxed hashing complexity") {
        ankerl::nanobench::Bench b;
        b.relative(true).performanceCounters(true);

        for (size_t i = 1; i <= maxLength; ++i) {
            std::generate(std::execution::unseq, trees.begin(), trees.end(), [&]() { return btc(rd, i, maxDepth); });
            totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
            b.complexityN(i).batch(totalNodes).run("relaxed", [&]() { 
                ankerl::nanobench::doNotOptimizeAway(std::for_each(trees.begin(), trees.end(), [](auto t) { t.Sort(Operon::HashMode::Relaxed); }));
            });
        }

        std::cout << b.complexityBigO() << "\n";
    }
}

} // namespace Test 
} // namespace Operon

