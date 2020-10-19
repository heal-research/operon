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
#include "core/pset.hpp"

#include "nanobench.h"
#include "operators/creator.hpp"

namespace Operon {
namespace Test {
    TEST_CASE("Tree creation performance")
    {
        size_t n = 5000;
        size_t minLength = 1;
        size_t maxLength = 100;
        size_t maxDepth = 1000;

        Operon::RandomGenerator rd(std::random_device {}());
        auto ds = Dataset("../data/Poly-10.csv", true);

        auto target = "Y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        std::uniform_int_distribution<size_t> sizeDistribution(minLength, maxLength);

        std::vector<Tree> trees(n);

        PrimitiveSet grammar;
        grammar.SetConfig(PrimitiveSet::Arithmetic);

        auto btc = BalancedTreeCreator { grammar, inputs };
        auto gtc = GrowTreeCreator { grammar, inputs };
        auto ptc = ProbabilisticTreeCreator { grammar, inputs };

        ankerl::nanobench::Bench b;
        b.performanceCounters(true);

        SUBCASE("BTC vs PTC")
        {
            b.batch(n).run("BTC", [&]() { std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), 0, maxDepth); }); });
            b.batch(n).run("PTC", [&]() { std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return ptc(rd, sizeDistribution(rd), 0, maxDepth); }); });
            //b.minEpochIterations(1000).batch(n).run("BTC (parallel)", [&]() { std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), maxDepth); }); });
            //b.minEpochIterations(1000).batch(n).run("PTC (parallel)", [&]() { std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return ptc(rd, sizeDistribution(rd), maxDepth); }); });
        }
        SUBCASE("BTC")
        {
            for (size_t i = 1; i <= maxLength; ++i) {
                b.complexityN(i).run("BTC", [&]() { std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return btc(rd, i, 0, maxDepth); }); });
            }
            std::cout << "BTC complexity: " << b.complexityBigO() << std::endl;
        }

        SUBCASE("PTC")
        {
            for (size_t i = 1; i <= maxLength; ++i) {
                b.complexityN(i).run("PTC", [&]() { std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return ptc(rd, i, 0, maxDepth); }); });
            }
            std::cout << "PTC complexity: " << b.complexityBigO() << std::endl;
        }
    }
} // namespace Test
} // namespace Operon
