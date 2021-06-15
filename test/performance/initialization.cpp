// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>
#include "core/dataset.hpp"
#include "interpreter/interpreter.hpp"
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
            b.batch(n).run("BTC", [&]() { std::generate(trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), 0, maxDepth); }); });
            b.batch(n).run("PTC", [&]() { std::generate(trees.begin(), trees.end(), [&]() { return ptc(rd, sizeDistribution(rd), 0, maxDepth); }); });
        }
        SUBCASE("BTC")
        {
            for (size_t i = 1; i <= maxLength; ++i) {
                b.complexityN(i).run("BTC", [&]() { std::generate(trees.begin(), trees.end(), [&]() { return btc(rd, i, 0, maxDepth); }); });
            }
            std::cout << "BTC complexity: " << b.complexityBigO() << std::endl;
        }

        SUBCASE("PTC")
        {
            for (size_t i = 1; i <= maxLength; ++i) {
                b.complexityN(i).run("PTC", [&]() { std::generate(trees.begin(), trees.end(), [&]() { return ptc(rd, i, 0, maxDepth); }); });
            }
            std::cout << "PTC complexity: " << b.complexityBigO() << std::endl;
        }
    }
} // namespace Test
} // namespace Operon
