// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vstat/vstat.hpp>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/core/types.hpp"

namespace Operon::Test {

TEST_CASE("Crossover produces valid trees", "[operators]")
{
    auto ds = Operon::Dataset("./data/Poly-10.csv", true);
    auto variables = ds.GetVariables();
    std::vector<Operon::Hash> inputs;
    for (auto const& v : variables) {
        if (v.Name != "Y") { inputs.push_back(v.Hash); }
    }

    constexpr size_t maxDepth{1000};
    constexpr size_t maxLength{100};

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);
    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0, maxLength};

    Operon::RandomGenerator rng(1234);

    SECTION("Child is a valid tree") {
        constexpr double internalNodeProbability{0.9};
        Operon::SubtreeCrossover cx(internalNodeProbability, maxDepth, maxLength);
        auto p1 = btc(rng, 7, 1, maxDepth); // NOLINT
        auto p2 = btc(rng, 5, 1, maxDepth); // NOLINT
        auto child = cx(rng, p1, p2);

        CHECK(child.Length() > 0);
    }

    SECTION("Child size is within bounds") {
        auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
        constexpr int n = 10000;
        std::vector<Tree> trees;
        trees.reserve(n);
        for (int i = 0; i < n; ++i) {
            trees.push_back(btc(rng, sizeDistribution(rng), 1UL, maxDepth));
        }

        std::uniform_int_distribution<size_t> dist(0, n - 1);
        Operon::SubtreeCrossover cx(0.9, maxDepth, maxLength);

        for (int i = 0; i < 1000; ++i) {
            auto p1 = dist(rng);
            auto p2 = dist(rng);
            auto child = cx(rng, trees[p1], trees[p2]);
            CHECK(child.Length() <= maxLength);
        }
    }
}

} // namespace Operon::Test
