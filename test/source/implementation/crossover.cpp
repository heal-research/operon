// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <random>
#include <vstat/vstat.hpp>
#include <fmt/core.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/core/types.hpp"

namespace Operon::Test {

TEST_CASE("Crossover")
{
    auto target = "Y";
    auto ds = Operon::Dataset("./data/Poly-10.csv", true);
    auto variables = ds.GetVariables();
    std::vector<Operon::Hash> inputs;
    for (auto const& v : variables) {
        if (v.Name != target) { inputs.push_back(v.Hash); }
    }

    auto const nrow { ds.Rows<std::size_t>() };
    Range range { 0, nrow };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc { grammar, inputs, /* bias= */ 0.0 };

    Operon::RandomGenerator random(1234);

    SUBCASE("Simple swap")
    {
        Operon::RandomGenerator rng(std::random_device{}());
        constexpr size_t maxDepth{1000};
        constexpr size_t maxLength{100};
        constexpr double internalNodeProbability{0.9};
        Operon::SubtreeCrossover cx(internalNodeProbability, maxDepth, maxLength);
        auto p1 = btc(rng, 7, 1, maxDepth); // NOLINT
        auto p2 = btc(rng, 5, 1, maxDepth); // NOLINT
        auto child = cx(rng, p1, p2);

        fmt::print("parent 1\n{}\n", TreeFormatter::Format(p1, ds, 2));
        fmt::print("parent 2\n{}\n", TreeFormatter::Format(p2, ds, 2));
        fmt::print("child\n{}\n", TreeFormatter::Format(child, ds, 2));
    }

    SUBCASE("Distribution of swap locations")
    {
        Operon::RandomGenerator rng(std::random_device{}());

        size_t maxDepth{1000}, maxLength{20};
        Operon::SubtreeCrossover cx(1.0, maxDepth, maxLength);

        std::vector<double> c1(maxLength);
        std::vector<double> c2(maxLength);

        uint64_t p1_term{0}, p1_func{0};
        uint64_t p2_term{0}, p2_func{0};

        for (int n = 0; n < 100000; ++n) {
            auto p1 = btc(rng, maxLength, 1, maxDepth);
            //auto p2 = btc(rng, maxLength, 1, maxDepth);
            auto p2 = p1;

            auto [i, j] = cx.FindCompatibleSwapLocations(rng, p1, p2);
            c1[i]++;
            c2[j]++;

            p1_term += p1[i].IsLeaf();
            p1_func += !p1[i].IsLeaf();

            p2_term += p2[i].IsLeaf();
            p2_func += !p2[i].IsLeaf();
        }

        fmt::print("p1_term: {}, p1_func: {}\n", p1_term, p1_func);
        fmt::print("p2_term: {}, p2_func: {}\n", p2_term, p2_func);

        fmt::print("parents swap location sampling counts:\n");
        for (size_t i = 0; i < maxLength; ++i) {
            fmt::print("{} {} {}\n", i, c1[i], c2[i]);
        }
    }

    SUBCASE("Child size") {
        const int n = 100000;
        std::vector<Tree> trees;
        constexpr size_t maxDepth{1000};
        constexpr size_t maxLength{100};
        auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
        for (int i = 0; i < n; ++i) {
            trees.push_back(btc(random, sizeDistribution(random), 1UL, maxDepth));
        }
        std::vector<std::array<size_t, 3>> sizes;

        std::uniform_int_distribution<size_t> dist(0, n-1);
        for (auto p : { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }) {
            Operon::SubtreeCrossover cx(p, maxDepth, maxLength);
            for (int i = 0; i < n; ++i) {
                auto p1 = dist(random);
                auto p2 = dist(random);
                auto c = cx(random, trees[p1], trees[p1]);
                sizes.push_back({trees[p1].Length(), trees[p2].Length(), c.Length()});
            }
            double m1 = vstat::univariate::accumulate<double>(sizes.begin(), sizes.end(), [](auto const& arr) { return arr[0]; }).mean;
            double m2 = vstat::univariate::accumulate<double>(sizes.begin(), sizes.end(), [](auto const& arr) { return arr[1]; }).mean;
            double m3 = vstat::univariate::accumulate<double>(sizes.begin(), sizes.end(), [](auto const& arr) { return arr[2]; }).mean;
            fmt::print("p: {:.1f}, parent1: {:.2f}, parent2: {:.2f}, child: {:.2f}\n", p, m1, m2, m3);
        }
    }
}
}
