// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <unordered_set>
#include <vstat/vstat.hpp>

#include "operon/core/tree.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/distance.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/hash/hash.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

TEST_CASE("Hash determinism", "[core]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    constexpr size_t maxLength = 20;
    Operon::RandomGenerator rd(42);
    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;

    auto tree1 = btc(rd, 20, 1, 1000);
    initializer(rd, tree1);

    // Copy the tree and hash both
    auto tree2 = tree1;
    auto h1 = tree1.Hash(Operon::HashMode::Strict);
    auto h2 = tree2.Hash(Operon::HashMode::Strict);

    // Same tree content should produce same hash
    auto& nodes1 = h1.Nodes();
    auto& nodes2 = h2.Nodes();
    REQUIRE(nodes1.size() == nodes2.size());
    for (size_t i = 0; i < nodes1.size(); ++i) {
        CHECK(nodes1[i].CalculatedHashValue == nodes2[i].CalculatedHashValue);
    }
}

TEST_CASE("Hash-based distance", "[core]")
{
    size_t n = 5000;
    size_t maxLength = 100;
    size_t minDepth = 1;
    size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    auto btc = BalancedTreeCreator{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;

    std::vector<Tree> trees(n);
    std::vector<Operon::Hash> seeds(n);
    std::generate(seeds.begin(), seeds.end(), [&]() -> Operon::RandomGenerator::result_type { return rd(); });
    for (size_t i = 0; i < n; ++i) {
        Operon::RandomGenerator rand(seeds[i]);
        trees[i] = btc(rand, sizeDistribution(rand), minDepth, maxDepth);
        initializer(rand, trees[i]);
    }

    // Compute pairwise Jaccard distances and verify they are in [0, 1]
    std::vector<Operon::Vector<Operon::Hash>> treeHashes;
    treeHashes.reserve(trees.size());

    for (auto& t : trees) {
        Operon::Vector<Operon::Hash> hh(t.Length());
        (void)t.Hash(Operon::HashMode::Strict);
        std::transform(t.Nodes().begin(), t.Nodes().end(), hh.begin(), [](auto& n) -> auto { return n.CalculatedHashValue; });
        std::sort(hh.begin(), hh.end());
        treeHashes.push_back(hh);
    }

    // Sample some pairs and check distances
    constexpr size_t samplePairs = 1000;
    std::uniform_int_distribution<size_t> indexDist(0, n - 1);
    for (size_t k = 0; k < samplePairs; ++k) {
        auto i = indexDist(rd);
        auto j = indexDist(rd);
        auto d = Operon::Distance::Jaccard(treeHashes[i], treeHashes[j]);
        CHECK(d >= 0.0);
        CHECK(d <= 1.0);
    }
}

TEST_CASE("Sorensen-Dice distance", "[core]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    constexpr size_t maxLength = 20;
    Operon::RandomGenerator rd(1234);
    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;

    auto tree1 = btc(rd, 20, 1, 1000);
    initializer(rd, tree1);
    auto tree2 = btc(rd, 20, 1, 1000);
    initializer(rd, tree2);

    (void)tree1.Hash(Operon::HashMode::Strict);
    (void)tree2.Hash(Operon::HashMode::Strict);

    Operon::Vector<Operon::Hash> h1(tree1.Length());
    Operon::Vector<Operon::Hash> h2(tree2.Length());
    std::transform(tree1.Nodes().begin(), tree1.Nodes().end(), h1.begin(), [](auto& n) -> auto { return n.CalculatedHashValue; });
    std::transform(tree2.Nodes().begin(), tree2.Nodes().end(), h2.begin(), [](auto& n) -> auto { return n.CalculatedHashValue; });
    std::sort(h1.begin(), h1.end());
    std::sort(h2.begin(), h2.end());

    auto sd = Operon::Distance::SorensenDice(h1, h2);
    CHECK(sd >= 0.0);
    CHECK(sd <= 1.0);

    // Self-distance should be 0
    auto selfDist = Operon::Distance::SorensenDice(h1, h1);
    CHECK(selfDist == Catch::Approx(0.0));
}

TEST_CASE("Hash collisions", "[core]")
{
    size_t n = 100000;
    size_t maxLength = 200;
    size_t minDepth = 0;
    size_t maxDepth = 100;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;
    initializer.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    std::vector<Operon::Hash> seeds(n);
    std::generate(seeds.begin(), seeds.end(), [&]() -> Operon::RandomGenerator::result_type { return rd(); });
    for (size_t i = 0; i < n; ++i) {
        Operon::RandomGenerator rand(seeds[i]);
        trees[i] = btc(rand, sizeDistribution(rand), minDepth, maxDepth);
        initializer(rand, trees[i]);
        std::ignore = trees[i].Hash(Operon::HashMode::Strict);
    }

    std::unordered_set<uint64_t> set64;
    auto totalNodes = std::transform_reduce(trees.begin(), trees.end(), size_t{0}, std::plus<size_t>{}, [](auto& tree) -> auto { return tree.Length(); });

    for (auto& tree : trees) {
        for (auto& node : tree.Nodes()) {
            set64.insert(node.CalculatedHashValue);
        }
        tree.Nodes().clear();
    }

    auto uniqueRatio = static_cast<double>(set64.size()) / static_cast<double>(totalNodes);
    // Strict hashing with random coefficients should yield near-zero collisions
    CHECK(uniqueRatio > 0.98);
}

TEST_CASE("Strict vs relaxed hashing modes", "[core]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    constexpr size_t maxLength = 20;
    Operon::RandomGenerator rd(42);
    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0, maxLength};
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;

    auto tree = btc(rd, 20, 1, 1000);
    initializer(rd, tree);

    auto treeStrict = tree;
    auto treeRelaxed = tree;
    std::ignore = treeStrict.Hash(Operon::HashMode::Strict);
    std::ignore = treeRelaxed.Hash(Operon::HashMode::Relaxed);

    // Both should produce valid hashes but they may differ
    CHECK(treeStrict.Nodes().back().CalculatedHashValue != 0);
    CHECK(treeRelaxed.Nodes().back().CalculatedHashValue != 0);
}

} // namespace Operon::Test
