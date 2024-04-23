// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <unordered_set>
#include <vstat/vstat.hpp>
#include <fmt/core.h>

#include "operon/core/tree.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/distance.hpp"
#include "operon/core/operator.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/hash/hash.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"


namespace Operon::Test {

void CalculateDistance(std::vector<Tree>& trees, const std::string& name) {
    std::vector<Operon::Vector<Operon::Hash>> treeHashes;
    treeHashes.reserve(trees.size());

    for (auto& t : trees) {
        Operon::Vector<Operon::Hash> hh(t.Length());
        (void) t.Hash(Operon::HashMode::Strict);
        std::transform(t.Nodes().begin(), t.Nodes().end(), hh.begin(), [](auto& n) { return n.CalculatedHashValue; });
        std::sort(hh.begin(), hh.end());
        treeHashes.push_back(hh);
    }

    vstat::univariate_accumulator<double> acc;

    for (size_t i = 0; i < treeHashes.size() - 1; ++i) {
        for (size_t j = i + 1; j < treeHashes.size(); ++j) {
            acc( Operon::Distance::Jaccard(treeHashes[i], treeHashes[j]) );
        }
    }
    vstat::univariate_statistics stats(acc);
    fmt::print("Average distance ({}): {}\n", name, stats.mean);
}

void CalculateDistanceWithSort(std::vector<Tree>& trees, const std::string& name) {
    std::vector<Operon::Vector<Operon::Hash>> treeHashes;
    treeHashes.reserve(trees.size());

    for (auto& t : trees) {
        Operon::Vector<Operon::Hash> hh(t.Length());
        t.Sort();
        std::transform(t.Nodes().begin(), t.Nodes().end(), hh.begin(), [](auto& n) { return n.CalculatedHashValue; });
        std::sort(hh.begin(), hh.end());
        treeHashes.push_back(hh);
    }

    vstat::univariate_accumulator<double> acc;

    for (size_t i = 0; i < treeHashes.size() - 1; ++i) {
        for (size_t j = i + 1; j < treeHashes.size(); ++j) {
            acc( Operon::Distance::Jaccard(treeHashes[i], treeHashes[j]) );
        }
    }

    vstat::univariate_statistics stats(acc);
    fmt::print("Average distance (sort) ({}): {}\n", name, stats.mean);
}

TEST_CASE("Hash-based distance") {
    size_t n = 5000;

    size_t maxLength = 100;
    size_t minDepth  = 1;
    size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);

    auto target = "Y";
    auto variables = ds.GetVariables();
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash); // remove target

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    std::vector<size_t> indices(n);
    std::vector<Operon::Hash> seeds(n);
    std::vector<Tree> trees(n);
    std::vector<Operon::Vector<Operon::Hash>> treeHashes(n);

    auto btc = BalancedTreeCreator { grammar, inputs };
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;

    std::iota(indices.begin(), indices.end(), 0);
    std::generate(seeds.begin(), seeds.end(), [&](){ return rd(); });
    std::transform(indices.begin(), indices.end(), trees.begin(), [&](auto i) {
        Operon::RandomGenerator rand(seeds[i]);
        auto tree = btc(rand, sizeDistribution(rand), minDepth, maxDepth);
        initializer(rand, tree);
        return tree;
    });

    std::vector<std::pair<Operon::HashFunction, std::string>> hashFunctions {
        { Operon::HashFunction::XXHash,    "XXHash" },
        { Operon::HashFunction::MetroHash, "MetroHash" },
        { Operon::HashFunction::FNV1Hash,  "FNV1Hash" },
    };

    for (const auto& [f, name] : hashFunctions) {
        CalculateDistance(trees, name);
    }
}

TEST_CASE("Hash collisions") {
    size_t n = 100000;
    size_t maxLength = 200;
    size_t minDepth = 0;
    size_t maxDepth = 100;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.GetVariables();
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash);

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    std::vector<size_t> indices(n);
    std::vector<Operon::Hash> seeds(n);
    std::vector<Tree> trees(n);

    auto btc = BalancedTreeCreator { grammar, inputs };
    Operon::CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>> initializer;
    initializer.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    std::iota(indices.begin(), indices.end(), 0);
    std::generate(seeds.begin(), seeds.end(), [&](){ return rd(); });
    std::transform(indices.begin(), indices.end(), trees.begin(), [&](auto i) {
        Operon::RandomGenerator rand(seeds[i]);
        auto tree = btc(rand, sizeDistribution(rand), minDepth, maxDepth);
        initializer(rand, tree);
        return tree.Hash(Operon::HashMode::Strict);
    });

    std::unordered_set<uint64_t> set64;
    std::unordered_set<uint32_t> set32;
    auto totalNodes = std::transform_reduce(trees.begin(), trees.end(), size_t { 0 }, std::plus<size_t> {}, [](auto& tree) { return tree.Length(); });

    for(auto& tree : trees) {
        for(auto& node : tree.Nodes()) {
            auto h = node.CalculatedHashValue;
            set64.insert(h);
            set32.insert(static_cast<uint32_t>(h & 0xFFFFFFFFLL)); // NOLINT
        }
        tree.Nodes().clear();
    }
    auto s64 = static_cast<double>(set64.size());
    auto s32 = static_cast<double>(set32.size());
    fmt::print("total nodes: {}, {:.3f}% unique, unique 64-bit hashes: {}, unique 32-bit hashes: {}, collision rate: {:.3f}%\n", totalNodes, s64/static_cast<double>(totalNodes) * 100, s64, s32, (1 - s32/s64) * 100);
}
} // namespace Operon::Test
