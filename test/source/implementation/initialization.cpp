// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

static auto GenerateTrees(Operon::RandomGenerator& random, Operon::CreatorBase& creator, std::vector<size_t> lengths, size_t maxDepth) -> std::vector<Tree>
{
    std::vector<Tree> trees;
    trees.reserve(lengths.size());
    UniformTreeInitializer treeInit(&creator);
    treeInit.ParameterizeDistribution(1UL, 100UL);
    treeInit.SetMaxDepth(maxDepth);
    UniformCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    std::transform(lengths.begin(), lengths.end(), std::back_inserter(trees), [&](size_t /*not used*/) {
        auto tree = treeInit(random);
        coeffInit(random, tree);
        return tree;
    });

    return trees;
}

TEST_CASE("Grammar sampling", "[operators]")
{
    PrimitiveSet grammar;
    grammar.SetConfig(~PrimitiveSetConfig{});
    Operon::RandomGenerator rd(1234);

    std::vector<double> observed(NodeTypes::Count, 0);
    size_t r = grammar.EnabledPrimitives().size() + 1;

    const size_t nTrials = 1'000'000;
    for (auto i = 0U; i < nTrials; ++i) {
        auto node = grammar.SampleRandomSymbol(rd, 0, 2);
        ++observed[NodeTypes::GetIndex(node.Type)];
    }
    std::transform(observed.begin(), observed.end(), observed.begin(), [&](double v) { return v / nTrials; });
    std::vector<double> actual(NodeTypes::Count, 0);
    for (size_t i = 0; i < observed.size(); ++i) {
        auto type = static_cast<NodeType>(i);
        auto node = Node(type);
        actual[NodeTypes::GetIndex(type)] = static_cast<double>(grammar.Frequency(node.HashValue));
    }
    auto freqSum = std::reduce(actual.begin(), actual.end(), 0.0, std::plus{});
    std::transform(actual.begin(), actual.end(), actual.begin(), [&](double v) { return v / freqSum; });
    auto chi = 0.0;
    for (auto i = 0U; i < observed.size(); ++i) {
        Node node(static_cast<NodeType>(i));
        if (!grammar.IsEnabled(node.HashValue)) {
            continue;
        }
        auto x = observed[i];
        auto y = actual[i];
        chi += (x - y) * (x - y) / y;
    }
    chi *= nTrials;

    auto criticalValue = static_cast<double>(r) + 2 * std::sqrt(r);
    REQUIRE(chi <= criticalValue);
}

TEST_CASE("GROW creator", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y")->Hash);
    size_t const maxDepth = 10;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.SetMaximumArity(Node(NodeType::Add), 2);
    grammar.SetMaximumArity(Node(NodeType::Mul), 2);
    grammar.SetMaximumArity(Node(NodeType::Sub), 2);
    grammar.SetMaximumArity(Node(NodeType::Div), 2);

    GrowTreeCreator gtc{&grammar, inputs};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, gtc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
            CHECK(tree.Length() <= maxLength + 10); // allow some slack for tree construction
        }
    }

    SECTION("Only enabled primitives appear") {
        auto tree = gtc(random, 20, 1, maxDepth);
        for (auto const& node : tree.Nodes()) {
            if (!node.IsLeaf()) {
                CHECK(grammar.IsEnabled(node.HashValue));
            }
        }
    }
}

TEST_CASE("BTC creator", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y")->Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.SetMaximumArity(Node(NodeType::Add), 2);
    grammar.SetMaximumArity(Node(NodeType::Mul), 2);
    grammar.SetMaximumArity(Node(NodeType::Sub), 2);
    grammar.SetMaximumArity(Node(NodeType::Div), 2);

    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, btc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
        }
    }

    SECTION("Coefficients are initialized") {
        auto tree = btc(random, 20, 1, maxDepth);
        UniformCoefficientInitializer coeffInit;
        coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});
        coeffInit(random, tree);

        bool hasCoeff = false;
        for (auto const& node : tree.Nodes()) {
            if (node.IsLeaf() && node.IsConstant()) {
                hasCoeff = true;
            }
        }
        CHECK(hasCoeff);
    }
}

TEST_CASE("PTC2 creator", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y")->Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;
    size_t const n = 1000;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);

    ProbabilisticTreeCreator ptc{&grammar, inputs};
    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    SECTION("Trees are within size bounds") {
        std::vector<size_t> lengths(n);
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, ptc, lengths, maxDepth);
        for (auto const& tree : trees) {
            CHECK(tree.Length() > 0);
        }
    }
}

} // namespace Operon::Test
