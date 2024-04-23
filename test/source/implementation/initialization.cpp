// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <doctest/doctest.h>
#include <fmt/core.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {

TEST_CASE("Sample nodes from grammar")
{
    PrimitiveSet grammar;
    grammar.SetConfig(static_cast<NodeType>(~uint32_t{0}));
    Operon::RandomGenerator rd(std::random_device {}());

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
        auto type = static_cast<NodeType>(1U << i);
        auto node = Node(type);
        actual[NodeTypes::GetIndex(type)] = static_cast<double>(grammar.Frequency(node.HashValue));
    }
    auto freqSum = std::reduce(actual.begin(), actual.end(), 0.0, std::plus {});
    std::transform(actual.begin(), actual.end(), actual.begin(), [&](double v) { return v / freqSum; });
    auto chi = 0.0;
    for (auto i = 0U; i < observed.size(); ++i) {
        Node node(static_cast<NodeType>(1U << i));
        if (!grammar.IsEnabled(node.HashValue)) {
            continue;
        }
        auto x = observed[i];
        auto y = actual[i];
        fmt::print("{:>8} observed {:.4f}, expected {:.4f}\n", node.Name(), x, y);
        chi += (x - y) * (x - y) / y;
    }
    chi *= nTrials;

    auto criticalValue = static_cast<double>(r) + 2 * std::sqrt(r);
    fmt::print("chi = {}, critical value = {}\n", chi, criticalValue);
    REQUIRE(chi <= criticalValue);
}

auto GenerateTrees(Operon::RandomGenerator& random, Operon::CreatorBase& creator, std::vector<size_t> lengths, size_t maxDepth) -> std::vector<Tree>
{
    std::vector<Tree> trees;
    trees.reserve(lengths.size());
    UniformTreeInitializer treeInit(creator);
    treeInit.ParameterizeDistribution(1UL, 100UL);
    treeInit.SetMaxDepth(maxDepth);
    UniformCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{-1}, Operon::Scalar{+1});

    std::transform(lengths.begin(), lengths.end(), std::back_inserter(trees), [&](size_t /*not used*/) {
        //auto tree = creator(random, len, 0, maxDepth);
        auto tree = treeInit(random);
        coeffInit(random, tree);
        return tree;
    });

    return trees;
}

auto CalculateSymbolFrequencies(const std::vector<Tree>& trees) -> std::array<size_t, NodeTypes::Count>
{
    std::array<size_t, NodeTypes::Count> symbolFrequencies{};
    symbolFrequencies.fill(0U);

    for (const auto& tree : trees) {
        for (const auto& node : tree.Nodes()) {
            symbolFrequencies[NodeTypes::GetIndex(node.Type)]++; // NOLINT
        }
    }

    return symbolFrequencies;
}

auto CalculateHistogram(const std::vector<size_t>& values) -> std::vector<size_t>
{
    auto [min, max] = std::minmax_element(values.begin(), values.end());

    std::vector<size_t> counts(*max + 1, 0UL);

    for (auto v : values) {
        counts[v]++;
    }

    return counts;
}

TEST_CASE("GROW")
{
    auto const* target = "Y";
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto variables = ds.GetVariables();
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash);
    size_t const maxDepth = 10;
    size_t const maxLength = 100;

    size_t const n = 10000;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.SetMaximumArity(Node(NodeType::Add), 2);
    grammar.SetMaximumArity(Node(NodeType::Mul), 2);
    grammar.SetMaximumArity(Node(NodeType::Sub), 2);
    grammar.SetMaximumArity(Node(NodeType::Div), 2);

    grammar.SetFrequency(Node(NodeType::Add), 4);
    grammar.SetFrequency(Node(NodeType::Mul), 1);
    grammar.SetFrequency(Node(NodeType::Sub), 1);
    grammar.SetFrequency(Node(NodeType::Div), 1);
    grammar.SetFrequency(Node(NodeType::Exp), 1);
    grammar.SetFrequency(Node(NodeType::Log), 1);

    GrowTreeCreator gtc{grammar, inputs};
    Operon::RandomGenerator random(std::random_device{}());

    std::vector<size_t> lengths(n);

    SUBCASE("Simple tree")
    {
        size_t const targetLength = 20;
        auto tree = gtc(random, targetLength, 1, maxDepth);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    SUBCASE("Symbol frequencies")
    {
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, gtc, lengths, maxDepth);
        auto totalLength = std::transform_reduce(trees.begin(), trees.end(), size_t { 0 }, std::plus<size_t> {}, [](auto& tree) { return tree.Length(); });
        fmt::print("Symbol frequencies:\n");
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1U << i));
            if (grammar.Contains(node) && grammar.IsEnabled(node)) {
                fmt::print("{}\t{:.3f} %\n", node.Name(), static_cast<double>(symbolFrequencies[i]) / static_cast<double>(totalLength));
            }
        }
    }

    SUBCASE("Length histogram")
    {
        const int reps = 50;
        std::vector<double> counts;

        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, gtc, lengths, maxDepth);
            std::vector<size_t> actualLengths(trees.size());
            std::transform(trees.begin(), trees.end(), actualLengths.begin(), [](const auto& t) { return t.Length(); });
            auto cnt = CalculateHistogram(actualLengths);
            if (cnt.size() > counts.size()) { counts.resize(cnt.size(), 0); }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts.at(j) += static_cast<double>(cnt[j]);
            }
        }

        fmt::print("Length histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }

    SUBCASE("Shape histogram")
    {
        const int reps = 50;
        std::vector<double> counts;

        double avgShape = 0.0;
        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, gtc, lengths, maxDepth);
            std::vector<size_t> shapes(trees.size());
            std::transform(trees.begin(), trees.end(), shapes.begin(), [](const auto& t) { return std::transform_reduce(t.Nodes().begin(), t.Nodes().end(), 0UL, std::plus<size_t> {}, [](const auto& node) { return node.Length + 1; }); });

            auto sum = std::reduce(shapes.begin(), shapes.end());
            avgShape += static_cast<double>(sum) / static_cast<double>(trees.size());
            auto cnt = CalculateHistogram(shapes);
            if (counts.size() < cnt.size()) {
                counts.resize(cnt.size());
            }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += static_cast<double>(cnt[j]);
            }
        }

        avgShape /= reps;
        fmt::print("Average shape: {}\n", avgShape);

        fmt::print("Shape histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }
}

TEST_CASE("BTC")
{
    const auto *target = "Y";
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto variables = ds.GetVariables();

    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;

    size_t const n = 10000;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.SetMaximumArity(Node(NodeType::Add), 2);
    grammar.SetMaximumArity(Node(NodeType::Mul), 2);
    grammar.SetMaximumArity(Node(NodeType::Sub), 2);
    grammar.SetMaximumArity(Node(NodeType::Div), 2);

    grammar.SetFrequency(Node(NodeType::Add), 4);
    grammar.SetFrequency(Node(NodeType::Mul), 1);
    grammar.SetFrequency(Node(NodeType::Sub), 1);
    grammar.SetFrequency(Node(NodeType::Div), 1);
    grammar.SetFrequency(Node(NodeType::Exp), 1);
    grammar.SetFrequency(Node(NodeType::Log), 1);

    BalancedTreeCreator btc { grammar, inputs, /* bias= */ 0.0 };

    Operon::RandomGenerator random(std::random_device {}());

    std::vector<size_t> lengths(n);

    SUBCASE("Simple tree")
    {
        auto tree = btc(random, 50, 1, maxDepth);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    SUBCASE("Symbol frequencies")
    {
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, btc, lengths, maxDepth);
        auto totalLength = std::transform_reduce(trees.begin(), trees.end(), size_t { 0 }, std::plus<size_t> {}, [](auto& tree) { return tree.Length(); });
        fmt::print("Symbol frequencies: \n");
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1U << i));
            if (!grammar.Contains(node) || !grammar.IsEnabled(node)) {
                continue;
            }
            fmt::print("{}\t{:.3f} %\n", node.Name(), static_cast<double>(symbolFrequencies[i]) / static_cast<double>(totalLength));
        }
    }

    SUBCASE("Length histogram")
    {
        const int reps = 50;
        std::vector<double> counts(maxLength + 1, 0);

        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, btc, lengths, maxDepth);
            std::vector<size_t> actualLengths(trees.size());
            std::transform(trees.begin(), trees.end(), actualLengths.begin(), [](const auto& t) { return t.Length(); });
            auto cnt = CalculateHistogram(actualLengths);
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += static_cast<double>(cnt[j]);
            }
        }

        fmt::print("Length histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }

    SUBCASE("Shape histogram")
    {
        const int reps = 50;
        std::vector<double> counts;

        double avgShape = 0.0;
        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, btc, lengths, maxDepth);
            std::vector<size_t> shapes(trees.size());
            std::transform(trees.begin(), trees.end(), shapes.begin(), [](const auto& t) { return std::transform_reduce(t.Nodes().begin(), t.Nodes().end(), 0UL, std::plus<size_t> {}, [](const auto& node) { return node.Length + 1; }); });

            auto sum = std::reduce(shapes.begin(), shapes.end());
            avgShape += static_cast<double>(sum) / static_cast<double>(trees.size());
            auto cnt = CalculateHistogram(shapes);
            if (counts.size() < cnt.size()) {
                counts.resize(cnt.size());
            }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += static_cast<double>(cnt[j]);
            }
        }

        avgShape /= reps;
        fmt::print("Average shape: {}\n", avgShape);

        fmt::print("Shape histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }
}

TEST_CASE("PTC2")
{
    const auto *target = "Y";
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto variables = ds.GetVariables();
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash);
    size_t const maxDepth = 1000;
    size_t const maxLength = 100;

    size_t const n = 10000;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);

    ProbabilisticTreeCreator ptc { grammar, inputs };

    Operon::RandomGenerator random(std::random_device {}());

    std::vector<size_t> lengths(n);

    SUBCASE("Simple tree")
    {
        long const targetLength = 10;
        auto tree = ptc(random, targetLength, 1, maxDepth);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    SUBCASE("Symbol frequencies")
    {
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, ptc, lengths, maxDepth);
        auto totalLength = std::transform_reduce(trees.begin(), trees.end(), size_t { 0 }, std::plus<size_t> {}, [](auto& tree) { return tree.Length(); });
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        fmt::print("Symbol frequencies:\n");
        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1U << i));
            if (!(grammar.Contains(node.HashValue) && grammar.IsEnabled(node.HashValue))) {
                continue;
            }
            fmt::print("{}\t{} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    SUBCASE("Length histogram")
    {
        const int reps = 50;
        std::vector<double> counts(maxLength + 1, 0);

        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, ptc, lengths, maxDepth);
            std::vector<size_t> actualLengths(trees.size());
            std::transform(trees.begin(), trees.end(), actualLengths.begin(), [](const auto& t) { return t.Length(); });
            auto cnt = CalculateHistogram(actualLengths);
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += static_cast<double>(cnt[j]);
            }
        }

        fmt::print("Length histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }

    SUBCASE("Shape histogram")
    {
        const int reps = 50;
        std::vector<double> counts;

        double avgShape = 0.0;
        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, ptc, lengths, maxDepth);
            std::vector<size_t> shapes(trees.size());
            std::transform(trees.begin(), trees.end(), shapes.begin(), [](const auto& t) { return std::transform_reduce(t.Nodes().begin(), t.Nodes().end(), 0UL, std::plus<size_t> {}, [](const auto& node) { return node.Length + 1; }); });
            auto sum = std::reduce(shapes.begin(), shapes.end());
            avgShape += static_cast<double>(sum) / static_cast<double>(trees.size());
            auto cnt = CalculateHistogram(shapes);
            if (counts.size() < cnt.size()) {
                counts.resize(cnt.size());
            }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += static_cast<double>(cnt[j]);
            }
        }

        avgShape /= reps;
        fmt::print("Average shape: {}\n", avgShape);

        fmt::print("Shape histogram: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            counts[i] /= reps;
            fmt::print("{}\t{}\n", i, counts[i]);
        }
    }
}
} // namespace Operon::Test
