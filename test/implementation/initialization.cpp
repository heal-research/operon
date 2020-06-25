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

#include <algorithm>
#include <doctest/doctest.h>
#include <execution>

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"


namespace Operon::Test {
TEST_CASE("Sample nodes from grammar")
{
    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.Enable(NodeType::Add, 5);
    Operon::Random rd(std::random_device {}());

    std::vector<double> observed(NodeTypes::Count, 0);
    size_t r = grammar.EnabledSymbols().size() + 1;

    const size_t nTrials = 1'000'000;
    for (auto i = 0u; i < nTrials; ++i) {
        auto node = grammar.SampleRandomSymbol(rd, 0, 2);
        ++observed[NodeTypes::GetIndex(node.Type)];
    }
    std::transform(std::execution::unseq, observed.begin(), observed.end(), observed.begin(), [&](double v) { return v / nTrials; });
    std::vector<double> actual(NodeTypes::Count, 0);
    for (size_t i = 0; i < observed.size(); ++i) {
        auto nodeType = static_cast<NodeType>(1u << i);
        actual[NodeTypes::GetIndex(nodeType)] = grammar.GetFrequency(nodeType);
    }
    auto freqSum = std::reduce(std::execution::unseq, actual.begin(), actual.end(), 0.0, std::plus {});
    std::transform(std::execution::unseq, actual.begin(), actual.end(), actual.begin(), [&](double v) { return v / freqSum; });
    auto chi = 0.0;
    for (auto i = 0u; i < observed.size(); ++i) {
        auto nodeType = static_cast<NodeType>(1u << i);
        if (!grammar.IsEnabled(nodeType))
            continue;
        auto x = observed[i];
        auto y = actual[i];
        fmt::print("{:>8} observed {:.4f}, expected {:.4f}\n", Node(nodeType).Name(), x, y);
        chi += (x - y) * (x - y) / y;
    }
    chi *= nTrials;

    auto criticalValue = r + 2 * std::sqrt(r);
    fmt::print("chi = {}, critical value = {}\n", chi, criticalValue);
    REQUIRE(chi <= criticalValue);
}

std::vector<Tree> GenerateTrees(Random& random, CreatorBase& creator, std::vector<size_t> lengths, size_t maxDepth = 1000)
{
    std::vector<Tree> trees;
    trees.reserve(lengths.size());

    std::transform(lengths.begin(), lengths.end(), std::back_inserter(trees), [&](size_t len) { return creator(random, len, 0, maxDepth); }); 
    return trees;
}

std::array<size_t, NodeTypes::Count> CalculateSymbolFrequencies(const std::vector<Tree>& trees) 
{
    std::array<size_t, NodeTypes::Count> symbolFrequencies;
    symbolFrequencies.fill(0u);

    for (const auto& tree : trees) {
        for (const auto& node : tree.Nodes()) {
            symbolFrequencies[NodeTypes::GetIndex(node.Type)]++;
        }
    }

    return symbolFrequencies;

}

std::vector<size_t> CalculateHistogram(const std::vector<size_t>& values) 
{
    auto [min, max] = std::minmax_element(values.begin(), values.end());

    std::vector<size_t> counts(*max + 1, 0ul);

    for (auto v : values) {
        counts[v]++;
    }
    
    return counts;
}

TEST_CASE("GROW") 
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t minDepth = 1,
           maxDepth = 10;

    size_t n = 10000;

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.Enable(NodeType::Add, 1);
    grammar.Enable(NodeType::Mul, 1);
    grammar.Enable(NodeType::Sub, 1);
    grammar.Enable(NodeType::Div, 1);
    grammar.Enable(NodeType::Exp, 1);
    grammar.Enable(NodeType::Log, 1);

    GrowTreeCreator grow{ grammar, inputs };

    Operon::Random random(std::random_device {}());

    SUBCASE("Symbol frequencies")
    {
        std::vector<Tree> trees(n);
        std::generate(trees.begin(), trees.end(), [&]() { return grow(random, 0, minDepth, maxDepth); });
        auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
        fmt::print("Symbol frequencies: \n");
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1u << i));
            if (!grammar.IsEnabled(node.Type))
                continue;
            fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    SUBCASE("Simple tree") 
    {
        auto tree = grow(random, 0, minDepth, maxDepth);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    SUBCASE("Length vs depth") 
    {
        int reps = 50;
        std::vector<size_t> counts(maxDepth+1, 0);
        std::vector<double> lengths(maxDepth+1, 0); 

        for (int i = 0; i < reps; ++i) {
            std::vector<Tree> trees(n);
            std::generate(trees.begin(), trees.end(), [&]() { return grow(random, 0, minDepth, maxDepth); });

            for(const auto& tree : trees) {
                counts[tree.Depth()] += 1;
                lengths[tree.Depth()] += tree.Length();
            }
        }

        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                lengths[i] /= counts[i];
            }
        }

        fmt::print("Length vs depth: \n");
        for (size_t i = 1; i < counts.size(); ++i) {
            fmt::print("{}\t{}\n", i, lengths[i]);
        }
    }
}

TEST_CASE("BTC")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 1000,
           maxLength = 100;

    size_t n = 10000;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.Enable(NodeType::Add, 1);
    grammar.Enable(NodeType::Mul, 1);
    grammar.Enable(NodeType::Sub, 1);
    grammar.Enable(NodeType::Div, 1);

    BalancedTreeCreator btc{ grammar, inputs, /* bias= */ 1.0};

    Operon::Random random(std::random_device {}());

    std::vector<size_t> lengths(n);


    SUBCASE("Symbol frequencies")
    {
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, btc, lengths);
        auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
        fmt::print("Symbol frequencies: \n");
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1u << i));
            if (!grammar.IsEnabled(node.Type))
                continue;
            fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    SUBCASE("Length histogram") 
    {
        int reps = 50;
        std::vector<double> counts(maxLength+1, 0);

        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, btc, lengths);
            std::vector<size_t> actualLengths(trees.size());
            std::transform(trees.begin(), trees.end(), actualLengths.begin(), [](const auto& t) { return t.Length(); });
            auto cnt = CalculateHistogram(actualLengths);
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += cnt[j];
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
        int reps = 50;
        std::vector<double> counts;

        double avgShape = 0.0;
        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, btc, lengths);
            std::vector<size_t> shapes(trees.size());
            std::transform(trees.begin(), trees.end(), shapes.begin(), [](const auto& t) { return std::transform_reduce(std::execution::seq, t.Nodes().begin(), t.Nodes().end(), 0UL, std::plus<size_t>{}, [](const auto& node) { return node.Length+1; }); });

            avgShape +=  std::reduce(shapes.begin(), shapes.end()) / static_cast<double>(trees.size());
            auto cnt = CalculateHistogram(shapes);
            if (counts.size() < cnt.size()) { counts.resize(cnt.size()); }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += cnt[j];
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
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 1000,
           maxLength = 100;

    size_t n = 10000;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.Enable(NodeType::Add, 1);
    grammar.Enable(NodeType::Mul, 1);
    grammar.Enable(NodeType::Sub, 1);
    grammar.Enable(NodeType::Div, 1);

    ProbabilisticTreeCreator ptc{ grammar, inputs };

    Operon::Random random(std::random_device {}());

    std::vector<size_t> lengths(n);

    SUBCASE("Simple tree") 
    {
        auto tree = ptc(random, 9, 0, maxDepth);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    SUBCASE("Symbol frequencies")
    {
        std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
        auto trees = GenerateTrees(random, ptc, lengths);
        auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
        auto symbolFrequencies = CalculateSymbolFrequencies(trees);

        fmt::print("Symbol frequencies: \n");
        for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
            auto node = Node(static_cast<NodeType>(1u << i));
            if (!grammar.IsEnabled(node.Type))
                continue;
            fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    SUBCASE("Length histogram") 
    {
        int reps = 50;
        std::vector<double> counts(maxLength+1, 0);

        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, ptc, lengths);
            std::vector<size_t> actualLengths(trees.size());
            std::transform(trees.begin(), trees.end(), actualLengths.begin(), [](const auto& t) { return t.Length(); });
            auto cnt = CalculateHistogram(actualLengths);
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += cnt[j];
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
        int reps = 50;
        std::vector<double> counts;

        double avgShape = 0.0;
        for (int i = 0; i < reps; ++i) {
            std::generate(lengths.begin(), lengths.end(), [&]() { return sizeDistribution(random); });
            auto trees = GenerateTrees(random, ptc, lengths);
            std::vector<size_t> shapes(trees.size());
            std::transform(trees.begin(), trees.end(), shapes.begin(), [](const auto& t) { return std::transform_reduce(std::execution::seq, t.Nodes().begin(), t.Nodes().end(), 0UL, std::plus<size_t>{}, [](const auto& node) { return node.Length+1; }); });
            avgShape +=  std::reduce(shapes.begin(), shapes.end()) / static_cast<double>(trees.size());
            auto cnt = CalculateHistogram(shapes);
            if (counts.size() < cnt.size()) { counts.resize(cnt.size()); }
            for (size_t j = 0; j < cnt.size(); ++j) {
                counts[j] += cnt[j];
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
}
