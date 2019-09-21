#include "operators/initialization.hpp"
#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include <algorithm>
#include <catch2/catch.hpp>
#include <execution>

namespace Operon::Test {
TEST_CASE("Sample nodes from grammar")
{
    Grammar grammar;
    grammar.SetConfig(Grammar::TypeCoherent);
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    std::vector<size_t> observed(4, 0);
    double r = observed.size() + 1;

    const size_t nTrials = 10'000;
    for (auto i = 0u; i < nTrials; ++i)
    {
        auto node = grammar.SampleRandomSymbol(rd, 0, 2);
        ++observed[NodeTypes::GetIndex(node.Type)];
    }
    auto freqSum = 0.0;
    fmt::print("Observed counts:\n");
    for(size_t i = 0; i < observed.size(); ++i)
    {
        auto nodeType = static_cast<NodeType>(1u << i);
        fmt::print("{}:\t{}\n", Node(nodeType).Name(), observed[i]);
        freqSum += grammar.GetFrequency(nodeType);
    }
    auto sum = 0.0;
    for(auto i = 0u; i < observed.size(); ++i)
    {
        auto nodeType = static_cast<NodeType>(1u << i);
        auto x = static_cast<double>(observed[i]);
        auto y = grammar.GetFrequency(nodeType) / freqSum; 
        fmt::print("observed {:.3f}, expected {:.3f}\n", x / nTrials, y);
        sum += x * x / y;
    }

    auto chi = sum / nTrials - nTrials; 
    auto c = 2 * std::sqrt(r);
    auto lower = r - c;
    auto upper = r + c;
    REQUIRE((chi >= lower && chi <= upper));
}

TEST_CASE("Tree initialization (grow)")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 12, maxLength = 50;

    const size_t nTrees = 10'000;

    auto creator = GrowTreeCreator(maxDepth, maxLength);
    Grammar grammar;
    grammar.SetConfig(Grammar::TypeCoherent);
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    auto trees = std::vector<Tree>(nTrees);
    std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

    auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
    fmt::print("Grow tree creator - length({},{}) = {}\n", maxDepth, maxLength, totalLength / trees.size());

    std::array<size_t, 14> symbolFrequencies;
    symbolFrequencies.fill(0u);
    for (const auto& tree : trees) {
        for (const auto& node : tree.Nodes()) {
            symbolFrequencies[NodeTypes::GetIndex(node.Type)]++;
        }
    }
    fmt::print("Symbol frequencies: \n");

    for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
        auto node = Node(static_cast<NodeType>(1u << i));
        if (!grammar.IsEnabled(node.Type))
            continue;
        fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
    }

    auto [minLen, maxLen] = std::minmax_element(trees.begin(), trees.end(), [](const auto& lhs, const auto& rhs) { return lhs.Length() < rhs.Length(); });

    std::vector<size_t> lengthHistogram(maxLen->Length() + 1);
    for(auto& tree : trees)
    {
        lengthHistogram[tree.Length()]++;
    }
    fmt::print("Tree length histogram:\n");
    for (auto i = 0u; i < lengthHistogram.size(); ++i)
    {
        auto v = lengthHistogram[i];
        if (v == 0) continue;
        fmt::print("{}:\t{}\n", i, lengthHistogram[i]);
    }
    auto [minDep, maxDep] = std::minmax_element(trees.begin(), trees.end(), [](const auto& lhs, const auto& rhs) { return lhs.Depth() < rhs.Depth(); });
    std::vector<size_t> depthHistogram(maxDep->Depth() + 1);
    for(auto& tree : trees)
    {
        depthHistogram[tree.Depth()]++;
    }
    fmt::print("Tree depth histogram:\n");
    for (auto i = 0u; i < depthHistogram.size(); ++i)
    {
        auto v = depthHistogram[i];
        if (v == 0) continue;
        fmt::print("{}:\t{}\n", i, depthHistogram[i]);
    }
}

TEST_CASE("Tree initialization (full)")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 12, maxLength = 50;

    const size_t nTrees = 10'000;

    auto creator = FullTreeCreator(maxDepth, maxLength);
    Grammar grammar;
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    auto trees = std::vector<Tree>(nTrees);
    std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

    auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
    fmt::print("Full tree creator - length({},{}) = {}\n", maxDepth, maxLength, totalLength / trees.size());

    std::array<size_t, 14> symbolFrequencies;
    symbolFrequencies.fill(0u);
    for (const auto& tree : trees) {
        for (const auto& node : tree.Nodes()) {
            symbolFrequencies[NodeTypes::GetIndex(node.Type)]++;
        }
    }
    fmt::print("Symbol frequencies: \n");

    for (size_t i = 0; i < symbolFrequencies.size(); ++i) {
        auto node = Node(static_cast<NodeType>(1u << i));
        if (!grammar.IsEnabled(node.Type))
            continue;
        fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
    }

    auto [minLen, maxLen] = std::minmax_element(trees.begin(), trees.end(), [](const auto& lhs, const auto& rhs) { return lhs.Length() < rhs.Length(); });

    std::vector<size_t> lengthHistogram(maxLen->Length() + 1);
    for(auto& tree : trees)
    {
        lengthHistogram[tree.Length()]++;
    }
    fmt::print("Tree length histogram:\n");
    for (auto i = 0u; i < lengthHistogram.size(); ++i)
    {
        auto v = lengthHistogram[i];
        if (v == 0) continue;
        fmt::print("{}:\t{}\n", i, lengthHistogram[i]);
    }
    auto [minDep, maxDep] = std::minmax_element(trees.begin(), trees.end(), [](const auto& lhs, const auto& rhs) { return lhs.Depth() < rhs.Depth(); });
    std::vector<size_t> depthHistogram(maxDep->Depth() + 1);
    for(auto& tree : trees)
    {
        depthHistogram[tree.Depth()]++;
    }
    fmt::print("Tree depth histogram:\n");
    for (auto i = 0u; i < depthHistogram.size(); ++i)
    {
        auto v = depthHistogram[i];
        if (v == 0) continue;
        fmt::print("{}:\t{}\n", i, depthHistogram[i]);
    }
}

TEST_CASE("Tree depth calculation")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 20, maxLength = 50;

    auto creator = GrowTreeCreator(maxDepth, maxLength);
    Grammar grammar;
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    //fmt::print("Min function arity: {}\n", grammar.MinimumFunctionArity());

    auto tree = creator(rd, grammar, inputs);
    fmt::print("{}\n", TreeFormatter::Format(tree, ds));
}
}
