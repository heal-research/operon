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

    std::vector<double> observed(NodeTypes::Count, 0);
    size_t r = grammar.EnabledSymbols().size() + 1;

    const size_t nTrials = 1'000'000;
    for (auto i = 0u; i < nTrials; ++i)
    {
        auto node = grammar.SampleRandomSymbol(rd, 0, 2);
        ++observed[NodeTypes::GetIndex(node.Type)];
    }
    std::transform(std::execution::unseq, observed.begin(), observed.end(), observed.begin(), [&](double v) { return v / nTrials; });
    std::vector<double> actual(NodeTypes::Count, 0);
    for(size_t i = 0; i < observed.size(); ++i)
    {
        auto nodeType = static_cast<NodeType>(1u << i);
        actual[NodeTypes::GetIndex(nodeType)] = grammar.GetFrequency(nodeType);
    }
    auto freqSum = std::reduce(std::execution::unseq, actual.begin(), actual.end(), 0.0, std::plus{});
    std::transform(std::execution::unseq, actual.begin(), actual.end(), actual.begin(), [&](double v) { return v / freqSum; });
    auto chi = 0.0;
    for(auto i = 0u; i < observed.size(); ++i)
    {
        auto nodeType = static_cast<NodeType>(1u << i);
        if (!grammar.IsEnabled(nodeType)) continue;
        auto x = observed[i];
        auto y = actual[i]; 
        fmt::print("{:>8} observed {:.4f}, expected {:.4f}\n", Node(nodeType).Name(), x , y);
        chi += (x - y) * (x - y) / y;
    }
    chi *= nTrials;
    
    auto criticalValue = r + 2 * std::sqrt(r);
    fmt::print("chi = {}, critical value = {}\n", chi, criticalValue);
    REQUIRE(chi <= criticalValue);
}

TEST_CASE("Tree shape")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 1000,
           maxLength = 50;
    auto sizeDistribution = std::uniform_int_distribution<size_t>(maxLength, maxLength);
    auto creator = GrowTreeCreator(sizeDistribution, maxDepth, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::TypeCoherent);
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    auto tree = creator(rd, grammar, inputs);
    fmt::print("Tree length: {}\n", tree.Length());
    fmt::print("{}\n", TreeFormatter::Format(tree, ds));
}

TEST_CASE("Tree initialization (grow)")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });

    size_t maxDepth = 1000, maxLength = 100;

    const size_t nTrees = 100'000;

    //auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
    auto sizeDistribution = std::normal_distribution<operon::scalar_t> { maxLength / 2.0, 10 };
    auto creator = GrowTreeCreator(sizeDistribution, maxDepth, maxLength);
    Grammar grammar;
    grammar.SetConfig(Grammar::TypeCoherent);
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    auto trees = std::vector<Tree>(nTrees);
    std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

    auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t> {}, [](const auto& tree) { return tree.Length(); });
    fmt::print("Grow tree creator - length({},{}) = {}\n", maxDepth, maxLength, totalLength / trees.size());

    SECTION("Symbol frequencies")
    {
        std::array<size_t, NodeTypes::Count> symbolFrequencies;
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
    }

    SECTION("Variable frequencies")
    {
        fmt::print("Variable frequencies:\n");
        size_t totalVars = 0;
        std::vector<size_t> variableFrequencies(inputs.size());
        for (const auto& t : trees) {
            for (const auto& node : t.Nodes()) {
                if (node.IsVariable()) {
                    if (auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto& v) { return node.HashValue == v.Hash; }); it != inputs.end()) {
                        variableFrequencies[it->Index]++;
                        totalVars++;
                    } else {
                        fmt::print("Could not find variable {} with hash {} and calculated hash {} within the inputs\n", node.Name(), node.HashValue, node.CalculatedHashValue);
                        std::exit(EXIT_FAILURE);
                    }
                }
            }
        }
        for (const auto& v : inputs) {
            fmt::print("{}\t{:.3f}%\n", ds.GetName(v.Hash), static_cast<operon::scalar_t>(variableFrequencies[v.Index]) / totalVars);
        }
    }

    SECTION("Tree length histogram")
    {
        std::vector<size_t> lengthHistogram(maxLength + 1);
        for(auto& tree : trees)
        {
            lengthHistogram[tree.Length()]++;
        }
        fmt::print("Tree length histogram:\n");
        for (auto i = 1u; i < lengthHistogram.size(); ++i)
        {
            fmt::print("{}\t{}\n", i, lengthHistogram[i]);
        }
    }

    SECTION("Tree depth histogram")
    {
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
            fmt::print("{}\t{}\n", i, depthHistogram[i]);
        }
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

    auto sizeDistribution = std::uniform_int_distribution<size_t>(2, maxLength);
    auto creator = GrowTreeCreator(sizeDistribution, maxDepth, maxLength);
    Grammar grammar;
    Operon::Random::JsfRand<64> rd(std::random_device {}());

    //fmt::print("Min function arity: {}\n", grammar.MinimumFunctionArity());

    auto tree = creator(rd, grammar, inputs);
    fmt::print("{}\n", TreeFormatter::Format(tree, ds));
}
}
