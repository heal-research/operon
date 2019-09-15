#include <catch2/catch.hpp>
#include <algorithm>
#include <execution>
#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/grammar.hpp"
#include "operators/initialization.hpp"

namespace Operon::Test
{
    TEST_CASE("Tree initialization (grow)")
    {
        auto target = "Y";
        auto ds = Dataset("../data/Poly-10 y.csv", true);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
        size_t maxDepth = 12, maxLength = 50;

        auto creator = GrowTreeCreator(maxDepth, maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd(std::random_device{}());
        
        auto trees = std::vector<Tree>(10000);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

        auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t>{ }, [](const auto& tree) { return tree.Length(); });
        fmt::print("Full tree creator - length({},{}) = {}\n", maxDepth, maxLength, totalLength / trees.size());
        auto findIdx = [](uint16_t v) -> gsl::index
        {
            gsl::index i = 0;
            while ((v >>= 1) != 0) { ++i; }
            return i;
        };

        std::array<size_t, 14> symbolFrequencies;
        symbolFrequencies.fill(0u);
        for(const auto& tree : trees)
        {
            for(const auto& node : tree.Nodes())
            {
                symbolFrequencies[findIdx(static_cast<uint16_t>(node.Type))]++;
            }
        }
        fmt::print("Symbol frequencies: \n");

        for (size_t i = 0; i < symbolFrequencies.size(); ++i)
        {
            auto node = Node(static_cast<NodeType>(1u << i));
            fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    TEST_CASE("Tree initialization (full)")
    {
        auto target = "Y";
        auto ds = Dataset("../data/Poly-10 y.csv", true);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
        size_t maxDepth = 12, maxLength = 50;

        auto creator = FullTreeCreator(static_cast<int>(std::log2(maxLength)), maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd(std::random_device{}());

        auto trees = std::vector<Tree>(10000);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

        auto totalLength = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0.0, std::plus<size_t>{ }, [](const auto& tree) { return tree.Length(); });
        fmt::print("Full tree creator - length({},{}) = {}\n", maxDepth, maxLength, totalLength / trees.size());
        auto findIdx = [](uint16_t v) -> gsl::index
        {
            gsl::index i = 0;
            while ((v >>= 1) != 0) { ++i; }
            return i;
        };

        std::array<size_t, 14> symbolFrequencies;
        symbolFrequencies.fill(0u);
        for(const auto& tree : trees)
        {
            for(const auto& node : tree.Nodes())
            {
                symbolFrequencies[findIdx(static_cast<uint16_t>(node.Type))]++;
            }
        }
        fmt::print("Symbol frequencies: \n");

        for (size_t i = 0; i < symbolFrequencies.size(); ++i)
        {
            auto node = Node(static_cast<NodeType>(1u << i));
            fmt::print("{}\t{:.3f} %\n", node.Name(), symbolFrequencies[i] / totalLength);
        }
    }

    TEST_CASE("Tree depth calculation")
    {
        auto target = "Y";
        auto ds = Dataset("../data/Poly-10 y.csv", true);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
        size_t maxDepth = 20, maxLength = 50;

        auto creator = GrowTreeCreator(maxDepth, maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd(std::random_device{}());

        fmt::print("Min function arity: {}\n", grammar.MinimumFunctionArity());

        auto tree = creator(rd, grammar, inputs);
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }
}

