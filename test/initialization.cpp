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
        size_t maxDepth = 5, maxLength = 50;

        auto creator = GrowTreeCreator(maxDepth, maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd(std::random_device{}());

        auto tree = creator(rd, grammar, inputs);
        fmt::print("Tree length: {}\n", tree.Length());
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
    }

    TEST_CASE("Tree initialization (full)")
    {
        auto target = "Y";
        auto ds = Dataset("../data/Poly-10 y.csv", true);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
        size_t maxDepth = 5, maxLength = 50;

        auto creator = FullTreeCreator(maxDepth, maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd(std::random_device{}());

        auto tree = creator(rd, grammar, inputs);
        fmt::print("Tree length: {}\n", tree.Length());
        fmt::print("{}\n", TreeFormatter::Format(tree, ds));
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

