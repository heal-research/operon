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
        auto ds = Dataset("../data/poly-10.csv", true);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
        size_t maxDepth = 20, maxLength = 50;

        auto creator = GrowTreeCreator(maxDepth, maxLength);
        Grammar grammar;
        Operon::Random::JsfRand<64> rd;

        fmt::print("Min function arity: {}\n", grammar.MinimumFunctionArity());

        auto tree = creator(rd, grammar, inputs);
        fmt::print("{}\n", InfixFormatter::Format(tree, ds));

        std::vector<Tree> trees(100000);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });

#if _MSC_VER
        double avgLength = std::reduce(trees.begin(), trees.end(), 0.0, [](double partial, const auto& t) { return partial + t.Length(); });
#else
        double avgLength = std::transform_reduce(std::execution::seq, trees.begin(), trees.end(), 0UL, [](size_t lhs, size_t rhs) { return lhs + rhs; }, [&](auto &t) { return t.Length();} );
#endif    
        avgLength /= trees.size();

        fmt::print("Average length: {}\n", avgLength);
    }
}

