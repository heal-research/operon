#include <doctest/doctest.h>

#include "core/dataset.hpp"
#include "core/format.hpp"
#include "core/pset.hpp"
#include "core/stats.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/mutation.hpp"

namespace Operon::Test {
TEST_CASE("InsertSubtreeMutation")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });
    size_t maxDepth = 1000,
           maxLength = 100;

    Range range { 0, 250 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.Enable(NodeType::Add, 1);
    grammar.Enable(NodeType::Mul, 1);
    grammar.Enable(NodeType::Sub, 1);
    grammar.Enable(NodeType::Div, 1);

    BalancedTreeCreator btc { grammar, inputs, /* bias= */ 0.0 };

    Operon::RandomGenerator random(std::random_device {}());
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
    auto targetLen = sizeDistribution(random);

    auto tree = btc(random, targetLen, 1, maxDepth);
    fmt::print("{}\n", TreeFormatter::Format(tree, ds));

    InsertSubtreeMutation mut(btc, 2 * targetLen, maxDepth);
    auto child = mut(random, tree);
    fmt::print("{}\n", TreeFormatter::Format(child, ds));

    //auto v1 = Evaluate<Operon::Scalar>(tree, ds, range); 
    //auto v2 = Evaluate<Operon::Scalar>(child, ds, range); 
}

}
