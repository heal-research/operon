// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>

#include "operon/core/dataset.hpp"
#include "operon/core/format.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"

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
    grammar.SetFrequency(Node(NodeType::Add).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Mul).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Sub).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Div).HashValue, 1);

    BalancedTreeCreator btc { grammar, inputs, /* bias= */ 0.0 };
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(std::random_device {}());
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
    auto targetLen = sizeDistribution(random);

    auto tree = btc(random, targetLen, 1, maxDepth);
    fmt::print("{}\n", TreeFormatter::Format(tree, ds));

    InsertSubtreeMutation mut(btc, cfi, 2 * targetLen, maxDepth);
    auto child = mut(random, tree);
    fmt::print("{}\n", TreeFormatter::Format(child, ds));
}

}
