// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <fmt/core.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
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
    auto variables = ds.GetVariables();
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target)->Hash);
    auto const maxDepth{1000};
    auto const maxLength{100};

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
