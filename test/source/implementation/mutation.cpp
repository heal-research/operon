// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"

namespace Operon::Test {

TEST_CASE("InsertSubtreeMutation produces valid tree", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y")->Hash);
    auto const maxDepth{1000};
    auto const maxLength{100};

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Log | NodeType::Exp);
    grammar.SetFrequency(Node(NodeType::Add).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Mul).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Sub).HashValue, 1);
    grammar.SetFrequency(Node(NodeType::Div).HashValue, 1);

    BalancedTreeCreator btc{&grammar, inputs, /* bias= */ 0.0};
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
    auto targetLen = sizeDistribution(random);

    auto tree = btc(random, targetLen, 1, maxDepth);

    InsertSubtreeMutation mut(gsl::not_null<Operon::CreatorBase const*>{&btc}, gsl::not_null<Operon::CoefficientInitializerBase const*>{&cfi}, 2 * targetLen, maxDepth);
    auto child = mut(random, tree);

    CHECK(child.Length() > 0);
    CHECK(child.Length() <= 2 * targetLen);
}

TEST_CASE("Mutation tree stays within bounds", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y")->Hash);
    auto const maxDepth{1000};
    auto const maxLength{50};

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc{&grammar, inputs};
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(1234);

    for (int i = 0; i < 100; ++i) {
        auto tree = btc(random, 10, 1, maxDepth);
        InsertSubtreeMutation mut(gsl::not_null<Operon::CreatorBase const*>{&btc}, gsl::not_null<Operon::CoefficientInitializerBase const*>{&cfi}, maxLength, maxDepth);
        auto child = mut(random, tree);
        CHECK(child.Length() > 0);
        CHECK(child.Length() <= static_cast<size_t>(maxLength));
    }
}

} // namespace Operon::Test
