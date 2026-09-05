// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"

#include "../../../source/core/subtree_rewrite.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"

namespace Operon::Test {

namespace {
    auto Add() -> Node
    {
        return Node::Function(static_cast<Hash>(BuiltinOp::Add), 2);
    }
} // namespace

TEST_CASE("Mapped subtree segments preserve Ref safety", "[operators]")
{
    auto nodes = Operon::Vector<Node> { Node::Constant(1), Node::Constant(2), Add(), Node::Ref(2), Add() };

    auto const unchangedSegments = Operon::Vector<detail::SourceSegment> { { 0, 3 }, { 3, 1 }, { 4, 1 } };
    auto const unchanged = detail::RewriteSegments(nodes, unchangedSegments);
    REQUIRE(unchanged[3].IsRef());
    CHECK(unchanged[3].RefTo == 2);
    CHECK(Tree(unchanged).UpdateNodes().Validate());

    auto const reorderedSegments = Operon::Vector<detail::SourceSegment> { { 3, 1 }, { 0, 3 }, { 4, 1 } };
    CHECK_THROWS_AS(detail::RewriteSegments(nodes, reorderedSegments), std::invalid_argument);

    auto const incompleteSegments = Operon::Vector<detail::SourceSegment> { { 0, 3 }, { 4, 1 } };
    CHECK_THROWS_AS(detail::RewriteSegments(nodes, incompleteSegments), std::invalid_argument);
}

TEST_CASE("ShuffleSubtreesMutation preserves Ref validity", "[operators]")
{
    auto const add = Add();
    auto const mul = Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2);
    auto const tree = Tree({ Node::Constant(1), Node::Constant(2), add, Node::Ref(2), mul }).UpdateNodes();
    auto random = Operon::RandomGenerator(1234);
    auto const mutation = ShuffleSubtreesMutation {};

    for (auto i = 0; i < 100; ++i) {
        auto const child = mutation(random, tree);
        CHECK(child.Validate());
    }
}

TEST_CASE("InsertSubtreeMutation produces valid tree", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    auto const maxDepth { 1000 };
    auto const maxLength { 100 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Add>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Mul>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Sub>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Div>().HashValue, 1);

    BalancedTreeCreator btc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(1234);
    auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
    auto targetLen = sizeDistribution(random);

    auto tree = btc(random, targetLen, 1, maxDepth);

    InsertSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &btc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, 2 * targetLen, maxDepth);
    auto child = mut(random, tree);

    CHECK(child.Length() > 0);
    CHECK(child.Length() <= 2 * targetLen);
    CHECK(child.Validate());
}

TEST_CASE("RemoveSubtreeMutation replaces a random subtree with the grammar-minimal terminal", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    auto const maxDepth { 1000 };
    auto const maxLength { 100 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Add>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Mul>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Sub>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Div>().HashValue, 1);

    BalancedTreeCreator btc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(1234);
    auto const targetLen = size_t { 30 };
    auto tree = btc(random, targetLen, 1, maxDepth);
    auto const originalLength = tree.Length();

    RemoveSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &btc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, maxDepth);

    // Every replacement subtree is a single terminal (the smallest possible),
    // so the result can only shrink or stay the same length - never grow -
    // regardless of which node happens to be picked.
    for (auto i = 0; i < 25; ++i) {
        auto child = mut(random, tree);
        CHECK(child.Length() > 0);
        CHECK(child.Length() <= originalLength);
        CHECK(child.Validate());
    }
}

TEST_CASE("RemoveSubtreeMutation on a single-node tree does not crash", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    auto const maxDepth { 1000 };
    auto const maxLength { 100 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Add>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Mul>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Sub>().HashValue, 1);
    grammar.SetFrequency(Util::MakeOp<BuiltinOp::Div>().HashValue, 1);

    BalancedTreeCreator btc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(4321);
    auto tree = btc(random, /*targetLen=*/1, 1, maxDepth);
    REQUIRE(tree.Length() == 1);

    RemoveSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &btc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, maxDepth);
    auto child = mut(random, tree);
    CHECK(child.Length() == 1);
}

TEST_CASE("Mutation tree stays within bounds", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    auto const maxDepth { 1000 };
    auto const maxLength { 50 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;

    Operon::RandomGenerator random(1234);

    for (int i = 0; i < 100; ++i) {
        auto tree = btc(random, 10, 1, maxDepth);
        InsertSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &btc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, maxLength, maxDepth);
        auto child = mut(random, tree);
        CHECK(child.Length() > 0);
        CHECK(child.Length() <= static_cast<size_t>(maxLength));
        CHECK(child.Validate());
    }
}

TEST_CASE("ReplaceSubtreeMutation via PTC2 respects maxDepth", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    constexpr size_t maxLength = 50;
    constexpr size_t maxDepth = 5;

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | BuiltinOp::Log | BuiltinOp::Exp);

    // PTC2 is the creator whose depth enforcement is being exercised here: the
    // mutation hands it a per-call depth budget and relies on the creator to
    // honor it when growing the replacement subtree.
    ProbabilisticTreeCreator const ptc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;
    ReplaceSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &ptc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, maxDepth, maxLength);

    Operon::RandomGenerator random(42);

    // start from a tree that already satisfies the budget (PTC2 enforces it),
    // then keep mutating; every replacement subtree must stay within its budget
    // so no tree in the chain can exceed maxDepth.
    auto tree = ptc(random, maxLength, 1, maxDepth);
    REQUIRE(tree.Depth() <= maxDepth);
    for (int i = 0; i < 5000; ++i) {
        tree = mut(random, std::move(tree));
        CHECK(tree.Depth() <= maxDepth);
        CHECK(tree.Validate());
    }
}

TEST_CASE("InsertSubtreeMutation leaves trees without eligible n-ary operators unchanged", "[operators]")
{
    auto ds = Dataset("./data/Poly-10.csv", true);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);
    auto const maxDepth { 1000 };
    auto const maxLength { 50 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc { &grammar, inputs, /* bias= */ 0.0, maxLength };
    UniformCoefficientInitializer cfi;

    auto const variableHash = ds.GetVariable("X1").value().Hash;
    Node variable(NodeType::Variable);
    variable.HashValue = variable.CalculatedHashValue = variableHash;

    auto sin = Util::MakeOp<BuiltinOp::Sin>();

    Tree const tree({ variable, sin });

    Operon::RandomGenerator random(1234);
    InsertSubtreeMutation const mut(gsl::not_null<Operon::CreatorBase const*> { &btc }, gsl::not_null<Operon::CoefficientInitializerBase const*> { &cfi }, maxLength, maxDepth);
    auto child = mut(random, tree);

    CHECK(child.Length() == tree.Length());
    CHECK(child[child.Length() - 1].IsOp<BuiltinOp::Sin>());
    CHECK(child[0].HashValue == variableHash);
}

} // namespace Operon::Test
