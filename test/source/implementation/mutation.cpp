// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <vector>

#include "../operon_test.hpp"

#include "../../../source/core/subtree_rewrite.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"

namespace Operon::Test {

namespace {
    auto Add() -> Node
    {
        return Node::Function(static_cast<Hash>(BuiltinOp::Add), 2);
    }

    auto Variable(Hash hash) -> Node
    {
        Node node(NodeType::Variable);
        node.HashValue = node.CalculatedHashValue = hash;
        node.Optimize = false;
        return node;
    }

    auto Fixture(Tree const& tree) -> std::string
    {
        std::string result;
        for (auto const& node : tree.Nodes()) {
            fmt::format_to(std::back_inserter(result), "{{type={}, hash={}, value={}, arity={}, length={}, depth={}, level={}, parent={}, ref={}}}",
                static_cast<unsigned>(node.Type), node.HashValue, node.Value, node.Arity, node.Length, node.Depth, node.Level, node.Parent, node.RefTo);
        }
        return result;
    }

    auto SameNodeMetadata(Node const& lhs, Node const& rhs) -> bool
    {
        return lhs.HashValue == rhs.HashValue && lhs.CalculatedHashValue == rhs.CalculatedHashValue && lhs.Value == rhs.Value
            && lhs.Arity == rhs.Arity && lhs.Length == rhs.Length && lhs.Depth == rhs.Depth && lhs.Level == rhs.Level
            && lhs.Parent == rhs.Parent && lhs.Type == rhs.Type && lhs.IsEnabled == rhs.IsEnabled && lhs.Optimize == rhs.Optimize
            && lhs.RefTo == rhs.RefTo;
    }

    auto Evaluate(Tree const& tree, Dataset const& dataset) -> std::vector<Scalar>
    {
        using DTable = DispatchTable<Scalar>;
        return Interpreter<Scalar, DTable>::Evaluate(tree, dataset, Range{0, dataset.Rows<std::size_t>()});
    }

    void CheckFiniteEqual(std::vector<Scalar> const& before, std::vector<Scalar> const& after, Scalar tolerance)
    {
        REQUIRE(before.size() == after.size());
        for (std::size_t row = 0; row < before.size(); ++row) {
            REQUIRE(std::isfinite(before[row]));
            REQUIRE(std::isfinite(after[row]));
            CHECK(std::abs(before[row] - after[row]) <= tolerance);
        }
    }
} // namespace


TEST_CASE("Mapped subtree segments preserve Ref safety", "[operators]")
{
    auto nodes = Operon::Vector<Node> { Node::Constant(1), Node::Constant(2), Add(), Node::Ref(2), Add() };

    auto const unchangedSegments = Operon::Vector<detail::PermutationSegment> { { 0, 3 }, { 3, 1 }, { 4, 1 } };
    auto const unchanged = detail::PermuteSegments(nodes, unchangedSegments);
    REQUIRE(unchanged);
    CHECK((*unchanged)[3].IsRef());
    CHECK((*unchanged)[3].RefTo == 2);
    CHECK(Tree(*unchanged).UpdateNodes().Validate());

    auto const selfContained = Operon::Vector<Node> { Node::Constant(1), Node::Ref(0), Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2), Node::Constant(2), Add() };
    auto const reorderedSegments = Operon::Vector<detail::PermutationSegment> { { 3, 1 }, { 0, 3 }, { 4, 1 } };
    auto const reordered = detail::PermuteSegments(selfContained, reorderedSegments);
    REQUIRE(reordered);
    CHECK((*reordered)[2].IsRef());
    CHECK((*reordered)[2].RefTo == 1);
    CHECK(Tree(*reordered).UpdateNodes().Validate());

    auto const incompleteSegments = Operon::Vector<detail::PermutationSegment> { { 0, 3 }, { 4, 1 } };
    CHECK_FALSE(detail::PermuteSegments(nodes, incompleteSegments));

    auto const malformedSegments = Operon::Vector<detail::PermutationSegment> { { 0, 1 }, { 2, 1 }, { 1, 1 } };
    CHECK_FALSE(detail::PermuteSegments(Operon::Vector<Node> { Node::Constant(1), Node::Constant(2), Add() }, malformedSegments));
}

TEST_CASE("Mapped subtree segments reject out-of-range Refs", "[operators]")
{
    // The postfix shape is valid, but RefTo is intentionally outside the raw
    // source span. Permuting identity segments must reject it without indexing
    // destinations out of bounds.
    auto nodes = Operon::Vector<Node> { Node::Constant(1), Node::Ref(99), Add() };
    auto const identity = Operon::Vector<detail::PermutationSegment> { { 0, nodes.size() } };
    CHECK_FALSE(detail::PermuteSegments(nodes, identity));
}

TEST_CASE("Tree transforms preserve deterministic finite-domain properties", "[properties][tree-transforms]")
{
    constexpr std::uint64_t seed = 0xF62026ULL;
    constexpr Scalar tolerance = 1e-5F;
    // Domain deliberately excludes zero and +/-1: it avoids invalid identities
    // such as 0^0 and division by zero. Non-finite output is a failure, never
    // silently compared or special-cased.
    Dataset const domain({ "X", "Y" }, { { -2.0F, -0.5F, 0.5F, 2.0F }, { 0.0F, 0.0F, 0.0F, 0.0F } });
    auto const x = domain.GetVariable("X").value().Hash;

    SECTION("UpdateNodes is metadata-idempotent") {
        auto tree = Tree({ Variable(x), Node::Constant(2), Add(), Node::Ref(2), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const once = tree.Nodes();
        tree.UpdateNodes();
        INFO(fmt::format("seed={:#x}, tree={}", seed, Fixture(tree)));
        REQUIRE(tree.Validate());
        REQUIRE(tree.Nodes().size() == once.size());
        for (std::size_t index = 0; index < once.size(); ++index) {
            CHECK(SameNodeMetadata(tree.Nodes()[index], once[index]));
        }
    }

    SECTION("Sort preserves finite commutative evaluation") {
        auto tree = Tree({ Variable(x), Node::Constant(3), Add(), Node::Constant(2), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const before = Evaluate(tree, domain);
        std::ignore = tree.Hash(HashMode::Strict);
        tree.Sort();
        INFO(fmt::format("seed={:#x}, tree={}", seed, Fixture(tree)));
        REQUIRE(tree.Validate());
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);

        // Fmin/Fmax only rearrange finite, distinct constants, where their
        // commutativity is bit-exact and no signed-zero/NaN IEEE edge applies.
        auto extrema = Tree({ Node::Constant(-2), Node::Constant(3), Node::Function(Hash(BuiltinOp::Fmax), 2) }).UpdateNodes();
        std::ignore = extrema.Hash(HashMode::Strict);
        CHECK(std::bit_cast<std::uint32_t>(extrema.Nodes()[0].Value) == std::bit_cast<std::uint32_t>(Scalar{-2}));
        CHECK(std::bit_cast<std::uint32_t>(extrema.Nodes()[1].Value) == std::bit_cast<std::uint32_t>(Scalar{3}));
    }

    SECTION("Reduce then Simplify preserves finite evaluation") {
        auto tree = Tree({ Variable(x), Node::Constant(2), Add(), Node::Constant(3), Add(), Node::Constant(1), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const before = Evaluate(tree, domain);
        tree.Reduce().Simplify();
        INFO(fmt::format("seed={:#x}, tree={}", seed, Fixture(tree)));
        REQUIRE(tree.Validate());
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }
}

TEST_CASE("Structural operators honor deterministic configured bounds", "[properties][operators]")
{
    constexpr std::uint64_t seed = 0xF6B0A0D5ULL;
    constexpr std::size_t maxDepth = 4;
    constexpr std::size_t maxLength = 15;
    Dataset const domain({ "X", "Y" }, { { -2.0F, -0.5F, 0.5F, 2.0F }, { 0.0F, 0.0F, 0.0F, 0.0F } });
    auto inputs = domain.VariableHashes();
    std::erase(inputs, domain.GetVariable("Y").value().Hash);
    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);
    ProbabilisticTreeCreator const creator { &grammar, inputs, 0.0, maxLength };
    UniformCoefficientInitializer initializer;
    RandomGenerator random(seed);
    ReplaceSubtreeMutation const mutation { gsl::not_null<CreatorBase const*> { &creator }, gsl::not_null<CoefficientInitializerBase const*> { &initializer }, maxDepth, maxLength };
    SubtreeCrossover const crossover { 0.75, maxDepth, maxLength };

    auto left = creator(random, 11, 1, maxDepth);
    auto right = creator(random, 9, 1, maxDepth);
    for (std::size_t iteration = 0; iteration < 64; ++iteration) {
        left = mutation(random, std::move(left));
        auto child = crossover(random, left, right);
        INFO(fmt::format("seed={:#x}, iteration={}, mutation={}, crossover={}", seed, iteration, Fixture(left), Fixture(child)));
        REQUIRE(left.Validate());
        CHECK(left.Length() <= maxLength);
        CHECK(left.Depth() <= maxDepth);
        REQUIRE(child.Validate());
        CHECK(child.Length() <= maxLength);
        CHECK(child.Depth() <= maxDepth);
        right = std::move(child);
    }
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
