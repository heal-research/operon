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

    void CheckFinite(std::vector<Scalar> const& values)
    {
        for (auto const value : values) {
            REQUIRE(std::isfinite(value));
        }
    }

    void CheckFiniteEqual(std::vector<Scalar> const& before, std::vector<Scalar> const& after, Scalar tolerance)
    {
        REQUIRE(before.size() == after.size());
        CheckFinite(before);
        CheckFinite(after);
        for (std::size_t row = 0; row < before.size(); ++row) {
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
    // source span. This verifies the post-fix rejection contract; without a
    // sanitizer it cannot deterministically diagnose the prior out-of-bounds access.
    auto nodes = Operon::Vector<Node> { Node::Constant(1), Node::Ref(99), Add() };
    auto const identity = Operon::Vector<detail::PermutationSegment> { { 0, nodes.size() } };
    CHECK_FALSE(detail::PermuteSegments(nodes, identity));

    // RefTo equal to the source size is the first invalid index; reject it as
    // well so the guard is pinned as >= rather than merely >.
    auto boundary = Operon::Vector<Node> { Node::Constant(1), Node::Ref(3), Add() };
    auto const boundaryIdentity = Operon::Vector<detail::PermutationSegment> { { 0, boundary.size() } };
    CHECK_FALSE(detail::PermuteSegments(boundary, boundaryIdentity));
}

TEST_CASE("Sort leaves malformed out-of-range Refs untouched", "[operators]")
{
    for (auto const refTo : { uint16_t{3}, uint16_t{99} }) {
        auto tree = Tree({ Node::Constant(1), Node::Ref(refTo), Add() }).UpdateNodes();
        auto const before = Fixture(tree);

        REQUIRE_FALSE(tree.Validate());
        tree.Sort();

        CHECK(Fixture(tree) == before);
        auto const validation = tree.Validate();
        REQUIRE_FALSE(validation);
        CHECK(validation.error() == TreeValidationError::RefNotBackward);
    }
}

TEST_CASE("Tree transforms preserve finite evaluation", "[properties][tree-transforms]")
{
    constexpr Scalar tolerance = 1e-5F;
    // A single finite input column keeps the evaluation contract explicit:
    // every tree in this corpus must produce finite output for every row.
    Dataset const domain({ "X" }, { { -2.0F, -0.5F, 0.5F, 2.0F } });
    auto const x = domain.GetVariable("X").value().Hash;

    SECTION("UpdateNodes is metadata-idempotent") {
        auto tree = Tree({ Variable(x), Node::Constant(2), Add(), Node::Ref(2), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const once = tree.Nodes();
        tree.UpdateNodes();
        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        REQUIRE(tree.Nodes().size() == once.size());
        for (std::size_t index = 0; index < once.size(); ++index) {
            CHECK(SameNodeMetadata(tree.Nodes()[index], once[index]));
        }
    }

    SECTION("Sort preserves finite commutative evaluation, including extrema") {
        auto tree = Tree({ Variable(x), Node::Constant(3), Add(), Node::Constant(2), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const before = Evaluate(tree, domain);
        tree.Sort();
        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);

        for (auto const op : { BuiltinOp::Fmin, BuiltinOp::Fmax }) {
            // Use unlike leaves: constants share a hash, so cannot prove a reordering.
            auto extrema = Tree({ Variable(x), Node::Constant(3), Node::Function(Hash(op), 2) }).UpdateNodes();
            [[maybe_unused]] auto const& hashed = extrema.Hash(HashMode::Strict);
            if (extrema[0] < extrema[1]) {
                std::swap(extrema.Nodes()[0], extrema.Nodes()[1]);
                [[maybe_unused]] auto const& rehashed = extrema.UpdateNodes().Hash(HashMode::Strict);
            }
            REQUIRE(extrema[1] < extrema[0]);
            auto const extremaBefore = Evaluate(extrema, domain);
            auto const beforeFixture = Fixture(extrema);
            extrema.Sort();
            INFO(fmt::format("op={}, tree={}", static_cast<unsigned>(op), Fixture(extrema)));
            REQUIRE(extrema.Validate());
            CHECK(extrema[0] < extrema[1]);
            CHECK(Fixture(extrema) != beforeFixture);
            CheckFiniteEqual(extremaBefore, Evaluate(extrema, domain), tolerance);
        }
    }
    SECTION("Sort snapshots unequal child spans before reordering") {
        auto tree = Tree({ Node::Constant(1), Node::Constant(9), Node::Constant(4), Add(), Add() }).UpdateNodes();
        [[maybe_unused]] auto const& hashed = tree.Hash(HashMode::Strict);
        auto const before = Evaluate(tree, domain);

        tree.Sort();

        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        REQUIRE(tree.Length() == 5);
        CHECK(tree[2].IsFunction());
        CHECK(tree[3].IsConstant());
        CHECK(tree[3].Value == 1);
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }

    SECTION("Sort remaps self-contained Ref subtrees") {
        auto tree = Tree({ Node::Constant(1), Node::Constant(5), Node::Ref(1), Node::Function(Hash(BuiltinOp::Mul), 2), Add() }).UpdateNodes();
        [[maybe_unused]] auto const& hashed = tree.Hash(HashMode::Strict);
        auto const before = Evaluate(tree, domain);

        tree.Sort();

        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        REQUIRE(tree[1].IsRef());
        CHECK(tree[1].RefTo == 0);
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }

    SECTION("Sort leaves a cross-subtree forward Ref order unchanged") {
        auto tree = Tree({ Node::Constant(5), Node::Ref(0), Add() }).UpdateNodes();
        [[maybe_unused]] auto const& hashed = tree.Hash(HashMode::Strict);
        auto const before = Evaluate(tree, domain);
        auto const beforeFixture = Fixture(tree);

        tree.Sort();

        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        CHECK(Fixture(tree) == beforeFixture);
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }

    SECTION("Reduce flattens nested addition without changing evaluation") {
        auto tree = Tree({ Variable(x), Node::Constant(2), Add(), Node::Constant(3), Add() }).UpdateNodes();
        auto const before = Evaluate(tree, domain);
        auto const beforeFixture = Fixture(tree);
        auto const beforeLength = tree.Length();
        tree.Reduce();
        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        CHECK(tree.Length() < beforeLength);
        CHECK(Fixture(tree) != beforeFixture);
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }

    SECTION("Simplify removes multiplicative identity without changing evaluation") {
        auto tree = Tree({ Variable(x), Node::Constant(1), Node::Function(Hash(BuiltinOp::Mul), 2) }).UpdateNodes();
        auto const before = Evaluate(tree, domain);
        auto const beforeFixture = Fixture(tree);
        tree.Simplify();
        INFO(fmt::format("tree={}", Fixture(tree)));
        REQUIRE(tree.Validate());
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
        CHECK(Fixture(tree) != beforeFixture);
        CheckFiniteEqual(before, Evaluate(tree, domain), tolerance);
    }
}

TEST_CASE("Structural operators honor deterministic configured bounds", "[properties][operators]")
{
    constexpr std::size_t maxDepth = 4;
    constexpr std::size_t maxLength = 15;
    Dataset const domain({ "X" }, { { -2.0F, -0.5F, 0.5F, 2.0F } });
    auto const inputs = domain.VariableHashes();
    PrimitiveSet grammar;
    // Restrict generated insertions to finite arithmetic on the test domain.
    grammar.SetConfig(PrimitiveSet::Arithmetic);
    grammar.Disable(Util::MakeOp<BuiltinOp::Div>().HashValue);
    grammar.SetMaximumArity(Add(), 3);
    ProbabilisticTreeCreator const creator { &grammar, inputs, 0.0, maxLength };
    UniformCoefficientInitializer initializer;

    SECTION("direct crossover inserts the selected donor subtree") {
        auto const left = Tree({ Node::Constant(11), Node::Constant(13), Add() }).UpdateNodes();
        auto const right = Tree({ Node::Constant(29), Node::Constant(31), Add() }).UpdateNodes();
        auto const beforeFixture = Fixture(left);
        auto const child = CrossoverBase::Cross(left, right, 0, 2);
        INFO(fmt::format("left={}, right={}, child={}", Fixture(left), Fixture(right), Fixture(child)));
        REQUIRE(child.Validate());
        CHECK(child.Length() == 5);
        CHECK(child[0].Value == 29);
        CHECK(child[1].Value == 31);
        CHECK(Fixture(child) != beforeFixture);
        CheckFinite(Evaluate(child, domain));
    }
    SECTION("configured crossover rejects oversized and overdeep donors") {
        auto const left = Tree({ Node::Constant(11), Node::Constant(13), Add() }).UpdateNodes();
        auto const right = Tree({ Node::Constant(1), Node::Constant(2), Add(), Node::Constant(3), Add(), Node::Constant(4), Add() }).UpdateNodes();
        auto const invalidDonor = right.Splice(right.Length() - 1U);
        REQUIRE(invalidDonor.Depth() > 2);
        REQUIRE(invalidDonor.Length() > 3);
        auto const unconstrained = CrossoverBase::Cross(left, right, 0, right.Length() - 1U);
        REQUIRE(unconstrained.Depth() > 2);
        REQUIRE(unconstrained.Length() > 3);
        auto const beforeFixture = Fixture(left);
        SubtreeCrossover const crossover { 1.0, 2, 3 };
        auto random = RandomGenerator(0x5EEDU);
        auto crossed = false;

        for (size_t i = 0; i < 64; ++i) {
            auto const child = crossover(random, left, right);
            INFO(fmt::format("iteration={}, child={}", i, Fixture(child)));
            REQUIRE(child.Validate());
            CHECK(child.Depth() <= 2);
            CHECK(child.Length() <= 3);
            CheckFinite(Evaluate(child, domain));
            crossed = crossed || Fixture(child) != beforeFixture;
        }
        REQUIRE(crossed);
    }

    SECTION("insertion grows a known eligible n-ary node") {
        auto const parent = Tree({ Variable(inputs[0]), Node::Constant(2), Add() }).UpdateNodes();
        auto random = RandomGenerator(0xF6B0A0D5ULL);
        InsertSubtreeMutation const mutation { gsl::not_null<CreatorBase const*> { &creator }, gsl::not_null<CoefficientInitializerBase const*> { &initializer }, maxDepth, maxLength };
        auto const child = mutation(random, parent);
        INFO(fmt::format("parent={}, child={}", Fixture(parent), Fixture(child)));
        REQUIRE(child.Validate());
        CHECK(child.Length() > parent.Length());
        CHECK(child.Length() <= maxLength);
        CHECK(child.Depth() <= maxDepth);
        CheckFinite(Evaluate(child, domain));
    }
}

TEST_CASE("ShuffleSubtreesMutation reorders distinct child subtrees", "[operators]")
{
    Dataset const domain({ "X" }, { { -2.0F, -0.5F, 0.5F, 2.0F } });
    auto const x = domain.GetVariable("X").value().Hash;
    auto const add = Add();
    // A ternary Add gives std::shuffle more than one non-identity permutation;
    // unlike leaves make every permutation externally observable.
    auto ternaryAdd = add;
    ternaryAdd.Arity = 3;
    auto const tree = Tree({ Variable(x), Node::Constant(2), Node::Constant(3), ternaryAdd }).UpdateNodes();
    auto const before = Evaluate(tree, domain);
    auto const beforeFixture = Fixture(tree);
    auto random = Operon::RandomGenerator(1234);
    auto const mutation = ShuffleSubtreesMutation {};
    auto changed = false;

    for (auto i = 0; i < 100; ++i) {
        auto const child = mutation(random, tree);
        INFO(fmt::format("iteration={}, tree={}", i, Fixture(child)));
        REQUIRE(child.Validate());
        CheckFiniteEqual(before, Evaluate(child, domain), 1e-5F);
        changed = changed || Fixture(child) != beforeFixture;
    }
    REQUIRE(changed);
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
