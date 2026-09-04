// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include <random>

#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/variable.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/core/types.hpp"
#include "operon/core/node.hpp"

#include "../../../source/core/subtree_rewrite.hpp"

namespace Operon::Test {

TEST_CASE("Subtree rewrites preserve and rebase backward Refs", "[operators]")
{
    auto const add = Node::Function(Hash(BuiltinOp::Add), 2);
    auto const mul = Node::Function(Hash(BuiltinOp::Mul), 2);

    SECTION("incoming source Ref targets replacement root") {
        auto source = Tree({Node::Constant(2), Node::Constant(3), add, Node::Ref(2), mul}).UpdateNodes();
        auto replacement = Tree({Node::Constant(5)}).UpdateNodes();
        auto const rewritten = detail::RewriteSubtree(source.Nodes(), detail::DescribeSubtree(source.Nodes(), 2), replacement.Nodes());
        auto const child = Tree(rewritten).UpdateNodes();
        CHECK(child.Nodes()[1].IsRef());
        CHECK(child.Nodes()[1].RefTo == 0);
        CHECK(child.Nodes()[2].Length == 2);
    }

    SECTION("self-contained donor Refs are rebased") {
        auto source = Tree({Node::Constant(2), Node::Constant(3), add}).UpdateNodes();
        auto donor = Tree({Node::Constant(4), Node::Ref(0), add}).UpdateNodes();
        auto const rewritten = detail::RewriteSubtree(source.Nodes(), detail::DescribeSubtree(source.Nodes(), 0), donor.Nodes());
        auto const child = Tree(rewritten).UpdateNodes();
        CHECK(child.Nodes()[1].IsRef());
        CHECK(child.Nodes()[1].RefTo == 0);
        CHECK(child.Nodes().back().Length == 4);
    }

    SECTION("splice rebases self-contained Refs and rejects external Refs") {
        auto selfContained = Tree({Node::Constant(1), Node::Ref(0), add}).UpdateNodes();
        auto const spliced = selfContained.Splice(2);
        CHECK(spliced.Nodes()[1].RefTo == 0);

        auto external = Tree({Node::Constant(1), Node::Constant(2), add, Node::Ref(2), mul}).UpdateNodes();
        CHECK_THROWS_AS(external.Splice(3), std::invalid_argument);
    }

    SECTION("external donor Refs are rejected before P5") {
        auto source = Tree({Node::Constant(2), Node::Constant(3), add}).UpdateNodes();
        Operon::Vector<Node> donor{Node::Constant(4), Node::Ref(3), add};
        CHECK_THROWS_AS(detail::RewriteSubtree(source.Nodes(), detail::DescribeSubtree(source.Nodes(), 0), donor), std::invalid_argument);
    }
}

TEST_CASE("Crossover leaves parent unchanged for an external Ref donor", "[operators]")
{
    auto const add = Node::Function(Hash(BuiltinOp::Add), 2);
    auto const mul = Node::Function(Hash(BuiltinOp::Mul), 2);
    auto lhs = Tree({Node::Constant(2), Node::Constant(3), add}).UpdateNodes();
    auto rhs = Tree({Node::Constant(4), Node::Ref(0), Node::Constant(5), mul}).UpdateNodes();

    auto const child = CrossoverBase::Cross(lhs, rhs, 2, 1);
    CHECK(child.Nodes().size() == lhs.Nodes().size());
    CHECK(child.Nodes()[0].Value == lhs.Nodes()[0].Value);
    CHECK(child.Nodes().back().HashValue == lhs.Nodes().back().HashValue);
}

TEST_CASE("Crossover produces valid trees", "[operators]")
{
    auto ds = Operon::Dataset("./data/Poly-10.csv", true);
    auto variables = ds.GetVariables();
    std::vector<Operon::Hash> inputs;
    for (auto const& v : variables) {
        if (v.Name != "Y") { inputs.push_back(v.Hash); }
    }

    constexpr size_t maxDepth{1000};
    constexpr size_t maxLength{100};

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);
    BalancedTreeCreator const btc{&grammar, inputs, /* bias= */ 0.0, maxLength};

    Operon::RandomGenerator rng(1234);

    SECTION("Child is a valid tree") {
        constexpr double internalNodeProbability{0.9};
        Operon::SubtreeCrossover const cx(internalNodeProbability, maxDepth, maxLength);
        auto p1 = btc(rng, 7, 1, maxDepth); // NOLINT
        auto p2 = btc(rng, 5, 1, maxDepth); // NOLINT
        auto child = cx(rng, p1, p2);

        CHECK(child.Length() > 0);
    }

    SECTION("Child size is within bounds") {
        auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
        constexpr int n = 10000;
        std::vector<Tree> trees;
        trees.reserve(n);
        for (int i = 0; i < n; ++i) {
            trees.push_back(btc(rng, sizeDistribution(rng), 1UL, maxDepth));
        }

        std::uniform_int_distribution<size_t> dist(0, n - 1);
        Operon::SubtreeCrossover const cx(0.9, maxDepth, maxLength);

        for (int i = 0; i < 1000; ++i) {
            auto p1 = dist(rng);
            auto p2 = dist(rng);
            auto child = cx(rng, trees[p1], trees[p2]);
            CHECK(child.Length() <= maxLength);
        }
    }
}

} // namespace Operon::Test
