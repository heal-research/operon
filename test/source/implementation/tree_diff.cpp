// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"

namespace Operon::Test {

namespace {
constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();
} // namespace

TEST_CASE("BuildJacobianDag - single constant", "[tree_diff]")
{
    // Tree: one optimizable constant c = 3.14. df/dc = 1.
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(3.14F);
    c.Optimize = true;
    nodes.push_back(c);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.OriginalSize == 1);
    REQUIRE(dag.Roots.size() == 1);
    REQUIRE(dag.Roots[0] != NoGrad);

    auto& derivNode = dag.Nodes[dag.Roots[0]];
    CHECK(derivNode.IsConstant());
    CHECK(derivNode.Value == Catch::Approx(1.0F));
}

TEST_CASE("BuildJacobianDag - variable leaf yields zero gradient", "[tree_diff]")
{
    // Variable has Optimize=true by default (IsLeaf()), but Deriv returns Zero for variables.
    Operon::Vector<Node> nodes;
    nodes.push_back(Node{NodeType::Variable});
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 1);
    // Variable derivative is zero (SIZE_MAX sentinel).
    CHECK(dag.Roots[0] == NoGrad);
}

TEST_CASE("BuildJacobianDag - no optimizable nodes means no roots", "[tree_diff]")
{
    // A constant with Optimize=false should not appear in Roots.
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(1.0F);
    c.Optimize = false;
    nodes.push_back(c);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);
    CHECK(dag.Roots.empty());
}

TEST_CASE("BuildJacobianDag - Add(c1, c2) both partials are 1", "[tree_diff]")
{
    // Tree: c1 + c2. df/dc1 = 1, df/dc2 = 1.
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node add{NodeType::Add}; add.Arity = 2; add.Length = 2;
    nodes.push_back(c1);
    nodes.push_back(c2);
    nodes.push_back(add);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 2);
    // Both partials are non-zero.
    CHECK(dag.Roots[0] != NoGrad);
    CHECK(dag.Roots[1] != NoGrad);
    // Hash-cons: both are the same Constant(1) node.
    CHECK(dag.Roots[0] == dag.Roots[1]);
    CHECK(dag.Nodes[dag.Roots[0]].IsConstant());
    CHECK(dag.Nodes[dag.Roots[0]].Value == Catch::Approx(1.0F));
}

TEST_CASE("BuildJacobianDag - Mul(c1, c2) product rule", "[tree_diff]")
{
    // Tree: c1 * c2. df/dc1 = c2, df/dc2 = c1.
    // With our implementation: each partial = Mul(Constant(1), Ref(other_c)).
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node mul{NodeType::Mul}; mul.Arity = 2; mul.Length = 2;
    nodes.push_back(c1);
    nodes.push_back(c2);
    nodes.push_back(mul);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 2);
    CHECK(dag.Roots[0] != NoGrad);
    CHECK(dag.Roots[1] != NoGrad);
    // The two partials reference different constants so they have distinct roots.
    CHECK(dag.Roots[0] != dag.Roots[1]);
    // Both root nodes should be Mul (from the product rule: Const(1) * Ref(other)).
    CHECK(dag.Nodes[dag.Roots[0]].Type == NodeType::Mul);
    CHECK(dag.Nodes[dag.Roots[1]].Type == NodeType::Mul);
}

TEST_CASE("BuildJacobianDag - Sin(c) gives Mul chain", "[tree_diff]")
{
    // Tree: sin(c). df/dc = cos(c) * 1.
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(1.0F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Sin}); // Arity=1, Length=1 by constructor
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 1);
    CHECK(dag.Roots[0] != NoGrad);
    // df/dc = fp * dj where fp = Cos(Ref(c)) and dj = Constant(1). Root is Mul.
    CHECK(dag.Nodes[dag.Roots[0]].Type == NodeType::Mul);
}

TEST_CASE("BuildJacobianDag - Tanh(c) derivative is 1 - result^2", "[tree_diff]")
{
    // Tree: tanh(c). df/dc = (1 - tanh(c)^2) * 1. Root is Mul.
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(0.5F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Tanh});
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 1);
    CHECK(dag.Roots[0] != NoGrad);
    CHECK(dag.Nodes[dag.Roots[0]].Type == NodeType::Mul);
}

TEST_CASE("BuildJacobianDag - original nodes are preserved", "[tree_diff]")
{
    // The first OriginalSize entries of dag.Nodes must be bit-for-bit identical to the input.
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(7.0F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Exp});
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.OriginalSize == tree.Length());
    for (std::size_t i = 0; i < dag.OriginalSize; ++i) {
        CHECK(dag.Nodes[i].Type == tree.Nodes()[i].Type);
        CHECK(dag.Nodes[i].Value == tree.Nodes()[i].Value);
    }
}

TEST_CASE("BuildJacobianDag - dag nodes never grow beyond uint16_t for small trees", "[tree_diff]")
{
    // RefTo is uint16_t; dag size must stay within bounds.
    Operon::Vector<Node> nodes;
    for (int k = 0; k < 4; ++k) {
        auto c = Node::Constant(static_cast<float>(k + 1));
        c.Optimize = true;
        nodes.push_back(c);
    }
    // Add4 = c1+c2+c3+c4
    Node add{NodeType::Add}; add.Arity = 4; add.Length = 4;
    nodes.push_back(add);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.Roots.size() == 4);
    // All 4 partials should be the same Constant(1) node (hash-consed).
    for (auto r : dag.Roots) {
        CHECK(r != NoGrad);
        CHECK(r == dag.Roots[0]);
    }
    CHECK(dag.Nodes.size() <= std::numeric_limits<uint16_t>::max());
}

} // namespace Operon::Test
