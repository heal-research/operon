// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/analyzers/node_impact.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"

namespace Operon::Test {

namespace {
    auto MakeLinearDataset() -> Operon::Dataset {
        // y = x0 + x1, exactly - x0's coefficient node contributes, the
        // Constant{0} node added on top does not.
        std::vector<Operon::Scalar> x0{ 1, 2, 3, 4, 5, 6, 7, 8 };
        std::vector<Operon::Scalar> x1{ 2, 3, 4, 5, 6, 7, 8, 9 };
        std::vector<Operon::Scalar> y(x0.size());
        for (size_t i = 0; i < x0.size(); ++i) { y[i] = x0[i] + x1[i]; }
        return Operon::Dataset({ "x0", "x1", "y" }, { x0, x1, y });
    }
} // namespace

TEST_CASE("NodeImpact - irrelevant subtree has ~zero impact, relevant subtree does not", "[analyzers][node_impact]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const x1 = ds.GetVariable("x1").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Node nX1(NodeType::Variable); nX1.HashValue = x1.Hash;

    // (x0 + x1) + 0 - the "+ 0" wraps the real model in a no-op addition of
    // an irrelevant constant, so the root's impact should be indistinguishable
    // from the (x0 + x1) subtree's impact, while the Constant{0} leaf's own
    // impact should be ~zero (replacing "0" by its own mean, "0", is a no-op).
    Tree const tree = Tree({ nX0, nX1, Node(NodeType::Add), Node::Constant(0.0), Node(NodeType::Add) }).UpdateNodes();

    auto const range = Operon::Range(0, ds.Rows());
    auto const impact = Operon::NodeImpact(tree, ds, target, range);

    REQUIRE(impact.size() == tree.Length());

    auto const constantZeroIdx = 3UL; // Constant(0) node, see the initializer list above
    auto const addX0X1Idx = 2UL;      // (x0 + x1) node
    auto const rootIdx = 4UL;         // (x0 + x1) + 0

    CHECK(std::abs(impact[constantZeroIdx]) < 1e-6);
    CHECK(impact[addX0X1Idx] > 0.5);
    CHECK(std::abs(impact[rootIdx] - impact[addX0X1Idx]) < 1e-6);
}

TEST_CASE("NodeImpact - a non-zero-start range aligns predictions against the matching target rows", "[analyzers][node_impact]")
{
    // Same (x0, x1, y) relationship as the linear dataset above, but with two
    // junk rows prepended that only the [2, Rows()) range should ever see -
    // if `actual` isn't sliced to `range` the same way `predicted` is, this
    // scores against the wrong rows and the two computations below diverge.
    std::vector<Operon::Scalar> x0{ 100, 100, 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<Operon::Scalar> x1{ 100, 100, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<Operon::Scalar> y(x0.size());
    for (size_t i = 0; i < x0.size(); ++i) { y[i] = x0[i] + x1[i]; }
    Operon::Dataset const prefixed({ "x0", "x1", "y" }, { x0, x1, y });

    auto ds = MakeLinearDataset(); // the same 8 real rows, with no junk prefix
    auto const x0Var = ds.GetVariable("x0").value();
    auto const x1Var = ds.GetVariable("x1").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0(NodeType::Variable); nX0.HashValue = x0Var.Hash;
    Node nX1(NodeType::Variable); nX1.HashValue = x1Var.Hash;
    Tree const tree = Tree({ nX0, nX1, Node(NodeType::Add) }).UpdateNodes();

    auto const impactFull = Operon::NodeImpact(tree, ds, target, Operon::Range(0, ds.Rows()));
    auto const impactPrefixed = Operon::NodeImpact(tree, prefixed, target, Operon::Range(2, prefixed.Rows()));

    REQUIRE(impactFull.size() == impactPrefixed.size());
    for (size_t i = 0; i < impactFull.size(); ++i) {
        CHECK(std::abs(impactFull[i] - impactPrefixed[i]) < 1e-6);
    }
}

TEST_CASE("NodeImpact - duplicate variable occurrences are scored independently", "[analyzers][node_impact]")
{
    // y = x0 * x0 - x0 appears twice as two separate leaf nodes (not shared
    // via Ref), so each occurrence is its own postfix index and should get
    // its own (here: equal, by symmetry) impact rather than one shared value.
    std::vector<Operon::Scalar> x0{ 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<Operon::Scalar> y(x0.size());
    for (size_t i = 0; i < x0.size(); ++i) { y[i] = x0[i] * x0[i]; }
    Operon::Dataset ds({ "x0", "y" }, { x0, y });
    auto const x0Var = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0a(NodeType::Variable); nX0a.HashValue = x0Var.Hash;
    Node nX0b(NodeType::Variable); nX0b.HashValue = x0Var.Hash;
    Tree const tree = Tree({ nX0a, nX0b, Node(NodeType::Mul) }).UpdateNodes();

    auto const range = Operon::Range(0, ds.Rows());
    auto const impact = Operon::NodeImpact(tree, ds, target, range);

    REQUIRE(impact.size() == 3);
    CHECK(impact[0] > 0.2);  // replacing either leaf occurrence with mean(x0) should hurt R2 ...
    CHECK(impact[1] > 0.2);
    CHECK(std::abs(impact[0] - impact[1]) < 1e-6); // ... and by symmetry, identically
}

TEST_CASE("NodeImpact - trees with Ref nodes are unsupported", "[analyzers][node_impact]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;

    // x0 * Ref(x0) - structural sharing via a backward Ref.
    Tree const tree = Tree({ nX0, Node::Ref(0), Node(NodeType::Mul) }).UpdateNodes();

    auto const range = Operon::Range(0, ds.Rows());
    auto const impact = Operon::NodeImpact(tree, ds, target, range);

    CHECK(impact.empty());
}

} // namespace Operon::Test
