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
