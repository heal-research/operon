// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/analyzers/gradient_importance.hpp"
#include "operon/analyzers/permutation_importance.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"

namespace Operon::Test {

namespace {
    auto MakeLinearDataset() -> Operon::Dataset {
        // y = x0 + 0*x1: x0 matters, x1 is dataset noise the model ignores.
        Operon::RandomGenerator rng(42);
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
        std::vector<Operon::Scalar> x0(200);
        std::vector<Operon::Scalar> x1(200);
        std::vector<Operon::Scalar> y(200);
        for (size_t i = 0; i < x0.size(); ++i) {
            x0[i] = dist(rng);
            x1[i] = dist(rng);
            y[i] = x0[i];
        }
        return Operon::Dataset({ "x0", "x1", "y" }, { x0, x1, y });
    }
} // namespace

TEST_CASE("PermutationImportance - relevant variable outranks unused one", "[analyzers][permutation_importance]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Tree const tree = Tree({ nX0 }).UpdateNodes(); // y_hat = x0

    auto const range = Operon::Range(0, ds.Rows());
    Operon::RandomGenerator rng(1234);
    auto const importance = Operon::PermutationImportance(tree, ds, target, range, rng, /*nRepeats=*/10);

    // Only x0 appears in the tree, so only x0 gets an importance entry.
    REQUIRE(importance.size() == 1);
    CHECK(importance.front().Variable == x0.Hash);
    CHECK(importance.front().Mean > 0.5); // shuffling the one variable the model uses should tank R2
}

TEST_CASE("GradientImportance - relevant variable outranks unused one", "[analyzers][gradient_importance]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const x1 = ds.GetVariable("x1").value();

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Node nX1(NodeType::Variable); nX1.HashValue = x1.Hash;
    Node nConstZero = Node::Constant(0.0);
    // y_hat = x0 + 0*x1 - x1 appears in the tree (so it gets scored) but
    // contributes nothing, unlike the shuffle-based test above where an unused
    // variable would simply be absent from the result.
    Tree const tree = Tree({ nX0, nX1, nConstZero, Node(NodeType::Mul), Node(NodeType::Add) }).UpdateNodes();

    auto const range = Operon::Range(0, ds.Rows());
    auto const importance = Operon::GradientImportance(tree, ds, range);

    REQUIRE(importance.size() == 2);
    auto const x0It = std::ranges::find(importance, x0.Hash, [](auto const& p) -> Operon::Hash { return p.first; });
    auto const x1It = std::ranges::find(importance, x1.Hash, [](auto const& p) -> Operon::Hash { return p.first; });
    REQUIRE(x0It != importance.end());
    REQUIRE(x1It != importance.end());

    CHECK(x0It->second > 0.9);  // d(x0)/d(x0) = 1 exactly
    CHECK(x1It->second < 1e-6); // d(0*x1)/d(x1) = 0 exactly
}

} // namespace Operon::Test
