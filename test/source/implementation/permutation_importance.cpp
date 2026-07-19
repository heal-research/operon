// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"

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

TEST_CASE("PermutationImportance - range-less overload matches an explicit whole-dataset range", "[analyzers][permutation_importance]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Tree const tree = Tree({ nX0 }).UpdateNodes();

    Operon::RandomGenerator rngExplicit(1234);
    auto const importanceExplicit = Operon::PermutationImportance(tree, ds, target, Operon::Range(0, ds.Rows()), rngExplicit, /*nRepeats=*/10);

    Operon::RandomGenerator rngDefault(1234);
    auto const importanceDefault = Operon::PermutationImportance(tree, ds, target, rngDefault, /*nRepeats=*/10);

    REQUIRE(importanceExplicit.size() == importanceDefault.size());
    CHECK(importanceExplicit.front().Mean == importanceDefault.front().Mean);
    CHECK(importanceExplicit.front().Std == importanceDefault.front().Std);
}

TEST_CASE("PermutationImportance - a non-zero-start range aligns predictions against the matching target rows", "[analyzers][permutation_importance]")
{
    // Same 200 real rows as MakeLinearDataset, with two junk rows prepended
    // that only the [2, Rows()) range should ever see - if `actual` isn't
    // sliced to `range` the same way the (perturbed) predictions are, this
    // scores against the wrong rows and the two computations below diverge.
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    std::vector<Operon::Scalar> x0p{ 100, 100 };
    std::vector<Operon::Scalar> x1p{ 100, 100 };
    std::vector<Operon::Scalar> yp{ 100, 100 };
    auto appendCol = [](std::vector<Operon::Scalar>& dst, Operon::Span<Operon::Scalar const> src) -> void {
        dst.insert(dst.end(), src.begin(), src.end());
    };
    appendCol(x0p, ds.GetValues(x0.Hash));
    appendCol(x1p, ds.GetValues(ds.GetVariable("x1").value().Hash));
    appendCol(yp, ds.GetValues(target));
    Operon::Dataset const prefixed({ "x0", "x1", "y" }, { x0p, x1p, yp });

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Tree const tree = Tree({ nX0 }).UpdateNodes(); // y_hat = x0

    Operon::RandomGenerator rngFull(1234);
    auto const importanceFull = Operon::PermutationImportance(tree, ds, target, Operon::Range(0, ds.Rows()), rngFull, /*nRepeats=*/10);

    Operon::RandomGenerator rngPrefixed(1234);
    auto const importancePrefixed = Operon::PermutationImportance(tree, prefixed, target, Operon::Range(2, prefixed.Rows()), rngPrefixed, /*nRepeats=*/10);

    REQUIRE(importanceFull.size() == 1);
    REQUIRE(importancePrefixed.size() == 1);
    CHECK(std::abs(importanceFull.front().Mean - importancePrefixed.front().Mean) < 1e-6);
    CHECK(std::abs(importanceFull.front().Std - importancePrefixed.front().Std) < 1e-6);
}

TEST_CASE("PermutationImportance - duplicate variable occurrences collapse to one entry", "[analyzers][permutation_importance]")
{
    // y = x0 * x0 - x0 appears twice in the tree (two separate leaf nodes,
    // not shared via Ref), but importance is inherently per-*variable*, not
    // per-occurrence: shuffling the one underlying dataset column perturbs
    // both references simultaneously, so there should be exactly one entry,
    // not two.
    std::vector<Operon::Scalar> x0(200);
    std::vector<Operon::Scalar> y(200);
    Operon::RandomGenerator seedRng(7);
    std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
    for (size_t i = 0; i < x0.size(); ++i) {
        x0[i] = dist(seedRng);
        y[i] = x0[i] * x0[i];
    }
    Operon::Dataset ds({ "x0", "y" }, { x0, y });
    auto const x0Var = ds.GetVariable("x0").value();
    auto const target = ds.GetVariable("y").value().Hash;

    Node nX0a(NodeType::Variable); nX0a.HashValue = x0Var.Hash;
    Node nX0b(NodeType::Variable); nX0b.HashValue = x0Var.Hash;
    Tree const tree = Tree({ nX0a, nX0b, Util::MakeOp<BuiltinOp::Mul>() }).UpdateNodes();

    auto const range = Operon::Range(0, ds.Rows());
    Operon::RandomGenerator rng(1234);
    auto const importance = Operon::PermutationImportance(tree, ds, target, range, rng, /*nRepeats=*/10);

    REQUIRE(importance.size() == 1);
    CHECK(importance.front().Variable == x0Var.Hash);
    CHECK(importance.front().Mean > 0.5);
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
    Tree const tree = Tree({ nX0, nX1, nConstZero, Util::MakeOp<BuiltinOp::Mul>(), Util::MakeOp<BuiltinOp::Add>() }).UpdateNodes();

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

TEST_CASE("GradientImportance - range-less overload matches an explicit whole-dataset range", "[analyzers][gradient_importance]")
{
    auto ds = MakeLinearDataset();
    auto const x0 = ds.GetVariable("x0").value();

    Node nX0(NodeType::Variable); nX0.HashValue = x0.Hash;
    Tree const tree = Tree({ nX0 }).UpdateNodes();

    auto const importanceExplicit = Operon::GradientImportance(tree, ds, Operon::Range(0, ds.Rows()));
    auto const importanceDefault = Operon::GradientImportance(tree, ds);

    REQUIRE(importanceExplicit.size() == importanceDefault.size());
    CHECK(importanceExplicit.front().second == importanceDefault.front().second);
}

} // namespace Operon::Test
