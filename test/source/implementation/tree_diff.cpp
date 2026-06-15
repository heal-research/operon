// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

namespace {

constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();

using DTable = DispatchTable<Operon::Scalar>;
using Interp = Interpreter<Operon::Scalar, DTable>;

// Evaluate all symbolic derivative columns in `dag` via the interpreter.
//
// For each root r = dag.Roots[k], builds a sub-tree covering dag.Nodes[0..r]
// (which always includes all original nodes plus derivative nodes up to r) and
// calls Interpreter::Evaluate. This is the naive single-column path; correctness
// takes priority over speed here. A NoGrad root produces a zero column.
auto EvalDagJacobian(
    JacobianDag const& dag,
    Operon::Span<Operon::Scalar const> coeff,
    Dataset const& ds,
    Range range,
    DTable const& dtable
) -> Eigen::Array<Operon::Scalar, -1, -1>
{
    auto const nRows  = static_cast<Eigen::Index>(range.Size());
    auto const nConst = static_cast<Eigen::Index>(dag.Roots.size());
    Eigen::Array<Operon::Scalar, -1, -1> jac(nRows, nConst);

    for (Eigen::Index k = 0; k < nConst; ++k) {
        auto const r = dag.Roots[static_cast<std::size_t>(k)];
        if (r == NoGrad) {
            jac.col(k).setZero();
            continue;
        }
        // Sub-tree covers dag.Nodes[0..r]; last node IS the derivative root.
        Operon::Vector<Node> subnodes(
            dag.Nodes.cbegin(),
            dag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(r) + 1
        );
        Tree t{std::move(subnodes)};
        Interp const interp{&dtable, &ds, &t};
        auto col = interp.Evaluate(coeff, range);
        jac.col(k) = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>>(col.data(), nRows);
    }
    return jac;
}

// Make a primitive set containing only ops our differentiator handles correctly.
// Aq / Powabs / Fmin / Fmax / Abs / Floor / Ceil / Sqrtabs are excluded
// (they return zero gradient from BuildJacobianDag).
auto MakeSupportedPset() -> PrimitiveSet {
    PrimitiveSet ps;
    ps.SetConfig(
        NodeType::Add | NodeType::Mul | NodeType::Sub | NodeType::Div |
        NodeType::Exp | NodeType::Log | NodeType::Logabs | NodeType::Log1p |
        NodeType::Sin | NodeType::Cos | NodeType::Tan  |
        NodeType::Asin | NodeType::Acos | NodeType::Atan |
        NodeType::Sinh | NodeType::Cosh | NodeType::Tanh |
        NodeType::Sqrt | NodeType::Cbrt | NodeType::Square |
        NodeType::Pow  |
        NodeType::Constant | NodeType::Variable
    );
    return ps;
}

// Generate `n` random trees using `pset`, lengths in [1, maxLen], with only
// Constant nodes marked Optimize=true and random leaf values in [-2, +2].
auto GenerateTrees(
    RandomGenerator& rng,
    PrimitiveSet& pset,
    Dataset const& ds,
    int n,
    std::size_t maxLen
) -> std::vector<Tree>
{
    std::uniform_real_distribution<Operon::Scalar> valDist(-2.F, +2.F);
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    BalancedTreeCreator const btc{&pset, ds.VariableHashes(), /*bias=*/0.0, maxLen};

    std::vector<Tree> trees;
    trees.reserve(n);
    for (int i = 0; i < n; ++i) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        for (auto& nd : tree.Nodes()) {
            nd.Optimize = nd.IsLeaf(); // both constants and variable weights are coefficients
            if (nd.IsLeaf()) { nd.Value = valDist(rng); }
        }
        trees.push_back(std::move(tree));
    }
    return trees;
}

} // namespace

// ============================================================
// Structural unit tests (small, hand-crafted)
// ============================================================

TEST_CASE("BuildJacobianDag - single constant", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(3.14F); c.Optimize = true;
    nodes.push_back(c);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);

    REQUIRE(dag.OriginalSize == 1);
    REQUIRE(dag.Roots.size() == 1);
    REQUIRE(dag.Roots[0] != NoGrad);

    auto& dn = dag.Nodes[dag.Roots[0]];
    CHECK(dn.IsConstant());
    CHECK(dn.Value == Catch::Approx(1.0F));
}

TEST_CASE("BuildJacobianDag - non-optimizable variable leaf yields zero gradient", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto v = Node{NodeType::Variable}; v.Optimize = false;
    nodes.push_back(v);
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    CHECK(dag.Roots.empty()); // no optimizable coefficients → no columns
}

TEST_CASE("BuildJacobianDag - optimizable variable leaf yields unweighted variable", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto v = Node{NodeType::Variable}; v.Optimize = true; v.Value = 2.5F;
    nodes.push_back(v);
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 1); // one coefficient
    REQUIRE(dag.Roots[0] != NoGrad);
    // d(w * X)/dw = X: the derivative node should be a Variable with Value=1 (unweighted)
    auto& dn = dag.Nodes[dag.Roots[0]];
    CHECK(dn.IsVariable());
    CHECK(dn.Value == Catch::Approx(1.0F));
    CHECK_FALSE(dn.Optimize);
}

TEST_CASE("BuildJacobianDag - no optimizable nodes means no roots", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(1.0F); c.Optimize = false;
    nodes.push_back(c);
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    CHECK(dag.Roots.empty());
}

TEST_CASE("BuildJacobianDag - Add(c1,c2) both partials are 1", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node add{NodeType::Add}; add.Arity = 2; add.Length = 2;
    nodes.push_back(c1); nodes.push_back(c2); nodes.push_back(add);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 2);
    CHECK(dag.Roots[0] != NoGrad);
    CHECK(dag.Roots[1] != NoGrad);
    // Hash-consed: both partials share the same Constant(1) node.
    CHECK(dag.Roots[0] == dag.Roots[1]);
    CHECK(dag.Nodes[dag.Roots[0]].IsConstant());
    CHECK(dag.Nodes[dag.Roots[0]].Value == Catch::Approx(1.0F));
}

TEST_CASE("BuildJacobianDag - Mul(c1,c2) product rule", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node mul{NodeType::Mul}; mul.Arity = 2; mul.Length = 2;
    nodes.push_back(c1); nodes.push_back(c2); nodes.push_back(mul);
    Tree tree{nodes};

    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 2);
    CHECK(dag.Roots[0] != NoGrad);
    CHECK(dag.Roots[1] != NoGrad);
    // Different constants → different derivative roots.
    CHECK(dag.Roots[0] != dag.Roots[1]);
    // Product rule emits Mul(Const(1), Ref(other)) for each column.
    CHECK(dag.Nodes[dag.Roots[0]].Type == NodeType::Mul);
    CHECK(dag.Nodes[dag.Roots[1]].Type == NodeType::Mul);
}

TEST_CASE("BuildJacobianDag - Sin(c) root is Mul", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(1.0F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Sin});
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 1);
    REQUIRE(dag.Roots[0] != NoGrad);
    CHECK(dag.Nodes[dag.Roots[0]].Type == NodeType::Mul);
}

TEST_CASE("BuildJacobianDag - original nodes are preserved exactly", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(7.0F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Exp});
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.OriginalSize == tree.Length());
    for (std::size_t i = 0; i < dag.OriginalSize; ++i) {
        CHECK(dag.Nodes[i].Type  == tree.Nodes()[i].Type);
        CHECK(dag.Nodes[i].Value == tree.Nodes()[i].Value);
    }
}

TEST_CASE("BuildJacobianDag - Add4 hash-cons collapses all partials", "[tree_diff]")
{
    Operon::Vector<Node> nodes;
    for (int k = 0; k < 4; ++k) {
        auto c = Node::Constant(static_cast<float>(k + 1)); c.Optimize = true;
        nodes.push_back(c);
    }
    Node add{NodeType::Add}; add.Arity = 4; add.Length = 4;
    nodes.push_back(add);
    Tree tree{nodes};
    auto dag = BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 4);
    for (auto r : dag.Roots) {
        CHECK(r != NoGrad);
        CHECK(r == dag.Roots[0]); // all four partials are the same Constant(1)
    }
}

// ============================================================
// Correctness test: compare BuildJacobianDag + EvalDagJacobian
// against JacRev over a large set of random trees.
// ============================================================

TEST_CASE("BuildJacobianDag correctness vs JacRev - random trees", "[tree_diff]")
{
    constexpr auto nRows  = 100;
    constexpr auto nCols  = 5;
    constexpr auto nTrees = 1000;
    constexpr auto maxLen = 30;
    constexpr auto eps    = 1e-3F; // relaxed for potential numerical differences
    constexpr auto maxDivergeRate = 0.02; // allow 2% divergence due to numerics

    Operon::RandomGenerator rng(42UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, ds.Rows<std::size_t>()};

    auto pset = MakeSupportedPset();
    auto const trees = GenerateTrees(rng, pset, ds, nTrees, maxLen);

    std::size_t finiteMismatch = 0; // jrev finite, dag not
    std::size_t finiteDiverge  = 0; // both finite but differ > eps
    std::size_t totalCols      = 0;

    for (auto const& tree : trees) {
        auto const coeff = tree.GetCoefficients();
        if (coeff.empty()) { continue; } // tree has no constants

        Interp const interp{&dtable, &ds, &tree};
        auto const jrev = interp.JacRev(coeff, range);

        auto const dag  = BuildJacobianDag(tree);
        auto const jdag = EvalDagJacobian(dag, coeff, ds, range, dtable);

        auto const nk = jrev.cols();
        totalCols += static_cast<std::size_t>(nk);

        for (Eigen::Index k = 0; k < nk; ++k) {
            auto const colRev = jrev.col(k);
            auto const colDag = jdag.col(k);
            bool const revFin = std::isfinite(colRev.sum());
            bool const dagFin = std::isfinite(colDag.sum());
            if (revFin && !dagFin) {
                ++finiteMismatch;
            } else if (revFin && dagFin && !colRev.isApprox(colDag, eps)) {
                ++finiteDiverge;
            }
        }
    }

    INFO("finite mismatch: " << finiteMismatch << " / " << totalCols);
    INFO("finite diverge:  " << finiteDiverge  << " / " << totalCols);
    CHECK(finiteMismatch == 0);
    CHECK(static_cast<double>(finiteDiverge) / static_cast<double>(std::max(totalCols, 1UL)) < maxDivergeRate);
}

// ============================================================
// Performance test: BuildJacobianDag vs JacRev
// ============================================================

TEST_CASE("BuildJacobianDag performance vs JacRev", "[tree_diff][performance]")
{
    constexpr auto nRows = 1000;
    constexpr auto nCols = 10;
    constexpr auto nTrees = 500;
    constexpr auto maxLen = 100;

    Operon::RandomGenerator rng(0UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, ds.Rows<std::size_t>()};

    auto pset = MakeSupportedPset();
    auto const trees = GenerateTrees(rng, pset, ds, nTrees, maxLen);

    // Pre-build all dags outside the benchmark loops.
    std::vector<JacobianDag> dags;
    dags.reserve(trees.size());
    for (auto const& tree : trees) { dags.push_back(BuildJacobianDag(tree)); }

    // Pre-collect coefficients.
    std::vector<std::vector<Operon::Scalar>> coeffs;
    coeffs.reserve(trees.size());
    for (auto const& tree : trees) { coeffs.push_back(tree.GetCoefficients()); }

    nb::Bench bench;
    bench.timeUnit(std::chrono::milliseconds(1), "ms");
    bench.relative(true); // first case is 100%; subsequent are relative

    // ---- JacRev (baseline) ----
    bench.run("JacRev", [&]() {
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& tree  = trees[i];
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            nb::doNotOptimizeAway(Interp{&dtable, &ds, &tree}.JacRev(coeff, range));
        }
    });

    // ---- BuildJacobianDag only (construction cost) ----
    bench.run("BuildJacobianDag", [&]() {
        for (auto const& tree : trees) {
            nb::doNotOptimizeAway(BuildJacobianDag(tree));
        }
    });

    // ---- BuildJacobianDag + EvalDagJacobian (full pipeline) ----
    bench.run("BuildDag+EvalDag", [&]() {
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            auto dag = BuildJacobianDag(trees[i]);
            nb::doNotOptimizeAway(EvalDagJacobian(dag, coeff, ds, range, dtable));
        }
    });

    // ---- EvalDagJacobian only (pre-built dag, evaluation cost) ----
    bench.run("EvalDag (prebuilt)", [&]() {
        for (std::size_t i = 0; i < dags.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            nb::doNotOptimizeAway(EvalDagJacobian(dags[i], coeff, ds, range, dtable));
        }
    });

    bench.render(nb::templates::csv(), std::cout);
}

} // namespace Operon::Test
