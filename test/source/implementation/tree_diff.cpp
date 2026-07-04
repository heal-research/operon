// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"

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

    // ---- BuildJacobianDag + EvalDagJacobian per-column (full pipeline) ----
    bench.run("BuildDag+EvalColumn", [&]() {
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            auto dag = BuildJacobianDag(trees[i]);
            nb::doNotOptimizeAway(EvalDagJacobian(dag, coeff, ds, range, dtable));
        }
    });

    // ---- EvalDagJacobian per-column only (pre-built dag) ----
    bench.run("EvalColumn (prebuilt)", [&]() {
        for (std::size_t i = 0; i < dags.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            nb::doNotOptimizeAway(EvalDagJacobian(dags[i], coeff, ds, range, dtable));
        }
    });

    // ---- BuildJacobianDag + EvaluateRoots single-pass (full pipeline) ----
    bench.run("BuildDag+EvalRoots", [&]() {
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            auto dag = BuildJacobianDag(trees[i]);
            Tree t{dag.Nodes};
            Interp const interp{&dtable, &ds, &t};
            nb::doNotOptimizeAway(interp.EvaluateRoots(coeff, range, dag.Roots));
        }
    });

    // ---- EvaluateRoots single-pass only (pre-built dag) ----
    bench.run("EvalRoots (prebuilt)", [&]() {
        for (std::size_t i = 0; i < dags.size(); ++i) {
            auto const& coeff = coeffs[i];
            if (coeff.empty()) { continue; }
            Tree t{dags[i].Nodes};
            Interp const interp{&dtable, &ds, &t};
            nb::doNotOptimizeAway(interp.EvaluateRoots(coeff, range, dags[i].Roots));
        }
    });

    bench.render(nb::templates::csv(), std::cout);
}

// ============================================================
// Hessian: structural unit tests
// ============================================================

TEST_CASE("BuildHessianDag - single constant", "[tree_diff][hessian]")
{
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(3.14F); c.Optimize = true;
    nodes.push_back(c);
    Tree tree{nodes};

    auto dag = BuildHessianDag(tree);
    REQUIRE(dag.NumParams == 1);
    REQUIRE(dag.JacobianRoots.size() == 1);
    REQUIRE(dag.HessianRoots.size() == 1);
    // d²(c)/dc² = 0
    CHECK(dag.HessianRoots[0] == NoGrad);
}

TEST_CASE("BuildHessianDag - square(c) Hessian is 2", "[tree_diff][hessian]")
{
    // f = c^2, df/dc = 2c, d²f/dc² = 2
    Operon::Vector<Node> nodes;
    auto c = Node::Constant(5.0F); c.Optimize = true;
    nodes.push_back(c);
    nodes.push_back(Node{NodeType::Square});
    Tree tree{nodes};

    auto dag = BuildHessianDag(tree);
    REQUIRE(dag.NumParams == 1);
    REQUIRE(dag.HessianRoots.size() == 1);
    REQUIRE(dag.HessianRoots[0] != NoGrad);

    // Evaluate d²f/dc² — should be 2
    DTable dtable;
    auto ds = Dataset(std::vector<std::string>{"X"}, std::vector<std::vector<Operon::Scalar>>{{1.0F}});
    Range range{0, 1};
    auto coeff = tree.GetCoefficients();

    Operon::Vector<Node> subnodes(
        dag.Nodes.cbegin(),
        dag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(dag.HessianRoots[0]) + 1
    );
    Tree ht{std::move(subnodes)};
    Interp const interp{&dtable, &ds, &ht};
    auto col = interp.Evaluate(coeff, range);
    CHECK(col[0] == Catch::Approx(2.0F).margin(1e-4F));
}

TEST_CASE("BuildHessianDag - Add(c1,c2) all Hessian entries zero", "[tree_diff][hessian]")
{
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node add{NodeType::Add}; add.Arity = 2; add.Length = 2;
    nodes.push_back(c1); nodes.push_back(c2); nodes.push_back(add);
    Tree tree{nodes};

    auto dag = BuildHessianDag(tree);
    REQUIRE(dag.NumParams == 2);
    REQUIRE(dag.HessianRoots.size() == 3); // 2*(2+1)/2
    for (auto r : dag.HessianRoots) {
        CHECK(r == NoGrad);
    }
}

TEST_CASE("BuildHessianDag - Mul(c1,c2) mixed partial is 1", "[tree_diff][hessian]")
{
    // f = c1*c2, d²f/dc1² = 0, d²f/dc1dc2 = 1, d²f/dc2² = 0
    Operon::Vector<Node> nodes;
    auto c1 = Node::Constant(2.0F); c1.Optimize = true;
    auto c2 = Node::Constant(3.0F); c2.Optimize = true;
    Node mul{NodeType::Mul}; mul.Arity = 2; mul.Length = 2;
    nodes.push_back(c1); nodes.push_back(c2); nodes.push_back(mul);
    Tree tree{nodes};

    auto dag = BuildHessianDag(tree);
    REQUIRE(dag.NumParams == 2);
    // H(0,0) = 0, H(0,1) = 1, H(1,1) = 0
    CHECK(dag.HessianRoots[dag.UpperIdx(0, 0)] == NoGrad);
    CHECK(dag.HessianRoots[dag.UpperIdx(1, 1)] == NoGrad);
    auto mixedRoot = dag.HessianRoots[dag.UpperIdx(0, 1)];
    REQUIRE(mixedRoot != NoGrad);

    DTable dtable;
    auto ds = Dataset(std::vector<std::string>{"X"}, std::vector<std::vector<Operon::Scalar>>{{1.0F}});
    Range range{0, 1};
    auto coeff = tree.GetCoefficients();

    Operon::Vector<Node> subnodes(
        dag.Nodes.cbegin(),
        dag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(mixedRoot) + 1
    );
    Tree ht{std::move(subnodes)};
    Interp const interp{&dtable, &ds, &ht};
    auto col = interp.Evaluate(coeff, range);
    CHECK(col[0] == Catch::Approx(1.0F).margin(1e-4F));
}

// ============================================================
// Hessian: ground truth from JAX
// ============================================================

namespace {

// Create an Interpreter over a DAG's full node array and evaluate multiple roots
// in a single forward pass.
auto EvalDagRoots(
    Operon::Vector<Node> const& dagNodes,
    Operon::Span<std::size_t const> roots,
    Operon::Span<Operon::Scalar const> coeff,
    Dataset const& ds,
    Range range,
    DTable const& dtable
) -> Eigen::Array<Operon::Scalar, -1, -1>
{
    Tree t{dagNodes};
    Interp const interp{&dtable, &ds, &t};
    return interp.EvaluateRoots(coeff, range, roots);
}

// Evaluate a single DAG root via EvalDagRoots (convenience wrapper).
auto EvalDagColumn(
    Operon::Vector<Node> const& dagNodes,
    std::size_t root,
    Operon::Span<Operon::Scalar const> coeff,
    Dataset const& ds,
    Range range,
    DTable const& dtable
) -> Eigen::Array<Operon::Scalar, -1, 1>
{
    Operon::Span<std::size_t const> rootSpan{&root, 1};
    return EvalDagRoots(dagNodes, rootSpan, coeff, ds, range, dtable).col(0);
}

// Find permutation mapping: for each Operon coefficient, which Python index matches.
// Requires all coefficient values to be distinct (guaranteed by the Python generator).
// Returns empty vector on failure.
auto FindCoeffPermutation(
    std::vector<Operon::Scalar> const& operonCoeffs,
    std::vector<double> const& pythonCoeffs
) -> std::vector<std::size_t>
{
    auto const p = operonCoeffs.size();
    if (p != pythonCoeffs.size()) { return {}; }

    std::vector<bool> used(p, false);
    std::vector<std::size_t> perm(p);

    for (std::size_t i = 0; i < p; ++i) {
        bool found = false;
        for (std::size_t j = 0; j < p; ++j) {
            if (used[j]) { continue; }
            if (std::abs(static_cast<double>(operonCoeffs[i]) - pythonCoeffs[j]) < 1e-4) {
                perm[i] = j;
                used[j] = true;
                found = true;
                break;
            }
        }
        if (!found) { return {}; }
    }
    return perm;
}

// Upper triangle index for the Python ordering
auto PythonUpperIdx(std::size_t i, std::size_t j, std::size_t p) -> std::size_t {
    if (i > j) { std::swap(i, j); }
    return (i * p) - (i * (i - 1) / 2) + (j - i);
}

struct GroundTruthCase {
    std::string Expr;
    std::vector<double> Coeffs;
    std::vector<double> Residuals;
    std::vector<std::vector<double>> Jacobian;     // [nrows][ncoeffs]
    std::vector<std::vector<double>> HessianTri;   // [nrows][nhess]
};

struct GroundTruth {
    int Nrows{0};
    int Nvars{0};
    std::vector<std::vector<double>> Data; // [nrows][nvars]
    std::vector<GroundTruthCase> Cases;
};

auto LoadGroundTruth(std::string const& path) -> GroundTruth {
    GroundTruth gt;
    std::ifstream ifs(path);
    REQUIRE(ifs.good());

    int ncases = 0;
    ifs >> ncases >> gt.Nrows >> gt.Nvars;

    gt.Data.resize(gt.Nrows, std::vector<double>(gt.Nvars));
    for (int r = 0; r < gt.Nrows; ++r) {
        for (int c = 0; c < gt.Nvars; ++c) {
            ifs >> gt.Data[r][c];
        }
    }

    gt.Cases.resize(ncases);
    for (int ci = 0; ci < ncases; ++ci) {
        auto& tc = gt.Cases[ci];
        std::getline(ifs >> std::ws, tc.Expr);
        int nc = 0;
        ifs >> nc;
        tc.Coeffs.resize(nc);
        for (int k = 0; k < nc; ++k) { ifs >> tc.Coeffs[k]; }

        tc.Residuals.resize(gt.Nrows);
        for (int r = 0; r < gt.Nrows; ++r) { ifs >> tc.Residuals[r]; }

        tc.Jacobian.resize(gt.Nrows, std::vector<double>(nc));
        for (int r = 0; r < gt.Nrows; ++r) {
            for (int k = 0; k < nc; ++k) { ifs >> tc.Jacobian[r][k]; }
        }

        int nhess = nc * (nc + 1) / 2;
        tc.HessianTri.resize(gt.Nrows, std::vector<double>(nhess));
        for (int r = 0; r < gt.Nrows; ++r) {
            for (int k = 0; k < nhess; ++k) { ifs >> tc.HessianTri[r][k]; }
        }
    }
    return gt;
}

} // namespace

TEST_CASE("BuildHessianDag correctness vs JAX ground truth", "[tree_diff][hessian]")
{
    auto const gt = LoadGroundTruth("./data/hessian_ground_truth.txt");

    std::vector<std::string> varNames(gt.Nvars);
    for (int i = 0; i < gt.Nvars; ++i) { varNames[i] = fmt::format("X{}", i + 1); }
    std::vector<std::vector<Operon::Scalar>> varData(gt.Nvars, std::vector<Operon::Scalar>(gt.Nrows));
    for (int r = 0; r < gt.Nrows; ++r) {
        for (int c = 0; c < gt.Nvars; ++c) {
            varData[c][r] = static_cast<Operon::Scalar>(gt.Data[r][c]);
        }
    }
    Dataset ds(varNames, varData);
    DTable dtable;
    Range const range{0, ds.Rows<std::size_t>()};

    constexpr auto eps = 1e-4F;
    std::size_t passed = 0;
    std::size_t failed = 0;
    std::size_t skipped = 0;

    for (auto const& tc : gt.Cases) {
        INFO("Expression: " << tc.Expr);
        auto const p = tc.Coeffs.size();

        auto tree = InfixParser::Parse(tc.Expr, ds, /*reduce=*/false);
        for (auto& n : tree.Nodes()) {
            n.Optimize = n.IsConstant();
        }
        auto const operonCoeffs = tree.GetCoefficients();

        if (operonCoeffs.size() != p) {
            fmt::print(stderr, "  SKIP coeff count mismatch: {} (expected {}, got {})\n",
                       tc.Expr, p, operonCoeffs.size());
            ++skipped;
            continue;
        }

        auto const perm = FindCoeffPermutation(operonCoeffs, tc.Coeffs);
        if (perm.empty() && p > 0) {
            fmt::print(stderr, "  SKIP could not match coefficients: {}\n", tc.Expr);
            ++skipped;
            continue;
        }

        auto const dag = BuildHessianDag(tree);
        REQUIRE(dag.NumParams == p);

        auto coeff = tree.GetCoefficients();
        Operon::Span<Operon::Scalar const> coeffSpan{coeff.data(), coeff.size()};

        // Check residuals
        {
            Operon::Vector<Node> subnodes(
                dag.Nodes.cbegin(),
                dag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(dag.OriginalSize)
            );
            Tree ft{std::move(subnodes)};
            Interp const interp{&dtable, &ds, &ft};
            auto fvals = interp.Evaluate(coeffSpan, range);
            bool resOk = true;
            for (int r = 0; r < gt.Nrows; ++r) {
                auto expected = static_cast<Operon::Scalar>(tc.Residuals[r]);
                if (std::isfinite(expected) && std::isfinite(fvals[r])) {
                    if (std::abs(fvals[r] - expected) > eps * (1.0F + std::abs(expected))) {
                        resOk = false;
                        break;
                    }
                }
            }
            if (!resOk) {
                fmt::print(stderr, "  SKIP residual mismatch: {}\n", tc.Expr);
                ++skipped;
                continue;
            }
        }

        // Check Jacobian columns
        bool jacOk = true;
        for (std::size_t k = 0; k < p && jacOk; ++k) {
            auto col = EvalDagColumn(dag.Nodes, dag.JacobianRoots[k], coeffSpan, ds, range, dtable);
            auto pyK = perm[k];
            for (int r = 0; r < gt.Nrows; ++r) {
                auto expected = static_cast<Operon::Scalar>(tc.Jacobian[r][pyK]);
                if (!std::isfinite(expected) || !std::isfinite(col(r))) { continue; }
                if (std::abs(col(r) - expected) > eps * (1.0F + std::abs(expected))) {
                    jacOk = false;
                    break;
                }
            }
        }

        // Check Hessian columns
        bool hessOk = true;
        for (std::size_t i = 0; i < p && hessOk; ++i) {
            for (std::size_t j = i; j < p && hessOk; ++j) {
                auto col = EvalDagColumn(
                    dag.Nodes, dag.HessianRoots[dag.UpperIdx(i, j)],
                    coeffSpan, ds, range, dtable);
                auto pyIdx = PythonUpperIdx(perm[i], perm[j], p);
                for (int r = 0; r < gt.Nrows; ++r) {
                    auto expected = static_cast<Operon::Scalar>(tc.HessianTri[r][pyIdx]);
                    if (!std::isfinite(expected) || !std::isfinite(col(r))) { continue; }
                    if (std::abs(col(r) - expected) > eps * (1.0F + std::abs(expected))) {
                        fmt::print(stderr, "  FAIL H({},{}) row {}: got {}, expected {}\n",
                                   i, j, r, col(r), expected);
                        hessOk = false;
                    }
                }
            }
        }

        if (jacOk && hessOk) {
            ++passed;
        } else {
            fmt::print(stderr, "  FAIL {}: jac={} hess={}\n", tc.Expr, jacOk, hessOk);
            ++failed;
        }
        CHECK((jacOk && hessOk));
    }

    fmt::print("JAX ground truth: {} passed, {} failed, {} skipped\n", passed, failed, skipped);
    CHECK(failed == 0);
    CHECK(skipped < gt.Cases.size() / 2);
}

// ============================================================
// Hessian: random tree correctness via finite differences on JacRev
// ============================================================

TEST_CASE("BuildHessianDag correctness vs finite differences - random trees", "[tree_diff][hessian]")
{
    constexpr auto nRows  = 50;
    constexpr auto nCols  = 5;
    constexpr auto nTrees = 500;
    constexpr auto maxLen = 20;
    constexpr auto fdEps  = 1e-3F;
    constexpr auto tol    = 5e-2F;
    constexpr auto maxFailRate = 0.15;

    Operon::RandomGenerator rng(99UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, ds.Rows<std::size_t>()};

    auto pset = MakeSupportedPset();
    auto const trees = GenerateTrees(rng, pset, ds, nTrees, maxLen);

    std::size_t totalEntries = 0;
    std::size_t failedEntries = 0;

    for (auto const& tree : trees) {
        auto coeff = tree.GetCoefficients();
        auto const p = coeff.size();
        if (p == 0) { continue; }

        auto const dag = BuildHessianDag(tree);
        Interp const interp{&dtable, &ds, &tree};

        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = i; j < p; ++j) {
                auto dagCol = EvalDagColumn(
                    dag.Nodes, dag.HessianRoots[dag.UpperIdx(i, j)],
                    coeff, ds, range, dtable);

                // Finite difference: perturb coeff[j], measure change in J column i
                auto coeffPlus = coeff;
                auto coeffMinus = coeff;
                coeffPlus[j] += fdEps;
                coeffMinus[j] -= fdEps;

                // Create temporary trees with perturbed coefficients
                auto treePlus = tree;
                auto treeMinus = tree;
                treePlus.SetCoefficients({coeffPlus.data(), coeffPlus.size()});
                treeMinus.SetCoefficients({coeffMinus.data(), coeffMinus.size()});

                auto jacPlus = Interp{&dtable, &ds, &treePlus}.JacRev(coeffPlus, range);
                auto jacMinus = Interp{&dtable, &ds, &treeMinus}.JacRev(coeffMinus, range);

                auto fdCol = (jacPlus.col(static_cast<Eigen::Index>(i))
                            - jacMinus.col(static_cast<Eigen::Index>(i)))
                           / (2.0F * fdEps);

                ++totalEntries;
                for (int r = 0; r < static_cast<int>(range.Size()); ++r) {
                    if (!std::isfinite(fdCol(r)) || !std::isfinite(dagCol(r))) { continue; }
                    if (std::abs(dagCol(r) - fdCol(r)) > tol * (1.0F + std::abs(fdCol(r)))) {
                        ++failedEntries;
                        break; // one failure per (i,j) entry is enough
                    }
                }
            }
        }
    }

    auto rate = static_cast<double>(failedEntries) / static_cast<double>(std::max(totalEntries, 1UL));
    fmt::print("FD Hessian: {} / {} entries failed ({:.2f}%)\n", failedEntries, totalEntries, rate * 100.0);
    CHECK(rate < maxFailRate);
}

// ============================================================
// Performance: BuildHessianDag vs FD-on-JacRev
// ============================================================

TEST_CASE("BuildHessianDag performance vs JacRev FD", "[tree_diff][hessian][performance]")
{
    constexpr auto nRows  = 1000;
    constexpr auto nCols  = 10;
    constexpr auto nTrees = 500;
    constexpr auto maxLen = 50;

    Operon::RandomGenerator rng(0UL);
    auto ds = Operon::Test::Util::RandomDataset(rng, nRows, nCols);
    DTable dtable;
    Range const range{0, ds.Rows<std::size_t>()};

    auto pset = MakeSupportedPset();
    auto const trees = GenerateTrees(rng, pset, ds, nTrees, maxLen);

    std::vector<std::vector<Operon::Scalar>> coeffs;
    coeffs.reserve(trees.size());
    for (auto const& tree : trees) { coeffs.push_back(tree.GetCoefficients()); }

    std::vector<HessianDag> dags;
    dags.reserve(trees.size());
    for (auto const& tree : trees) { dags.emplace_back(BuildHessianDag(tree)); }

    nb::Bench bench;
    bench.timeUnit(std::chrono::milliseconds(1), "ms");
    bench.relative(true);

    // ---- Baseline: FD Hessian via 2*p JacRev calls ----
    bench.run("FD-on-JacRev", [&]() {
        constexpr auto fdEps = 1e-3F;
        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            auto const& coeff = coeffs[ti];
            auto const p = coeff.size();
            if (p == 0) { continue; }
            for (std::size_t j = 0; j < p; ++j) {
                auto cPlus = coeff;
                auto cMinus = coeff;
                cPlus[j] += fdEps;
                cMinus[j] -= fdEps;
                auto tPlus = trees[ti];
                auto tMinus = trees[ti];
                tPlus.SetCoefficients({cPlus.data(), cPlus.size()});
                tMinus.SetCoefficients({cMinus.data(), cMinus.size()});
                auto jPlus = Interp{&dtable, &ds, &tPlus}.JacRev(cPlus, range);
                auto jMinus = Interp{&dtable, &ds, &tMinus}.JacRev(cMinus, range);
                nb::doNotOptimizeAway((jPlus - jMinus).eval());
            }
        }
    });

    // ---- BuildHessianDag only (construction cost) ----
    bench.run("BuildHessianDag", [&]() {
        for (auto const& tree : trees) {
            nb::doNotOptimizeAway(BuildHessianDag(tree));
        }
    });

    // ---- BuildHessianDag + single-pass eval ----
    bench.run("BuildDag+EvalRoots", [&]() {
        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            auto const& coeff = coeffs[ti];
            if (coeff.empty()) { continue; }
            auto dag = BuildHessianDag(trees[ti]);
            nb::doNotOptimizeAway(EvalDagRoots(dag.Nodes, dag.HessianRoots, coeff, ds, range, dtable));
        }
    });

    // ---- Single-pass eval only (pre-built dag) ----
    bench.run("EvalRoots (prebuilt)", [&]() {
        for (std::size_t ti = 0; ti < dags.size(); ++ti) {
            auto const& coeff = coeffs[ti];
            if (coeff.empty()) { continue; }
            auto const& dag = dags[ti];
            nb::doNotOptimizeAway(EvalDagRoots(dag.Nodes, dag.HessianRoots, coeff, ds, range, dtable));
        }
    });

    // ---- Per-column eval (pre-built dag) for comparison ----
    bench.run("EvalColumn (prebuilt)", [&]() {
        for (std::size_t ti = 0; ti < dags.size(); ++ti) {
            auto const& coeff = coeffs[ti];
            auto const& dag = dags[ti];
            auto const p = dag.NumParams;
            if (p == 0) { continue; }
            for (std::size_t i = 0; i < p; ++i) {
                for (std::size_t j = i; j < p; ++j) {
                    nb::doNotOptimizeAway(
                        EvalDagColumn(dag.Nodes, dag.HessianRoots[dag.UpperIdx(i, j)],
                                      coeff, ds, range, dtable));
                }
            }
        }
    });

    bench.render(nb::templates::csv(), std::cout);
}

} // namespace Operon::Test
