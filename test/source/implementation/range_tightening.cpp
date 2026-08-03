// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <random>
#include <vector>

#include "../operon_test.hpp"

#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/range_tightening.hpp"
#include "operon/operators/creator.hpp"

namespace Operon::Test {

namespace {
    using S  = Operon::Scalar;
    using IE = IntervalEvaluator;

    auto Var(Operon::Hash h, double weight = 1.0) -> Operon::Node
    {
        Operon::Node n(Operon::NodeType::Variable, h);
        n.Value = static_cast<Operon::Scalar>(weight);
        return n;
    }

    auto Const(double v) -> Operon::Node { return Operon::Node::Constant(v); }

    auto Domains() -> IE::DomainMap { return IE::DomainMap{}; }
} // namespace

TEST_CASE("TightenRange - falls back to naive when a variable's gradient is unsupported (Abs)", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto absNode = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Abs), 1);
    auto tree = Operon::Tree({Var(X1), absNode}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{-1}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(naive.inf() == Catch::Approx(0.0).margin(1e-4));
    REQUIRE(naive.sup() == Catch::Approx(1.0).margin(1e-4));
    REQUIRE(tightened.inf() == Catch::Approx(naive.inf()).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(naive.sup()).margin(1e-4));
}

TEST_CASE("TightenRange - falls back to naive for n-ary Div (arity > 2, unsupported by Deriv)", "[range_tightening]")
{
    // Deriv() only handles Div at arity 1 or 2, returning NoGrad for
    // arity >= 3 - the structural pre-check must catch this the same way
    // it catches Abs, or TightenRange silently understates the gradient.
    constexpr Operon::Hash X1{1};
    auto divNode = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Div), 3);
    // Children are consumed nearest-first, so the numerator (X1) must be
    // pushed last (nearest the op) for this to mean X1 / 2 / 2.
    auto tree = Operon::Tree({Const(2), Const(2), Var(X1), divNode}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(naive.inf() == Catch::Approx(0.0).margin(1e-4));
    REQUIRE(naive.sup() == Catch::Approx(0.25).margin(1e-4));
    REQUIRE(tightened.inf() == Catch::Approx(naive.inf()).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(naive.sup()).margin(1e-4));
}

TEST_CASE("TightenRangeBisected - stays sound when an undifferentiated op is behind the recursion", "[range_tightening]")
{
    // TightenRangeBisected's variable-selection step calls
    // BuildVariableGradientDag directly (not through TightenRange's own
    // pre-check), so it can see an understated-but-nonzero gradient root
    // for X * abs(X). Soundness must still hold because every actual
    // enclosure returned at each recursion level goes through
    // TightenRange, which does bail to naive on Abs.
    constexpr Operon::Hash X1{1};
    auto x1 = Var(X1);
    auto x2 = Var(X1);
    auto absNode = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Abs), 1);
    auto mulNode = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2);
    auto tree = Operon::Tree({x1, x2, absNode, mulNode}).UpdateNodes(); // X * abs(X)
    auto d = Domains();
    d[X1] = {S{-1}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);
    auto const bisected  = TightenRangeBisected(tree, d, coeff, 3);

    // True range of x*|x| over [-1,1] is [-1,1].
    REQUIRE(tightened.inf() == Catch::Approx(naive.inf()).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(naive.sup()).margin(1e-4));
    REQUIRE(bisected.inf() <= -1.0 + 1e-3);
    REQUIRE(bisected.sup() >= 1.0 - 1e-3);
    REQUIRE(bisected.inf() >= naive.inf() - 1e-4);
    REQUIRE(bisected.sup() <= naive.sup() + 1e-4);
}

TEST_CASE("TightenRange - falls back to naive when only ONE occurrence of a variable is undifferentiated", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Abs), 1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Add), 2),
    }).UpdateNodes(); // X + abs(X)
    auto d = Domains();
    d[X1] = {S{-1}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(tightened.inf() <= naive.inf() + 1e-4);
    REQUIRE(tightened.sup() >= naive.sup() - 1e-4);
}

TEST_CASE("TightenRange - uses the live coeff span for an optimizable variable weight, not stale Node::Value", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto v = Var(X1, 1.0);
    v.Optimize = true;
    auto tree = Operon::Tree({v}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    std::vector<Operon::Scalar> const coeff{2.0F};

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(naive.inf() == Catch::Approx(0.0).margin(1e-4));
    REQUIRE(naive.sup() == Catch::Approx(2.0).margin(1e-4));
    REQUIRE(tightened.inf() == Catch::Approx(0.0).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(2.0).margin(1e-4));
}

TEST_CASE("TightenRange - constant tree matches naive exactly (no variables)", "[range_tightening]")
{
    auto tree = Operon::Tree({Const(2.5)}).UpdateNodes();
    auto const coeff = tree.GetCoefficients();
    auto const naive    = IE(&tree, Domains()).Evaluate(coeff);
    auto const tightened = TightenRange(tree, Domains(), coeff);

    REQUIRE(tightened.inf() == Catch::Approx(naive.inf()).margin(1e-6));
    REQUIRE(tightened.sup() == Catch::Approx(naive.sup()).margin(1e-6));
    REQUIRE(tightened.inf() == Catch::Approx(2.5).margin(1e-6));
}

TEST_CASE("TightenRange - single linear variable matches naive exactly (no dependency problem)", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({Var(X1, 2.0)}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{1}, S{3}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(tightened.inf() == Catch::Approx(naive.inf()).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(naive.sup()).margin(1e-4));
    REQUIRE(tightened.inf() == Catch::Approx(2.0).margin(1e-4));
    REQUIRE(tightened.sup() == Catch::Approx(6.0).margin(1e-4));
}

TEST_CASE("TightenRange - classic dependency-problem example x - x*x is tighter than naive", "[range_tightening]")
{
    // f(x) = x - x^2 over [0,1]: true range [0, 0.25]. Postfix "a - b" is
    // [b-subtree, a, Sub]; here a = x, b = x*x.
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2),
        Var(X1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 2),
    }).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(naive.inf() == Catch::Approx(-1.0).margin(1e-4));
    REQUIRE(naive.sup() == Catch::Approx(1.0).margin(1e-4));

    REQUIRE(tightened.inf() > naive.inf());
    REQUIRE(tightened.sup() < naive.sup());
    REQUIRE(tightened.inf() <= 0.0 + 1e-4);
    REQUIRE(tightened.sup() >= 0.25 - 1e-4);
}

TEST_CASE("TightenRangeBisected - x - x*x: bisection is a genuine further improvement over TightenRange", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2),
        Var(X1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 2),
    }).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const flat     = TightenRange(tree, d, coeff);
    auto const bisected = TightenRangeBisected(tree, d, coeff, 4);

    REQUIRE(bisected.inf() >= flat.inf() - 1e-4);
    REQUIRE(bisected.sup() <= flat.sup() + 1e-4);
    REQUIRE(bisected.sup() < flat.sup()); // genuine further improvement
    // Still soundly contains the true range [0, 0.25].
    REQUIRE(bisected.inf() <= 0.0 + 1e-3);
    REQUIRE(bisected.sup() >= 0.25 - 1e-3);
}

TEST_CASE("TightenRangeBisected - maxDepth 0 matches TightenRange exactly", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2),
        Var(X1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 2),
    }).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    auto const flat     = TightenRange(tree, d, coeff);
    auto const bisected = TightenRangeBisected(tree, d, coeff, 0);

    REQUIRE(bisected.inf() == Catch::Approx(flat.inf()).margin(1e-6));
    REQUIRE(bisected.sup() == Catch::Approx(flat.sup()).margin(1e-6));
}

TEST_CASE("TightenRangeBisected - never looser than TightenRange, soundness against random samples", "[range_tightening]")
{
    constexpr auto nTrees  = 100;
    constexpr auto nPoints = 20;
    constexpr auto maxLen  = 20;
    constexpr auto tol     = 1e-3F;

    Operon::RandomGenerator rng(88UL);

    PrimitiveSet pset;
    pset.SetConfig(
        BuiltinOp::Add | BuiltinOp::Mul | BuiltinOp::Sub | BuiltinOp::Div |
        BuiltinOp::Sin | BuiltinOp::Cos | BuiltinOp::Sqrt | BuiltinOp::Square |
        NodeType::Constant | NodeType::Variable
    );

    constexpr int nVars = 3;
    std::vector<std::string> names(nVars);
    for (int i = 0; i < nVars; ++i) { names[static_cast<std::size_t>(i)] = fmt::format("X{}", i + 1); }
    std::vector<std::vector<Operon::Scalar>> data(nVars, std::vector<Operon::Scalar>(1, Operon::Scalar{0}));
    Dataset const ds(names, data);
    auto const varHashes = ds.VariableHashes();

    std::uniform_real_distribution<Operon::Scalar> valDist(-2.F, 2.F);
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    std::uniform_real_distribution<Operon::Scalar> domainLoDist(-2.F, 0.F);
    std::uniform_real_distribution<Operon::Scalar> domainWidthDist(0.5F, 3.F);

    BalancedTreeCreator const btc{&pset, varHashes, /*bias=*/0.0, maxLen};
    using DTable = DispatchTable<Operon::Scalar>;
    using Interp = Interpreter<Operon::Scalar, DTable>;
    DTable dtable;

    std::size_t looserThanFlat = 0;
    std::size_t checked        = 0;
    std::size_t violated       = 0;

    for (int t = 0; t < nTrees; ++t) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        for (auto& nd : tree.Nodes()) {
            nd.Optimize = nd.IsLeaf();
            if (nd.IsLeaf()) { nd.Value = valDist(rng); }
        }

        IE::DomainMap domains;
        for (auto h : varHashes) {
            auto const lo = domainLoDist(rng);
            domains[h] = {lo, lo + domainWidthDist(rng)};
        }

        auto const coeff    = tree.GetCoefficients();
        auto const flat     = TightenRange(tree, domains, coeff);
        auto const bisected = TightenRangeBisected(tree, domains, coeff, 3);
        if (bisected.is_empty()) { continue; }

        if (bisected.inf() < flat.inf() - tol || bisected.sup() > flat.sup() + tol) { ++looserThanFlat; }

        std::vector<std::vector<Operon::Scalar>> pointData(
            static_cast<std::size_t>(nVars), std::vector<Operon::Scalar>(nPoints));
        for (std::size_t vi = 0; vi < varHashes.size(); ++vi) {
            auto const [lo, hi] = domains[varHashes[vi]];
            std::uniform_real_distribution<Operon::Scalar> pd(lo, hi);
            for (auto& v : pointData[vi]) { v = pd(rng); }
        }
        Dataset const pointDs(names, pointData);
        Range const range{0, nPoints};
        auto const values = Interp{&dtable, &pointDs, &tree}.Evaluate(coeff, range);

        for (auto v : values) {
            if (!std::isfinite(v)) { continue; }
            ++checked;
            if (v < bisected.inf() - tol || v > bisected.sup() + tol) { ++violated; }
        }
    }

    INFO("looser than flat TightenRange: " << looserThanFlat << " / " << nTrees);
    CHECK(looserThanFlat == 0);
    INFO("soundness violations: " << violated << " / " << checked);
    CHECK(violated == 0);
    CHECK(checked > 0);
}

TEST_CASE("TightenRange - result is always a subset of the naive enclosure", "[range_tightening]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    auto x1   = Var(X1);
    auto x2   = Var(X2);
    auto add  = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Add), 2);
    auto sin  = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sin), 1);
    auto tree = Operon::Tree({x1, x2, add, sin}).UpdateNodes(); // sin(x1 + x2)
    auto d = Domains();
    d[X1] = {S{-1}, S{2}};
    d[X2] = {S{0}, S{3}};
    auto const coeff = tree.GetCoefficients();

    auto const naive     = IE(&tree, IE::DomainMap{d}).Evaluate(coeff);
    auto const tightened = TightenRange(tree, d, coeff);

    REQUIRE(tightened.inf() >= naive.inf() - 1e-4);
    REQUIRE(tightened.sup() <= naive.sup() + 1e-4);
}

TEST_CASE("TightenRange - soundness against random point samples on random trees", "[range_tightening]")
{
    constexpr auto nTrees  = 300;
    constexpr auto nPoints = 20;
    constexpr auto maxLen  = 20;
    constexpr auto tol     = 1e-3F;

    Operon::RandomGenerator rng(77UL);

    PrimitiveSet pset;
    pset.SetConfig(
        BuiltinOp::Add | BuiltinOp::Mul | BuiltinOp::Sub | BuiltinOp::Div |
        BuiltinOp::Exp | BuiltinOp::Log | BuiltinOp::Sin | BuiltinOp::Cos |
        BuiltinOp::Sqrt | BuiltinOp::Square |
        NodeType::Constant | NodeType::Variable
    );

    constexpr int nVars = 3;
    std::vector<std::string> names(nVars);
    for (int i = 0; i < nVars; ++i) { names[static_cast<std::size_t>(i)] = fmt::format("X{}", i + 1); }
    std::vector<std::vector<Operon::Scalar>> data(nVars, std::vector<Operon::Scalar>(1, Operon::Scalar{0}));
    Dataset const ds(names, data); // single dummy row; only VariableHashes() is used below
    auto const varHashes = ds.VariableHashes();

    std::uniform_real_distribution<Operon::Scalar> valDist(-2.F, 2.F);
    std::uniform_int_distribution<std::size_t> lenDist(1, maxLen);
    std::uniform_real_distribution<Operon::Scalar> domainLoDist(-2.F, 0.F);
    std::uniform_real_distribution<Operon::Scalar> domainWidthDist(0.1F, 3.F);

    BalancedTreeCreator const btc{&pset, varHashes, /*bias=*/0.0, maxLen};

    using DTable = DispatchTable<Operon::Scalar>;
    using Interp = Interpreter<Operon::Scalar, DTable>;
    DTable dtable;

    std::size_t checked  = 0;
    std::size_t violated = 0;

    for (int t = 0; t < nTrees; ++t) {
        auto tree = btc(rng, lenDist(rng), 1, 1000);
        for (auto& nd : tree.Nodes()) {
            nd.Optimize = nd.IsLeaf();
            if (nd.IsLeaf()) { nd.Value = valDist(rng); }
        }

        IE::DomainMap domains;
        for (auto h : varHashes) {
            auto const lo = domainLoDist(rng);
            domains[h] = {lo, lo + domainWidthDist(rng)};
        }

        auto const coeff = tree.GetCoefficients();
        auto const tightened = TightenRange(tree, domains, coeff);
        if (tightened.is_empty()) { continue; } // domain edge (e.g. log of negative) - nothing to check

        // Random point dataset rows within the domain box.
        std::vector<std::vector<Operon::Scalar>> pointData(
            static_cast<std::size_t>(nVars), std::vector<Operon::Scalar>(nPoints));
        for (std::size_t vi = 0; vi < varHashes.size(); ++vi) {
            auto const [lo, hi] = domains[varHashes[vi]];
            std::uniform_real_distribution<Operon::Scalar> pd(lo, hi);
            for (auto& v : pointData[vi]) { v = pd(rng); }
        }
        Dataset const pointDs(names, pointData);
        Range const range{0, nPoints};
        auto const values = Interp{&dtable, &pointDs, &tree}.Evaluate(coeff, range);

        for (auto v : values) {
            if (!std::isfinite(v)) { continue; } // domain edge in the original function itself
            ++checked;
            if (v < tightened.inf() - tol || v > tightened.sup() + tol) { ++violated; }
        }
    }

    INFO("soundness violations: " << violated << " / " << checked);
    CHECK(violated == 0);
    CHECK(checked > 0); // sanity: the sweep actually exercised something
}

TEST_CASE("RangeCache - hit reproduces the same result as an uncached call", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2),
        Var(X1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 2),
    }).UpdateNodes(); // X - X*X
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    Operon::RandomGenerator rng(1234);
    std::array<Operon::Hash, 1> const varHashes{X1};
    Operon::Zobrist zobrist(rng, static_cast<int>(tree.Length()), varHashes);
    Operon::RangeCache cache(zobrist);

    auto const uncached = TightenRange(tree, d, coeff);
    REQUIRE(cache.Size() == 0);

    auto const firstCall = TightenRange(tree, d, coeff, &cache);
    REQUIRE(cache.Size() == 1); // populated on miss

    auto const secondCall = TightenRange(tree, d, coeff, &cache);
    REQUIRE(cache.Size() == 1); // hit, no new entry

    CHECK(firstCall.inf() == Catch::Approx(uncached.inf()).margin(1e-6));
    CHECK(firstCall.sup() == Catch::Approx(uncached.sup()).margin(1e-6));
    CHECK(secondCall.inf() == Catch::Approx(uncached.inf()).margin(1e-6));
    CHECK(secondCall.sup() == Catch::Approx(uncached.sup()).margin(1e-6));
}

TEST_CASE("RangeCache - a coefficient change invalidates the cache entry (never returns a stale bound)", "[range_tightening]")
{
    // The whole point of keying on coeff (not just structural hash, unlike
    // the fitness cache) is that TightenRange's soundness depends on the
    // actual weight values - a coefficient-independent cache would let a
    // locally-searched individual's changed weights silently reuse a bound
    // computed for different weights.
    constexpr Operon::Hash X1{1};
    auto v = Var(X1, 1.0);
    v.Optimize = true;
    auto tree = Operon::Tree({v}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};

    Operon::RandomGenerator rng(1234);
    std::array<Operon::Hash, 1> const varHashes{X1};
    Operon::Zobrist zobrist(rng, static_cast<int>(tree.Length()), varHashes);
    Operon::RangeCache cache(zobrist);

    std::vector<Operon::Scalar> const coeffA{1.0F};
    std::vector<Operon::Scalar> const coeffB{2.0F};

    auto const resultA = TightenRange(tree, d, coeffA, &cache);
    REQUIRE(cache.Size() == 1);
    auto const resultB = TightenRange(tree, d, coeffB, &cache);
    REQUIRE(cache.Size() == 2); // different coeff: a genuinely new entry, not a stale hit

    CHECK(resultA.sup() == Catch::Approx(1.0).margin(1e-4));  // f(x)=1*x over [0,1]
    CHECK(resultB.sup() == Catch::Approx(2.0).margin(1e-4));  // f(x)=2*x over [0,1]
}

TEST_CASE("RangeCache - TightenRangeBisected result does not collide with TightenRange's own cache entry", "[range_tightening]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({
        Var(X1), Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2),
        Var(X1),
        Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 2),
    }).UpdateNodes(); // X - X*X
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    auto const coeff = tree.GetCoefficients();

    Operon::RandomGenerator rng(1234);
    std::array<Operon::Hash, 1> const varHashes{X1};
    Operon::Zobrist zobrist(rng, static_cast<int>(tree.Length()), varHashes);
    Operon::RangeCache cache(zobrist);

    auto const flat     = TightenRange(tree, d, coeff, &cache);
    auto const bisected = TightenRangeBisected(tree, d, coeff, 4, &cache);

    // Bisection is a genuine improvement here (same as the uncached test
    // above) - if the two results collided in the cache, they'd be equal.
    CHECK(bisected.sup() < flat.sup());

    // Second bisected call must hit the cache and reproduce the same result.
    auto const bisectedAgain = TightenRangeBisected(tree, d, coeff, 4, &cache);
    CHECK(bisectedAgain.inf() == Catch::Approx(bisected.inf()).margin(1e-6));
    CHECK(bisectedAgain.sup() == Catch::Approx(bisected.sup()).margin(1e-6));
}

} // namespace Operon::Test
