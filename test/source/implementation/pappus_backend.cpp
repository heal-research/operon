// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Compiled only when pappus is enabled (OPERON_ENABLE_PAPPUS). Otherwise this
// translation unit is empty so the test executable builds without pappus.

#ifdef OPERON_ENABLE_PAPPUS

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/affine_evaluator.hpp"

namespace Operon::Test {

namespace {
    using S = Operon::Scalar;
    using IE = IntervalEvaluator;
    using AE = AffineEvaluator;

    // Build a variable node with a given hash and weight.
    auto Var(Operon::Hash h, double weight = 1.0) -> Operon::Node
    {
        Operon::Node n(Operon::NodeType::Variable, h);
        n.Value = static_cast<Operon::Scalar>(weight);
        return n;
    }

    auto Const(double v) -> Operon::Node { return Operon::Node::Constant(v); }

    // Interval contains a value (with tolerance for outward-rounded bounds).
    auto Contains(IE::Interval const& iv, double lo, double hi, double tol) -> bool
    {
        return iv.inf() <= lo + tol && iv.sup() + tol >= hi;
    }

    // Affine enclosure contains a value range.
    auto Contains(AE::Affine const& af, double lo, double hi, double tol) -> bool
    {
        auto const iv = af.to_interval();
        return iv.inf() <= lo + tol && iv.sup() + tol >= hi;
    }

    auto Domains() -> IE::DomainMap
    {
        return IE::DomainMap{};
    }
} // namespace

TEST_CASE("Interval backend: constants and variables", "[pappus][interval]")
{
    SECTION("constant tree") {
        auto tree = Operon::Tree({Const(2.5)}).UpdateNodes();
        IE eval(&tree, Domains());
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(2.5).margin(1e-5));
        REQUIRE(r.sup() == Catch::Approx(2.5).margin(1e-5));
    }

    SECTION("single variable over its domain") {
        constexpr Operon::Hash X1{1};
        auto tree = Operon::Tree({Var(X1)}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{3}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(1.0).margin(1e-5));
        REQUIRE(r.sup() == Catch::Approx(3.0).margin(1e-5));
    }

    SECTION("weighted variable") {
        constexpr Operon::Hash X1{1};
        auto tree = Operon::Tree({Var(X1, 2.0)}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{3}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(2.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(6.0).margin(1e-4));
    }
}

TEST_CASE("Interval backend: arithmetic", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    auto d = Domains();
    d[X1] = {S{1}, S{2}};
    d[X2] = {S{3}, S{4}};

    auto make = [&](Operon::Vector<Operon::Node> ns) -> IE {
        static Operon::Tree t; t = Operon::Tree(std::move(ns)).UpdateNodes(); // NOLINT
        return IE(&t, IE::DomainMap{d});
    };

    SECTION("X1 + X2 -> [4, 6]") {
        auto eval = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Add)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-5));
    }

    SECTION("X1 - X2 -> [-3, -1]") {
        // operon lays out binary-op children with the semantic LEFT operand at
        // the higher index (rightmost); `Tree::Indices` yields rightmost-first.
        auto eval = make({Var(X2), Var(X1), Operon::Node(Operon::NodeType::Sub)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -3.0, -1.0, 1e-5));
    }

    SECTION("X1 * X2 -> [3, 8]") {
        auto eval = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Mul)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-5));
    }

    SECTION("X1 / X2 -> [0.25, 0.6667]") {
        auto eval = make({Var(X2), Var(X1), Operon::Node(Operon::NodeType::Div)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(r.inf() <= 0.25 + 1e-5);
        REQUIRE(r.sup() + 1e-5 >= 2.0/3.0);
    }

    SECTION("neg(X1) -> [-2, -1]") {
        auto node = Operon::Node(Operon::NodeType::Sub);
        node.Arity = 1;
        auto eval = make({Var(X1), node});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -2.0, -1.0, 1e-5));
    }

    SECTION("inv(X1) -> [0.5, 1]") {
        auto node = Operon::Node(Operon::NodeType::Div);
        node.Arity = 1;
        auto eval = make({Var(X1), node});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 0.5, 1.0, 1e-5));
    }
}

TEST_CASE("Interval backend: unary transcendentals", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};
    auto d = Domains();

    auto make = [&](IE::Domain dom, Operon::NodeType op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t; t = Operon::Tree({Var(X1), Operon::Node(op)}).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = {S{dom.first}, S{dom.second}};
        IE eval(&t, std::move(dm));
        return {eval, eval.Evaluate(eval.GetTree()->GetCoefficients())};
    };

    SECTION("square([-1, 2]) -> [0, 4]") {
        auto [eval, r] = make({-1, 2}, Operon::NodeType::Square);
        REQUIRE(Contains(r, 0.0, 4.0, 1e-4));
    }

    SECTION("sqrt([1, 4]) -> [1, 2]") {
        auto [eval, r] = make({1, 4}, Operon::NodeType::Sqrt);
        REQUIRE(Contains(r, 1.0, 2.0, 1e-4));
    }

    SECTION("exp([0, 1]) -> [1, e]") {
        auto [eval, r] = make({0, 1}, Operon::NodeType::Exp);
        REQUIRE(r.inf() <= 1.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= std::exp(1.0));
    }

    SECTION("log([1, e]) -> [0, 1]") {
        auto [eval, r] = make({1, std::exp(1.0)}, Operon::NodeType::Log);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= 1.0);
    }
}

TEST_CASE("Interval backend: nested tree", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1}, X2{2}, X3{3};
    // (X1 + X2) * X3
    Operon::Vector<Operon::Node> ns{
        Var(X1), Var(X2), Operon::Node(Operon::NodeType::Add),
        Var(X3), Operon::Node(Operon::NodeType::Mul)};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();

    auto d = Domains();
    d[X1] = {S{1}, S{2}};
    d[X2] = {S{3}, S{4}};
    d[X3] = {S{5}, S{6}};
    IE eval(&tree, std::move(d));
    auto const r = eval.Evaluate(tree.GetCoefficients());
    // (X1+X2) in [4,6]; * X3 in [5,6] -> [20, 36]
    REQUIRE(Contains(r, 20.0, 36.0, 1e-3));
}

TEST_CASE("Interval backend: abs and composable ops", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};

    auto make = [&](IE::Domain dom, Operon::NodeType op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t; t = Operon::Tree({Var(X1), Operon::Node(op)}).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = {S{dom.first}, S{dom.second}};
        IE eval(&t, std::move(dm));
        return {eval, eval.Evaluate(eval.GetTree()->GetCoefficients())};
    };

    SECTION("abs([-3, -1]) -> [1, 3]") {
        auto [eval, r] = make({-3, -1}, Operon::NodeType::Abs);
        REQUIRE(Contains(r, 1.0, 3.0, 1e-4));
    }
    SECTION("abs([-1, 2]) -> [0, 2]") {
        auto [eval, r] = make({-1, 2}, Operon::NodeType::Abs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-4));
    }
    SECTION("abs([2, 5]) -> [2, 5]") {
        auto [eval, r] = make({2, 5}, Operon::NodeType::Abs);
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }
    SECTION("sqrtabs([-1, 4]) -> [0, 2]") {
        auto [eval, r] = make({-1, 4}, Operon::NodeType::Sqrtabs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-3));
    }
    SECTION("logabs([-3, -1]) -> [0, log(3)]") {
        auto [eval, r] = make({-3, -1}, Operon::NodeType::Logabs);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= std::log(3.0));
    }
}

TEST_CASE("Interval backend: fmin/fmax", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    auto d = Domains();
    d[X1] = {S{1}, S{3}};
    d[X2] = {S{2}, S{4}};

    auto make = [&](Operon::Vector<Operon::Node> ns) -> IE::Interval {
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        IE eval(&tree, IE::DomainMap{d});
        return eval.Evaluate(tree.GetCoefficients());
    };

    SECTION("fmin(X1, X2) -> [1, 3]") {
        // min([1,3], [2,4]) = [min(1,2), min(3,4)] = [1, 3]
        auto r = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Fmin)});
        REQUIRE(Contains(r, 1.0, 3.0, 1e-5));
    }
    SECTION("fmax(X1, X2) -> [2, 4]") {
        // max([1,3], [2,4]) = [max(1,2), max(3,4)] = [2, 4]
        auto r = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Fmax)});
        REQUIRE(Contains(r, 2.0, 4.0, 1e-5));
    }
}

TEST_CASE("Interval backend: aq", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    // aq(X1, X2) = X1 / sqrt(1 + X2^2)
    // X1 in [3, 6], X2 in [0, 2] -> 1+X2^2 in [1, 5] -> sqrt in [1, sqrt(5)]
    // -> aq in [3/sqrt(5), 6/1] = [1.34, 6]
    Operon::Vector<Operon::Node> ns{
        Var(X2), Var(X1), Operon::Node(Operon::NodeType::Aq)};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{3}, S{6}};
    d[X2] = {S{0}, S{2}};
    IE eval(&tree, std::move(d));
    auto const r = eval.Evaluate(tree.GetCoefficients());
    REQUIRE(r.inf() <= 3.0 / std::sqrt(5.0) + 1e-3);
    REQUIRE(r.sup() + 1e-3 >= 6.0);
}

TEST_CASE("Affine backend: aq", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    // aq(X1, X2) = X1 / sqrt(1 + X2^2)
    // X1 in [3, 6], X2 in [1, 2] -> 1+X2^2 in [2, 5] -> sqrt in [sqrt(2), sqrt(5)]
    // -> aq in [3/sqrt(5), 6/sqrt(2)] = [1.34, 4.24]
    // Note: X2 domain [0, 2] would cause affine overestimation in X2^2
    // (interval [-1, 4] instead of [0, 4]), making 1+X2^2 contain 0 and
    // triggering inv() to throw. This is a known affine arithmetic limitation.
    Operon::Vector<Operon::Node> ns{
        Var(X2), Var(X1), Operon::Node(Operon::NodeType::Aq)};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{3}, S{6}};
    d[X2] = {S{1}, S{2}};
    AE eval(&tree, std::move(d));
    auto const r = eval.Evaluate(tree.GetCoefficients());
    auto const iv = r.to_interval();
    // aq enclosure must contain the true range [3/sqrt(5), 6/sqrt(2)]
    REQUIRE(iv.inf() <= 3.0 / std::sqrt(5.0) + 1e-3);
    REQUIRE(iv.sup() + 1e-3 >= 6.0 / std::sqrt(2.0));
}

TEST_CASE("Affine backend: abs (non-zero-crossing)", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1};

    SECTION("abs of positive domain -> identity") {
        auto tree = Operon::Tree({Var(X1), Operon::Node(Operon::NodeType::Abs)}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{2}, S{5}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of negative domain -> negation") {
        auto tree = Operon::Tree({Var(X1), Operon::Node(Operon::NodeType::Abs)}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-5}, S{-2}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of zero-crossing domain -> throws") {
        auto tree = Operon::Tree({Var(X1), Operon::Node(Operon::NodeType::Abs)}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-1}, S{2}};
        AE eval(&tree, std::move(d));
        REQUIRE_THROWS_AS(eval.Evaluate(tree.GetCoefficients()), std::runtime_error);
    }
}

TEST_CASE("Interval backend: missing domain throws", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({Var(X1)}).UpdateNodes();
    IE eval(&tree, Domains()); // no domain for X1
    REQUIRE_THROWS_AS(eval.Evaluate(tree.GetCoefficients()), std::runtime_error);
}

TEST_CASE("Affine backend: arithmetic", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1}, X2{2};
    auto d = Domains();
    d[X1] = {S{1}, S{2}};
    d[X2] = {S{3}, S{4}};

    auto make = [&](Operon::Vector<Operon::Node> ns) -> AE {
        static Operon::Tree t; t = Operon::Tree(std::move(ns)).UpdateNodes(); // NOLINT
        return AE(&t, AE::DomainMap{d});
    };

    SECTION("X1 + X2 (exact in affine) -> [4, 6]") {
        auto eval = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Add)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-4));
    }

    SECTION("X1 * X2 (affine enclosure contains [3, 8])") {
        auto eval = make({Var(X1), Var(X2), Operon::Node(Operon::NodeType::Mul)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-3));
    }

    SECTION("X1 - X2 -> contains [-3, -1]") {
        auto eval = make({Var(X2), Var(X1), Operon::Node(Operon::NodeType::Sub)});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -3.0, -1.0, 1e-4));
    }
}

TEST_CASE("Affine backend: shared context", "[pappus][affine]")
{
    // Re-evaluating the same tree twice reuses the evaluator's context;
    // no mixed-context exception should arise.
    constexpr Operon::Hash X1{1}, X2{2};
    Operon::Vector<Operon::Node> ns{
        Var(X1), Var(X2), Operon::Node(Operon::NodeType::Add),
        Operon::Node(Operon::NodeType::Exp)};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    d[X2] = {S{0}, S{1}};
    AE eval(&tree, std::move(d));

    auto const c0 = eval.Evaluate(tree.GetCoefficients());
    auto const c1 = eval.Evaluate(tree.GetCoefficients());
    REQUIRE(c0.to_interval().inf() <= c1.to_interval().sup() + 1e-3);
    REQUIRE(c1.to_interval().inf() <= c0.to_interval().sup() + 1e-3);
}

TEST_CASE("Affine backend: no index reuse across evaluations", "[pappus][affine]")
{
    // Soundness: two Evaluate calls must not reuse noise-symbol indices.
    // If they did, subtracting results from independent evaluations would
    // falsely cancel (giving [0,0] instead of the true range [-2, 2]).
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({Var(X1)}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{1}, S{3}};
    AE eval(&tree, std::move(d));

    auto const r1 = eval.Evaluate(tree.GetCoefficients());
    auto const r2 = eval.Evaluate(tree.GetCoefficients());
    // r1 and r2 are from independent evaluations; r1 - r2 must enclose [-2, 2].
    auto const diff = pappus::ops::sub<Scalar>(r1, r2);
    auto const iv = diff.to_interval();
    REQUIRE(iv.inf() <= -2.0 + 1e-4);
    REQUIRE(iv.sup() + 1e-4 >= 2.0);
    // And must NOT be [0, 0] (the unsound result from index reuse).
    REQUIRE(iv.sup() - iv.inf() > 1.0);
}

TEST_CASE("Affine backend: inv over zero-containing domain throws", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1};
    auto node = Operon::Node(Operon::NodeType::Div);
    node.Arity = 1;
    auto tree = Operon::Tree({Var(X1), node}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{-1}, S{1}}; // contains zero
    AE eval(&tree, std::move(d));
    REQUIRE_THROWS_AS(eval.Evaluate(tree.GetCoefficients()), std::invalid_argument);
}

} // namespace Operon::Test

#endif // OPERON_ENABLE_PAPPUS
