// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../operon_test.hpp"

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"

namespace Operon::Test {

namespace {
    using S = Operon::Scalar;
    using IE = IntervalEvaluator<S>;
    using AE = AffineEvaluator<S>;

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

    auto Domains() -> IE::DomainMap { return IE::DomainMap {}; }

    // Evaluation must succeed; unwrap the enclosure.
    auto Eval(IE const& eval, Operon::Span<Operon::Scalar const> coeff) -> IE::Interval
    {
        auto const result = eval(coeff);
        REQUIRE(result);
        return *result;
    }

    auto Eval(AE const& eval, Operon::Span<Operon::Scalar const> coeff) -> AE::Affine
    {
        auto const result = eval(coeff);
        REQUIRE(result);
        return *result;
    }
} // namespace

TEST_CASE("Interval backend: constants and variables", "[pappus][interval]")
{
    SECTION("constant tree")
    {
        auto tree = Operon::Tree({ Const(2.5) }).UpdateNodes();
        IE eval(&tree, Domains());
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(2.5).margin(1e-5));
        REQUIRE(r.sup() == Catch::Approx(2.5).margin(1e-5));
    }

    SECTION("single variable over its domain")
    {
        constexpr Operon::Hash X1 { 1 };
        auto tree = Operon::Tree({ Var(X1) }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 3 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(1.0).margin(1e-5));
        REQUIRE(r.sup() == Catch::Approx(3.0).margin(1e-5));
    }

    SECTION("domain map is snapshotted at construction")
    {
        constexpr Operon::Hash X1 { 1 };
        auto tree = Operon::Tree({ Var(X1) }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 3 } };
        IE eval(&tree, d);
        d[X1] = { S { 2 }, S { 4 } };
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(1.0).margin(1e-5));
        REQUIRE(r.sup() == Catch::Approx(3.0).margin(1e-5));
    }

    SECTION("weighted variable")
    {
        constexpr Operon::Hash X1 { 1 };
        auto tree = Operon::Tree({ Var(X1, 2.0) }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 3 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(2.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(6.0).margin(1e-4));
    }
}

TEST_CASE("Interval backend: arithmetic", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    auto d = Domains();
    d[X1] = { S { 1 }, S { 2 } };
    d[X2] = { S { 3 }, S { 4 } };

    auto make = [&](Operon::Vector<Operon::Node> ns) -> IE {
        static Operon::Tree t;
        t = Operon::Tree(std::move(ns)).UpdateNodes(); // NOLINT
        return IE(&t, IE::DomainMap { d });
    };

    SECTION("X1 + X2 -> [4, 6]")
    {
        auto eval = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-5));
    }

    SECTION("X1 - X2 -> [-3, -1]")
    {
        // operon lays out binary-op children with the semantic LEFT operand at
        // the higher index (rightmost); `Tree::Indices` yields rightmost-first.
        auto eval = make({ Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Sub>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -3.0, -1.0, 1e-5));
    }

    SECTION("X1 * X2 -> [3, 8]")
    {
        auto eval = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-5));
    }

    SECTION("X1 / X2 -> [0.25, 0.6667]")
    {
        auto eval = make({ Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Div>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(r.inf() <= 0.25 + 1e-5);
        REQUIRE(r.sup() + 1e-5 >= 2.0 / 3.0);
    }

    SECTION("neg(X1) -> [-2, -1]")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Sub>();
        node.Arity = 1;
        auto eval = make({ Var(X1), node });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -2.0, -1.0, 1e-5));
    }

    SECTION("inv(X1) -> [0.5, 1]")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
        node.Arity = 1;
        auto eval = make({ Var(X1), node });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 0.5, 1.0, 1e-5));
    }
}

TEST_CASE("Interval backend: weighted n-ary subtraction and division", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    IE::DomainMap const domains { { X1, { S { 1 }, S { 2 } } }, { X2, { S { 3 }, S { 4 } } } };

    SECTION("2 * (X1 - X2 - 5) -> [-16, -12]")
    {
        auto sub = Util::MakeOp<Operon::BuiltinOp::Sub>();
        sub.Arity = 3;
        sub.Value = S { 2 };
        auto const tree = Operon::Tree({ Const(5), Var(X2), Var(X1), sub }).UpdateNodes();
        auto const result = Eval(IE(&tree, domains), tree.GetCoefficients());

        REQUIRE(Contains(result, -16.0, -12.0, 1e-5));
    }

    SECTION("4 * (X1 / 2 / 2) -> [1, 2]")
    {
        auto div = Util::MakeOp<Operon::BuiltinOp::Div>();
        div.Arity = 3;
        div.Value = S { 4 };
        auto const tree = Operon::Tree({ Const(2), Const(2), Var(X1), div }).UpdateNodes();
        auto const result = Eval(IE(&tree, domains), tree.GetCoefficients());
        REQUIRE(Contains(result, 1.0, 2.0, 1e-5));
    }
}

TEST_CASE("Interval backend: unary transcendentals", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };
    auto d = Domains();

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t;
        t = Operon::Tree({ Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1) }).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = { S { dom.first }, S { dom.second } };
        IE eval(&t, std::move(dm));
        return { eval, Eval(eval, eval.GetTree()->GetCoefficients()) };
    };

    SECTION("square([-1, 2]) -> [0, 4]")
    {
        auto [eval, r] = make({ -1, 2 }, Operon::BuiltinOp::Square);
        REQUIRE(Contains(r, 0.0, 4.0, 1e-4));
    }

    SECTION("sqrt([1, 4]) -> [1, 2]")
    {
        auto [eval, r] = make({ 1, 4 }, Operon::BuiltinOp::Sqrt);
        REQUIRE(Contains(r, 1.0, 2.0, 1e-4));
    }

    SECTION("exp([0, 1]) -> [1, e]")
    {
        auto [eval, r] = make({ 0, 1 }, Operon::BuiltinOp::Exp);
        REQUIRE(r.inf() <= 1.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= std::exp(1.0));
    }

    SECTION("log([1, e]) -> [0, 1]")
    {
        auto [eval, r] = make({ 1, std::exp(1.0) }, Operon::BuiltinOp::Log);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= 1.0);
    }
}

// Distinguishing-value coverage for the trig/hyperbolic registry entries not
// exercised elsewhere in this file: a hash-preserving swap between any two of
// these (e.g. Sin's lambda registered under Cos's slot) would not be caught
// by registry_coverage.cpp's presence-only check, but is caught here since
// each op's true range over its chosen monotonic domain is distinct.
TEST_CASE("Interval backend: trig and hyperbolic transcendentals", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> IE::Interval {
        auto tree = Operon::Tree({ Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1) }).UpdateNodes();
        auto dm = Domains();
        dm[X1] = { S { dom.first }, S { dom.second } };
        IE eval(&tree, std::move(dm));
        return Eval(eval, tree.GetCoefficients());
    };

    constexpr double Pi = 3.14159265358979323846;

    SECTION("sin([0, pi/6]) -> [0, 0.5]")
    {
        auto r = make({ 0, Pi / 6.0 }, Operon::BuiltinOp::Sin);
        REQUIRE(Contains(r, 0.0, 0.5, 1e-3));
    }
    SECTION("cos([0, pi/3]) -> [0.5, 1]")
    {
        auto r = make({ 0, Pi / 3.0 }, Operon::BuiltinOp::Cos);
        REQUIRE(Contains(r, 0.5, 1.0, 1e-3));
    }
    SECTION("tan([0, pi/4]) -> [0, 1]")
    {
        auto r = make({ 0, Pi / 4.0 }, Operon::BuiltinOp::Tan);
        REQUIRE(Contains(r, 0.0, 1.0, 1e-3));
    }
    SECTION("asin([0, 0.5]) -> [0, pi/6]")
    {
        auto r = make({ 0, 0.5 }, Operon::BuiltinOp::Asin);
        REQUIRE(Contains(r, 0.0, Pi / 6.0, 1e-3));
    }
    SECTION("acos([0, 0.5]) -> [pi/3, pi/2]")
    {
        auto r = make({ 0, 0.5 }, Operon::BuiltinOp::Acos);
        REQUIRE(Contains(r, Pi / 3.0, Pi / 2.0, 1e-3));
    }
    SECTION("atan([0, 1]) -> [0, pi/4]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Atan);
        REQUIRE(Contains(r, 0.0, Pi / 4.0, 1e-3));
    }
    SECTION("sinh([0, 1]) -> [0, sinh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Sinh);
        REQUIRE(Contains(r, 0.0, std::sinh(1.0), 1e-3));
    }
    SECTION("cosh([0, 1]) -> [1, cosh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Cosh);
        REQUIRE(Contains(r, 1.0, std::cosh(1.0), 1e-3));
    }
    SECTION("tanh([0, 1]) -> [0, tanh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Tanh);
        REQUIRE(Contains(r, 0.0, std::tanh(1.0), 1e-3));
    }
}

// Affine analog of the interval test above, same distinguishing-domain
// rationale.
TEST_CASE("Affine backend: trig and hyperbolic transcendentals", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 };

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> AE::Affine {
        auto tree = Operon::Tree({ Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1) }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { dom.first }, S { dom.second } };
        AE eval(&tree, std::move(d));
        return Eval(eval, tree.GetCoefficients());
    };

    constexpr double Pi = 3.14159265358979323846;

    SECTION("sin([0, pi/6]) -> contains [0, 0.5]")
    {
        auto r = make({ 0, Pi / 6.0 }, Operon::BuiltinOp::Sin);
        REQUIRE(Contains(r, 0.0, 0.5, 1e-2));
    }
    SECTION("cos([0, pi/3]) -> contains [0.5, 1]")
    {
        auto r = make({ 0, Pi / 3.0 }, Operon::BuiltinOp::Cos);
        REQUIRE(Contains(r, 0.5, 1.0, 1e-2));
    }
    SECTION("tan([0, pi/4]) -> contains [0, 1]")
    {
        auto r = make({ 0, Pi / 4.0 }, Operon::BuiltinOp::Tan);
        REQUIRE(Contains(r, 0.0, 1.0, 1e-2));
    }
    SECTION("asin([0, 0.5]) -> contains [0, pi/6]")
    {
        auto r = make({ 0, 0.5 }, Operon::BuiltinOp::Asin);
        REQUIRE(Contains(r, 0.0, Pi / 6.0, 1e-2));
    }
    SECTION("acos([0, 0.5]) -> contains [pi/3, pi/2]")
    {
        auto r = make({ 0, 0.5 }, Operon::BuiltinOp::Acos);
        REQUIRE(Contains(r, Pi / 3.0, Pi / 2.0, 1e-2));
    }
    SECTION("atan([0, 1]) -> contains [0, pi/4]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Atan);
        REQUIRE(Contains(r, 0.0, Pi / 4.0, 1e-2));
    }
    SECTION("sinh([0, 1]) -> contains [0, sinh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Sinh);
        REQUIRE(Contains(r, 0.0, std::sinh(1.0), 1e-2));
    }
    SECTION("cosh([0, 1]) -> contains [1, cosh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Cosh);
        REQUIRE(Contains(r, 1.0, std::cosh(1.0), 1e-2));
    }
    SECTION("tanh([0, 1]) -> contains [0, tanh(1)]")
    {
        auto r = make({ 0, 1 }, Operon::BuiltinOp::Tanh);
        REQUIRE(Contains(r, 0.0, std::tanh(1.0), 1e-2));
    }
}

TEST_CASE("Interval backend: nested tree", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 }, X3 { 3 };
    // (X1 + X2) * X3
    Operon::Vector<Operon::Node> ns { Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(), Var(X3),
        Util::MakeOp<Operon::BuiltinOp::Mul>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();

    auto d = Domains();
    d[X1] = { S { 1 }, S { 2 } };
    d[X2] = { S { 3 }, S { 4 } };
    d[X3] = { S { 5 }, S { 6 } };
    IE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients());
    // (X1+X2) in [4,6]; * X3 in [5,6] -> [20, 36]
    REQUIRE(Contains(r, 20.0, 36.0, 1e-3));
}

TEST_CASE("Interval backend: abs and composable ops", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t;
        t = Operon::Tree({ Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1) }).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = { S { dom.first }, S { dom.second } };
        IE eval(&t, std::move(dm));
        return { eval, Eval(eval, eval.GetTree()->GetCoefficients()) };
    };

    SECTION("abs([-3, -1]) -> [1, 3]")
    {
        auto [eval, r] = make({ -3, -1 }, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 1.0, 3.0, 1e-4));
    }
    SECTION("abs([-1, 2]) -> [0, 2]")
    {
        auto [eval, r] = make({ -1, 2 }, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-4));
    }
    SECTION("abs([2, 5]) -> [2, 5]")
    {
        auto [eval, r] = make({ 2, 5 }, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }
    SECTION("sqrtabs([-1, 4]) -> [0, 2]")
    {
        auto [eval, r] = make({ -1, 4 }, Operon::BuiltinOp::Sqrtabs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-3));
    }
    SECTION("logabs([-3, -1]) -> [0, log(3)]")
    {
        auto [eval, r] = make({ -3, -1 }, Operon::BuiltinOp::Logabs);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= std::log(3.0));
    }
}

TEST_CASE("Interval backend: fmin/fmax", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    auto d = Domains();
    d[X1] = { S { 1 }, S { 3 } };
    d[X2] = { S { 2 }, S { 4 } };

    auto make = [&](Operon::Vector<Operon::Node> ns) -> IE::Interval {
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        IE eval(&tree, IE::DomainMap { d });
        return Eval(eval, tree.GetCoefficients());
    };

    SECTION("fmin(X1, X2) -> [1, 3]")
    {
        // min([1,3], [2,4]) = [min(1,2), min(3,4)] = [1, 3]
        auto r = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmin>() });
        REQUIRE(Contains(r, 1.0, 3.0, 1e-5));
    }
    SECTION("fmax(X1, X2) -> [2, 4]")
    {
        // max([1,3], [2,4]) = [max(1,2), max(3,4)] = [2, 4]
        auto r = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmax>() });
        REQUIRE(Contains(r, 2.0, 4.0, 1e-5));
    }
}

TEST_CASE("Interval backend: aq", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    // aq(X1, X2) = X1 / sqrt(1 + X2^2)
    // X1 in [3, 6], X2 in [0, 2] -> 1+X2^2 in [1, 5] -> sqrt in [1, sqrt(5)]
    // -> aq in [3/sqrt(5), 6/1] = [1.34, 6]
    Operon::Vector<Operon::Node> ns { Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Aq>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 3 }, S { 6 } };
    d[X2] = { S { 0 }, S { 2 } };
    IE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients());
    REQUIRE(r.inf() <= 3.0 / std::sqrt(5.0) + 1e-3);
    REQUIRE(r.sup() + 1e-3 >= 6.0);
}

TEST_CASE("Affine backend: aq", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    // aq(X1, X2) = X1 / sqrt(1 + X2^2)
    // X1 in [3, 6], X2 in [1, 2] -> 1+X2^2 in [2, 5] -> sqrt in [sqrt(2), sqrt(5)]
    // -> aq in [3/sqrt(5), 6/sqrt(2)] = [1.34, 4.24]
    // Note: X2 domain [0, 2] would cause affine overestimation in X2^2
    // (interval [-1, 4] instead of [0, 4]), making 1+X2^2 contain 0 and
    // triggering inv() to throw. This is a known affine arithmetic limitation.
    Operon::Vector<Operon::Node> ns { Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Aq>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 3 }, S { 6 } };
    d[X2] = { S { 1 }, S { 2 } };
    AE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients());
    auto const iv = r.to_interval();
    // aq enclosure must contain the true range [3/sqrt(5), 6/sqrt(2)]
    REQUIRE(iv.inf() <= 3.0 / std::sqrt(5.0) + 1e-3);
    REQUIRE(iv.sup() + 1e-3 >= 6.0 / std::sqrt(2.0));
}

TEST_CASE("Affine backend: abs (non-zero-crossing)", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 };

    SECTION("abs of positive domain -> identity")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 2 }, S { 5 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of negative domain -> negation")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -5 }, S { -2 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of zero-crossing domain -> sound enclosure")
    {
        // Now supported via pappus affine abs (Chebyshev V-shape).
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -1 }, S { 2 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // abs([-1, 2]) = [0, 2]; affine enclosure must contain this.
        REQUIRE(Contains(r, 0.0, 2.0, 1e-3));
    }
}

TEST_CASE("Interval backend: cbrt, log1p, floor, ceil", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> IE::Interval {
        auto tree = Operon::Tree({ Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1) }).UpdateNodes();
        auto dm = Domains();
        dm[X1] = { S { dom.first }, S { dom.second } };
        IE eval(&tree, std::move(dm));
        return Eval(eval, tree.GetCoefficients());
    };

    SECTION("cbrt([-1, 8]) -> [-1, 2]")
    {
        auto r = make({ -1, 8 }, Operon::BuiltinOp::Cbrt);
        REQUIRE(Contains(r, -1.0, 2.0, 1e-4));
    }
    SECTION("cbrt([1, 27]) -> [1, 3]")
    {
        auto r = make({ 1, 27 }, Operon::BuiltinOp::Cbrt);
        REQUIRE(Contains(r, 1.0, 3.0, 1e-4));
    }
    SECTION("log1p([0, e-1]) -> [0, 1]")
    {
        auto r = make({ 0, std::exp(1.0) - 1.0 }, Operon::BuiltinOp::Log1p);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= 1.0);
    }
    SECTION("floor([-1.5, 8.7]) -> [-2, 8]")
    {
        auto r = make({ -1.5, 8.7 }, Operon::BuiltinOp::Floor);
        REQUIRE(Contains(r, -2.0, 8.0, 1e-5));
    }
    SECTION("ceil([-1.5, 8.7]) -> [-1, 9]")
    {
        auto r = make({ -1.5, 8.7 }, Operon::BuiltinOp::Ceil);
        REQUIRE(Contains(r, -1.0, 9.0, 1e-5));
    }
}

TEST_CASE("Affine backend: cbrt, log1p, floor, ceil, fmin, fmax", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };

    SECTION("cbrt([1, 27]) -> contains [1, 3]")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Cbrt>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 27 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 3.0, 1e-3));
    }
    SECTION("log1p([0, 1]) -> contains [0, log(2)]")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Log1p>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 0 }, S { 1 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        auto const iv = r.to_interval();
        REQUIRE(iv.inf() <= 0.0 + 1e-4);
        REQUIRE(iv.sup() + 1e-3 >= std::log(2.0));
    }
    SECTION("fmin([1,3], [2,4]) -> contains [1, 3]")
    {
        auto tree = Operon::Tree({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmin>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 3 } };
        d[X2] = { S { 2 }, S { 4 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 3.0, 1e-3));
    }
    SECTION("fmax([1,3], [2,4]) -> contains [2, 4]")
    {
        auto tree = Operon::Tree({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmax>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 3 } };
        d[X2] = { S { 2 }, S { 4 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 4.0, 1e-3));
    }
    SECTION("floor([-1.5, 8.7]) -> contains [-2, 8]")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Floor>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -1.5 }, S { 8.7 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, -2.0, 8.0, 1e-3));
    }
    SECTION("ceil([-1.5, 8.7]) -> contains [-1, 9]")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Ceil>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -1.5 }, S { 8.7 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, -1.0, 9.0, 1e-3));
    }
}

TEST_CASE("Interval backend: missing domain returns an error", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };
    auto tree = Operon::Tree({ Var(X1) }).UpdateNodes();
    IE eval(&tree, Domains()); // no domain for X1
    REQUIRE_FALSE(eval(tree.GetCoefficients()));
}

TEST_CASE("Affine backend: arithmetic", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    auto d = Domains();
    d[X1] = { S { 1 }, S { 2 } };
    d[X2] = { S { 3 }, S { 4 } };

    auto make = [&](Operon::Vector<Operon::Node> ns) -> AE {
        static Operon::Tree t;
        t = Operon::Tree(std::move(ns)).UpdateNodes(); // NOLINT
        return AE(&t, AE::DomainMap { d });
    };

    SECTION("X1 + X2 (exact in affine) -> [4, 6]")
    {
        auto eval = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-4));
    }

    SECTION("X1 * X2 (affine enclosure contains [3, 8])")
    {
        auto eval = make({ Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-3));
    }

    SECTION("X1 - X2 -> contains [-3, -1]")
    {
        auto eval = make({ Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Sub>() });
        auto const r = Eval(eval, eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -3.0, -1.0, 1e-4));
    }
}

TEST_CASE("Affine backend: shared context", "[pappus][affine]")
{
    // Re-evaluating the same tree twice reuses the evaluator's context;
    // no mixed-context exception should arise.
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    Operon::Vector<Operon::Node> ns { Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(),
        Util::MakeOp<Operon::BuiltinOp::Exp>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 0 }, S { 1 } };
    d[X2] = { S { 0 }, S { 1 } };
    AE eval(&tree, std::move(d));

    auto const c0 = Eval(eval, tree.GetCoefficients());
    auto const c1 = Eval(eval, tree.GetCoefficients());
    REQUIRE(c0.to_interval().inf() <= c1.to_interval().sup() + 1e-3);
    REQUIRE(c1.to_interval().inf() <= c0.to_interval().sup() + 1e-3);
}

TEST_CASE("Affine backend: no index reuse across evaluations", "[pappus][affine]")
{
    // Soundness: two Evaluate calls must not reuse noise-symbol indices.
    // If they did, subtracting results from independent evaluations would
    // falsely cancel (giving [0,0] instead of the true range [-2, 2]).
    constexpr Operon::Hash X1 { 1 };
    auto tree = Operon::Tree({ Var(X1) }).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 1 }, S { 3 } };
    AE eval(&tree, std::move(d));

    auto const r1 = Eval(eval, tree.GetCoefficients());
    auto const r2 = Eval(eval, tree.GetCoefficients());
    // r1 and r2 are from independent evaluations; r1 - r2 must enclose [-2, 2].
    auto const diff = pappus::ops::sub<Scalar>(r1, r2);
    auto const iv = diff.to_interval();
    REQUIRE(iv.inf() <= -2.0 + 1e-4);
    REQUIRE(iv.sup() + 1e-4 >= 2.0);
    // And must NOT be [0, 0] (the unsound result from index reuse).
    REQUIRE(iv.sup() - iv.inf() > 1.0);
}

TEST_CASE("Affine backend: inv over zero-containing domain is NaN-poisoned, not thrown", "[pappus][affine]")
{
    // affine_form::inv() no longer throws on a zero-crossing domain (see
    // affine_form::invalid() in affine.hpp for why) -- it returns a
    // NaN-centered form instead, which to_interval() surfaces as NaN bounds.
    constexpr Operon::Hash X1 { 1 };
    auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
    node.Arity = 1;
    auto tree = Operon::Tree({ Var(X1), node }).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { -1 }, S { 1 } }; // contains zero
    AE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients()).to_interval();
    REQUIRE(std::isnan(r.inf()));
    REQUIRE(std::isnan(r.sup()));
}

// ---------------------------------------------------------------------------
// Phase 7: Validation tests
// ---------------------------------------------------------------------------

TEST_CASE("Affine backend: mixed context throws", "[pappus][affine]")
{
    // Two AffineEvaluators own independent contexts. Combining forms from
    // different contexts must throw — pappus rejects this by design.
    constexpr Operon::Hash X1 { 1 };
    auto tree = Operon::Tree({ Var(X1) }).UpdateNodes();

    auto d1 = Domains();
    d1[X1] = { S { 1 }, S { 3 } };
    auto d2 = Domains();
    d2[X1] = { S { 2 }, S { 4 } };

    AE eval1(&tree, std::move(d1));
    AE eval2(&tree, std::move(d2));

    auto const r1 = Eval(eval1, tree.GetCoefficients());
    auto const r2 = Eval(eval2, tree.GetCoefficients());
    // Combining forms from different contexts must throw.
    REQUIRE_THROWS_AS(pappus::ops::add<Scalar>(r1, r2), std::invalid_argument);
}

TEST_CASE("Interval backend: empty interval propagation", "[pappus][interval]")
{
    // Out-of-domain operations (log of negative, sqrt of negative) return
    // interval::empty() (NaN bounds). Empty propagates silently through
    // subsequent ops. The caller must check is_empty().
    constexpr Operon::Hash X1 { 1 };

    SECTION("log of negative domain -> empty")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Log>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -3 }, S { -1 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.is_empty());
        REQUIRE(std::isnan(r.inf()));
        REQUIRE(std::isnan(r.sup()));
    }

    SECTION("sqrt of negative domain -> empty")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Sqrt>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -4 }, S { -1 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }

    SECTION("empty propagates through exp")
    {
        // exp(log(x)) for x in [-3, -1]: log returns empty, exp(empty) = empty
        Operon::Vector<Operon::Node> ns { Var(X1), Util::MakeOp<Operon::BuiltinOp::Log>(),
            Util::MakeOp<Operon::BuiltinOp::Exp>() };
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -3 }, S { -1 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }

    SECTION("log1p of domain below -1 -> empty")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Log1p>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -3 }, S { -2 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }
}

TEST_CASE("Interval backend: infinite bounds", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };

    SECTION("inv of zero-containing domain -> infinite")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
        node.Arity = 1;
        auto tree = Operon::Tree({ Var(X1), node }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -1 }, S { 1 } }; // contains zero
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.is_infinite());
        REQUIRE(r.inf() == -std::numeric_limits<S>::infinity());
        REQUIRE(r.sup() == +std::numeric_limits<S>::infinity());
    }

    SECTION("exp of unbounded domain -> unbounded upper")
    {
        // exp([0, inf]) = [1, inf]
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Exp>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 0 }, S { std::numeric_limits<S>::infinity() } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() <= 1.0 + 1e-4);
        REQUIRE(r.sup() == std::numeric_limits<S>::infinity());
    }
}

TEST_CASE("Interval backend: single-point (degenerate) intervals", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 };

    SECTION("constant domain [c, c]")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Square>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 2 }, S { 2 } }; // point interval
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(4.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(4.0).margin(1e-4));
    }

    SECTION("sqrt of point interval")
    {
        auto tree = Operon::Tree({ Var(X1), Util::MakeOp<Operon::BuiltinOp::Sqrt>() }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 9 }, S { 9 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(3.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(3.0).margin(1e-4));
    }
}

TEST_CASE("Interval backend: n-ary Fmin/Fmax with arity > 2", "[pappus][interval]")
{
    // Verify the n-ary fold handles more than 2 children (the affine evaluator
    // had a binary-only bug; this confirms the interval side is correct).
    constexpr Operon::Hash X1 { 1 }, X2 { 2 }, X3 { 3 };

    SECTION("ternary fmin")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmin>();
        node.Arity = 3;
        auto tree = Operon::Tree({ Var(X1), Var(X2), Var(X3), node }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 5 } };
        d[X2] = { S { 2 }, S { 4 } };
        d[X3] = { S { 3 }, S { 6 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // min([1,5], [2,4], [3,6]) = [min(1,2,3), min(5,4,6)] = [1, 4]
        REQUIRE(Contains(r, 1.0, 4.0, 1e-5));
    }

    SECTION("ternary fmax")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmax>();
        node.Arity = 3;
        auto tree = Operon::Tree({ Var(X1), Var(X2), Var(X3), node }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 5 } };
        d[X2] = { S { 2 }, S { 4 } };
        d[X3] = { S { 3 }, S { 6 } };
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // max([1,5], [2,4], [3,6]) = [max(1,2,3), max(5,4,6)] = [3, 6]
        REQUIRE(Contains(r, 3.0, 6.0, 1e-5));
    }
}

TEST_CASE("Affine backend: n-ary Fmin/Fmax with arity > 2", "[pappus][affine]")
{
    // Regression for the binary-only Fmin/Fmax bug. With arity=3, the
    // evaluator must fold all three children, not just the first two.
    constexpr Operon::Hash X1 { 1 }, X2 { 2 }, X3 { 3 };

    SECTION("ternary fmin")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmin>();
        node.Arity = 3;
        auto tree = Operon::Tree({ Var(X1), Var(X2), Var(X3), node }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 5 } };
        d[X2] = { S { 2 }, S { 4 } };
        d[X3] = { S { 3 }, S { 6 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // min enclosure must contain [1, 4]
        REQUIRE(Contains(r, 1.0, 4.0, 1e-3));
    }

    SECTION("ternary fmax")
    {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmax>();
        node.Arity = 3;
        auto tree = Operon::Tree({ Var(X1), Var(X2), Var(X3), node }).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 1 }, S { 5 } };
        d[X2] = { S { 2 }, S { 4 } };
        d[X3] = { S { 3 }, S { 6 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // max enclosure must contain [3, 6]
        REQUIRE(Contains(r, 3.0, 6.0, 1e-3));
    }
}

namespace {
    using Op = Operon::BuiltinOp;

    struct NaryOutcome {
        S lo;
        S hi;
        S maxAbsCenter;
    };

    // Variables X1..X5 with distinct, strictly positive domains, so every n-ary op (including Div) is well defined
    // and a swapped operand order changes the enclosure.
    auto NaryDomains() -> AE::DomainMap
    {
        return AE::DomainMap { { 1, { S { 8 }, S { 10 } } }, { 2, { S { 2 }, S { 3 } } },
            { 3, { S { 1.5F }, S { 2 } } }, { 4, { S { 4 }, S { 5 } } }, { 5, { S { 1 }, S { 1.25F } } } };
    }

    constexpr std::array<double, 6> NaryWeights { 0, 2.0, 0.5, 1.5, 3.0, 0.25 };
    constexpr S NaryOpWeight { 1.5F };

    // `operands` are in expression order (leftmost first); the postfix tree stores them reversed.
    auto EvalNary(Op op, std::vector<Operon::Node> operands, AE::DomainMap domains, S opWeight) -> NaryOutcome
    {
        Operon::Vector<Operon::Node> nodes(operands.rbegin(), operands.rend());
        auto fn = Operon::Node::Function(static_cast<Operon::Hash>(op), static_cast<uint16_t>(operands.size()));
        fn.Value = opWeight;
        nodes.push_back(fn);
        auto tree = Operon::Tree(std::move(nodes)).UpdateNodes();
        AE eval(&tree, std::move(domains));
        auto const r = Eval(eval, tree.GetCoefficients()).to_interval();
        return { r.inf(), r.sup(), eval.MaxAbsCenter() };
    }

    auto EvalVars(Op op, std::vector<int> const& vars, bool weighted) -> NaryOutcome
    {
        std::vector<Operon::Node> operands;
        for (auto v : vars) {
            operands.push_back(Var(static_cast<Operon::Hash>(v), weighted ? NaryWeights[v] : 1.0));
        }
        return EvalNary(op, std::move(operands), NaryDomains(), weighted ? NaryOpWeight : S { 1 });
    }

    auto EvalConsts(Op op, std::vector<double> const& values) -> NaryOutcome
    {
        std::vector<Operon::Node> operands;
        for (auto v : values) {
            operands.push_back(Const(v));
        }
        return EvalNary(op, std::move(operands), AE::DomainMap {}, S { 1 });
    }

    struct NaryRow {
        Op op;
        std::vector<int> vars; // expression order, leftmost operand first
        bool weighted;
        NaryOutcome expected;
    };

    // Scalar semantics of an n-ary op on already-weighted operand values.
    auto NaryReference(Op op, std::vector<double> const& v) -> double
    {
        if (v.size() == 1) {
            return op == Op::Sub ? -v[0] : (op == Op::Div ? 1.0 / v[0] : v[0]);
        }
        auto acc = v[0];
        for (std::size_t k = 1; k < v.size(); ++k) {
            switch (op) {
            case Op::Add:
                acc += v[k];
                break;
            case Op::Mul:
                acc *= v[k];
                break;
            case Op::Sub:
                acc -= v[k];
                break;
            case Op::Div:
                acc /= v[k];
                break;
            case Op::Fmin:
                acc = std::fmin(acc, v[k]);
                break;
            default:
                acc = std::fmax(acc, v[k]);
                break;
            }
        }
        return acc;
    }

    // The pinned endpoints/centers were captured with single-precision Scalar. They are matched to within 2 ULP there
    // (compiler/-march FMA contraction can move a last bit); for any other Scalar they only serve as a loose sanity
    // anchor, and correctness rests on the corner-containment and structural checks.
    auto PinMatches(S actual, S pinned) -> bool
    {
        if constexpr (std::is_same_v<S, float>) {
            auto lo = pinned;
            auto hi = pinned;
            for (int k = 0; k < 2; ++k) {
                lo = std::nextafter(lo, -std::numeric_limits<S>::infinity());
                hi = std::nextafter(hi, std::numeric_limits<S>::infinity());
            }
            return actual >= lo && actual <= hi;
        } else {
            auto const a = static_cast<double>(actual);
            auto const p = static_cast<double>(pinned);
            return std::fabs(a - p) <= 1e-4 * (1.0 + std::fabs(p));
        }
    }

    void CheckPinned(NaryOutcome const& actual, NaryOutcome const& expected)
    {
        CHECK(PinMatches(actual.lo, expected.lo));
        CHECK(PinMatches(actual.hi, expected.hi));
        CHECK(PinMatches(actual.maxAbsCenter, expected.maxAbsCenter));
    }

    // Evaluates `row` and checks (1) the pinned enclosure and MaxAbsCenter, and (2) that the enclosure is
    // semantically right: it contains the scalar result at every corner of the domain box (so a wrong operand order
    // or a dropped child cannot be pinned by accident) and, for the linear ops, is exact.
    void CheckNaryRow(NaryRow const& row)
    {
        auto const actual = EvalVars(row.op, row.vars, row.weighted);
        CheckPinned(actual, row.expected);

        auto const domains = NaryDomains();
        auto const opWeight = row.weighted ? static_cast<double>(NaryOpWeight) : 1.0;
        auto refLo = std::numeric_limits<double>::infinity();
        auto refHi = -refLo;
        for (unsigned mask = 0; mask < 32; ++mask) {
            std::vector<double> values;
            for (auto v : row.vars) {
                auto const& [lo, hi] = domains.at(static_cast<Operon::Hash>(v));
                auto const x = static_cast<double>(((mask >> (v - 1)) & 1U) != 0 ? hi : lo);
                values.push_back((row.weighted ? NaryWeights[v] : 1.0) * x);
            }
            auto const ref = opWeight * NaryReference(row.op, values);
            auto const tol = 1e-4 * (1.0 + std::fabs(ref));
            CHECK(static_cast<double>(actual.lo) <= ref + tol);
            CHECK(static_cast<double>(actual.hi) + tol >= ref);
            refLo = std::min(refLo, ref);
            refHi = std::max(refHi, ref);
        }
        if (row.op == Op::Add || row.op == Op::Sub) {
            auto const tol = 1e-4 * (1.0 + std::fabs(refLo) + std::fabs(refHi));
            CHECK(std::fabs(static_cast<double>(actual.lo) - refLo) <= tol);
            CHECK(std::fabs(static_cast<double>(actual.hi) - refHi) <= tol);
        }
    }
} // namespace

TEST_CASE("Affine backend: n-ary Add/Mul/Sub/Div/Fmin/Fmax, arity 1..5", "[pappus][affine][nary]")
{
    // Pinned from the evaluator's output; expression order is x1 op x2 op ... op xn, left-associated.
    static std::vector<NaryRow> const rows {
        { Op::Add, { 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Add, { 1, 2 }, false, { 0x1.4p+3F, 0x1.ap+3F, 0x1.7p+3F } },
        { Op::Add, { 1, 2, 3 }, false, { 0x1.7p+3F, 0x1.ep+3F, 0x1.a8p+3F } },
        { Op::Add, { 1, 2, 3, 4 }, false, { 0x1.fp+3F, 0x1.4p+4F, 0x1.1cp+4F } },
        { Op::Add, { 1, 2, 3, 4, 5 }, false, { 0x1.08p+4F, 0x1.54p+4F, 0x1.2ep+4F } },
        { Op::Mul, { 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Mul, { 1, 2 }, false, { 0x1.ep+3F, 0x1.ep+4F, 0x1.68p+4F } },
        { Op::Mul, { 1, 2, 3 }, false, { 0x1.2cp+4F, 0x1.ep+5F, 0x1.3bp+5F } },
        { Op::Mul, { 1, 2, 3, 4 }, false, { 0x1.b3p+5F, 0x1.2cp+8F, 0x1.626p+7F } },
        { Op::Mul, { 1, 2, 3, 4, 5 }, false, { 0x1.7acp+4F, 0x1.77p+8F, 0x1.8eacp+7F } },
        { Op::Sub, { 1 }, false, { -0x1.4p+3F, -0x1p+3F, 0x1.2p+3F } },
        { Op::Sub, { 1, 2 }, false, { 0x1.4p+2F, 0x1p+3F, 0x1.2p+3F } },
        { Op::Sub, { 1, 2, 3 }, false, { 0x1.8p+1F, 0x1.ap+2F, 0x1.2p+3F } },
        { Op::Sub, { 1, 2, 3, 4 }, false, { -0x1p+1F, 0x1.4p+1F, 0x1.2p+3F } },
        { Op::Sub, { 1, 2, 3, 4, 5 }, false, { -0x1.ap+1F, 0x1.8p+0F, 0x1.2p+3F } },
        { Op::Div, { 1 }, false, { 0x1.93e47cp-4F, 0x1.00001cp-3F, 0x1.2p+3F } },
        { Op::Div, { 1, 2 }, false, { 0x1.2c9a58p+1F, 0x1.400022p+2F, 0x1.2p+3F } },
        { Op::Div, { 1, 2, 3 }, false, { 0x1.d18f2cp-1F, 0x1.aaab0ap+1F, 0x1.2p+3F } },
        { Op::Div, { 1, 2, 3, 4 }, false, { 0x1.d874ap-4F, 0x1.aaab36p-1F, 0x1.2p+3F } },
        { Op::Div, { 1, 2, 3, 4, 5 }, false, { 0x1.f1b58p-7F, 0x1.aaab68p-1F, 0x1.2p+3F } },
        { Op::Fmin, { 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 2 }, false, { 0x1p+1F, 0x1.8p+1F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 2, 3 }, false, { 0x1.8p+0F, 0x1p+1F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 2, 3, 4 }, false, { 0x1.8p+0F, 0x1p+1F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 2, 3, 4, 5 }, false, { 0x1p+0F, 0x1.4p+0F, 0x1.2p+3F } },
        { Op::Fmax, { 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmax, { 1, 2 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmax, { 1, 2, 3 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmax, { 1, 2, 3, 4 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmax, { 1, 2, 3, 4, 5 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        // Per-node coefficients: weighted variables and a weighted op node.
        { Op::Add, { 1, 2, 3 }, true, { 0x1.cep+4F, 0x1.26p+5F, 0x1.068p+5F } },
        { Op::Add, { 1, 2, 3, 4, 5 }, true, { 0x1.7ap+5F, 0x1.ddcp+5F, 0x1.abep+5F } },
        { Op::Mul, { 1, 2, 3 }, true, { 0x1.518p+5F, 0x1.0ep+7F, 0x1.626p+6F } },
        { Op::Mul, { 1, 2, 3, 4, 5 }, true, { 0x1.3f92p+5F, 0x1.3c68p+9F, 0x1.8eacp+9F } },
        { Op::Sub, { 1, 2, 3 }, true, { 0x1.14p+4F, 0x1.92p+4F, 0x1.53p+4F } },
        { Op::Sub, { 1, 2, 3, 4, 5 }, true, { -0x1.6ep+2F, 0x1.bp+2F, 0x1.2p+4F } },
        { Op::Div, { 1, 2, 3 }, true, { 0x1.d18f34p+1F, 0x1.aaab08p+3F, 0x1.2p+4F } },
        { Op::Div, { 1, 2, 3, 4, 5 }, true, { 0x1.4bce8p-4F, 0x1.1c7242p+2F, 0x1.2p+4F } },
        { Op::Fmin, { 1, 2, 3 }, true, { 0x1.8p+0F, 0x1.2p+1F, 0x1.2p+4F } },
        { Op::Fmin, { 1, 2, 3, 4, 5 }, true, { 0x1.8p-2F, 0x1.ep-2F, 0x1.2p+4F } },
        { Op::Fmax, { 1, 2, 3 }, true, { 0x1.8p+4F, 0x1.ep+4F, 0x1.bp+4F } },
        { Op::Fmax, { 1, 2, 3, 4, 5 }, true, { 0x1.8p+4F, 0x1.ep+4F, 0x1.bp+4F } },
    };

    for (auto const& row : rows) {
        INFO("op=" << static_cast<int>(row.op) << " arity=" << row.vars.size() << " weighted=" << row.weighted);
        CheckNaryRow(row);
    }
}

TEST_CASE("Affine backend: n-ary ops with a repeated variable", "[pappus][affine][nary]")
{
    // A repeated variable occurrence shares one noise symbol, so x - x must collapse to exactly [0, 0] and
    // x1 - x2 - x1 to exactly -x2, regardless of where the repeat sits in the fold.
    static std::vector<NaryRow> const rows {
        { Op::Sub, { 1, 1 }, false, { 0x0p+0F, 0x0p+0F, 0x1.2p+3F } },
        { Op::Sub, { 1, 1, 1 }, false, { -0x1.4p+3F, -0x1p+3F, 0x1.2p+3F } },
        { Op::Sub, { 1, 2, 1 }, false, { -0x1.8p+1F, -0x1p+1F, 0x1.2p+3F } },
        { Op::Add, { 1, 1, 1 }, false, { 0x1.8p+4F, 0x1.ep+4F, 0x1.bp+4F } },
        { Op::Add, { 1, 2, 1 }, false, { 0x1.2p+4F, 0x1.7p+4F, 0x1.48p+4F } },
        { Op::Mul, { 1, 1, 1 }, false, { 0x1.e5p+8F, 0x1.f4p+9F, 0x1.734p+9F } },
        { Op::Mul, { 1, 2, 1 }, false, { 0x1.aep+6F, 0x1.2cp+8F, 0x1.978p+7F } },
        { Op::Div, { 1, 1 }, false, { 0x1.f8dd9cp-1F, 0x1.038ea4p+0F, 0x1.2p+3F } },
        { Op::Div, { 1, 1, 1 }, false, { 0x1.8ccbc4p-4F, 0x1.038ecp-3F, 0x1.2p+3F } },
        { Op::Div, { 1, 2, 1 }, false, { 0x1.26c71p-2F, 0x1.0ea636p-1F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 1, 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmax, { 1, 1 }, false, { 0x1p+3F, 0x1.4p+3F, 0x1.2p+3F } },
        { Op::Fmin, { 1, 2, 1 }, false, { 0x1p+1F, 0x1.8p+1F, 0x1.2p+3F } },
    };

    // x - x collapses to exactly [0, 0] at any precision.
    auto const cancelled = EvalVars(Op::Sub, { 1, 1 }, false);
    CHECK(cancelled.lo == S { 0 });
    CHECK(cancelled.hi == S { 0 });

    for (auto const& row : rows) {
        INFO("op=" << static_cast<int>(row.op) << " vars=" << row.vars.size() << " first=" << row.vars.front());
        CheckNaryRow(row);
    }
}

TEST_CASE("Affine backend: MaxAbsCenter sees n-ary accumulator intermediates", "[pappus][affine][nary]")
{
    // Each fold's running accumulator exceeds every leaf and the final result, so MaxAbsCenter() only reports the
    // right value if the center is tracked after every accumulation step, not just for emitted node values.
    struct ConstRow {
        Op op;
        std::vector<double> values;
        NaryOutcome expected;
        S intermediate;
    };
    static std::vector<ConstRow> const rows {
        { Op::Add, { 600, 600, -1100 }, { 0x1.9p+6F, 0x1.9p+6F, 0x1.2cp+10F }, S { 1200 } },
        { Op::Sub, { 600, -600, 1100 }, { 0x1.9p+6F, 0x1.9p+6F, 0x1.2cp+10F }, S { 1200 } },
        { Op::Mul, { 100, 100, 0.001 }, { 0x1.3ffffep+3F, 0x1.400002p+3F, 0x1.388p+13F }, S { 10000 } },
        { Op::Div, { 2, 0.001, 10 }, { 0x1.8ffffap+7F, 0x1.900002p+7F, 0x1.f4p+10F }, S { 2000 } },
    };

    for (auto const& row : rows) {
        INFO("op=" << static_cast<int>(row.op));
        auto const actual = EvalConsts(row.op, row.values);
        CheckPinned(actual, row.expected);
        CHECK(actual.maxAbsCenter == Catch::Approx(row.intermediate).epsilon(1e-6));
    }
}

TEST_CASE("Affine backend: max_terms condensation", "[pappus][affine]")
{
    // Verify that max_terms > 0 triggers condensation, keeping the form's
    // term count bounded. With max_terms=0 (default), terms grow unbounded.
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    // A tree that generates many noise terms: X1 * X2 * X1 * X2 (via Mul chain)
    Operon::Vector<Operon::Node> ns { Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>(), Var(X1),
        Util::MakeOp<Operon::BuiltinOp::Mul>(), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 1 }, S { 2 } };
    d[X2] = { S { 1 }, S { 2 } };

    SECTION("max_terms=0 (unbounded) — terms grow freely")
    {
        AE eval(&tree, std::move(d), /*maxTerms=*/0);
        auto const r = Eval(eval, tree.GetCoefficients());
        // The result is a valid enclosure of [1, 16]
        REQUIRE(Contains(r, 1.0, 16.0, 1e-3));
    }

    SECTION("max_terms=2 — form is condensed to <= 2 terms + remainder")
    {
        auto d2 = Domains();
        d2[X1] = { S { 1 }, S { 2 } };
        d2[X2] = { S { 1 }, S { 2 } };
        AE eval(&tree, std::move(d2), /*maxTerms=*/2);
        auto const r = Eval(eval, tree.GetCoefficients());
        // Still a valid (but potentially looser) enclosure of [1, 16]
        REQUIRE(Contains(r, 1.0, 16.0, 1e-2));
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Pow/Powabs child ordering — asymmetric domain detects j/k swap
// ---------------------------------------------------------------------------
// Post-order layout {X2, X1, Pow}: j = i-1 = index of X1 (LEFT = base),
// k = j - (nodes[j].Length+1) = index of X2 (RIGHT = exponent).
// Using asymmetric domains: pow([2,3],[1,2]) ⊇ [2,9].
// Swapped: pow([1,2],[2,3]) ⊇ [1,8] — upper bound 8, not 9 — detects the bug.

TEST_CASE("Interval backend: pow/powabs child ordering", "[pappus][interval]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };

    SECTION("pow(base=[2,3], exp=[1,2]) -> contains [2, 9]")
    {
        // {X2, X1, Pow}: X1 is base (j=i-1), X2 is exponent (k).
        Operon::Vector<Operon::Node> ns { Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>() };
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 2 }, S { 3 } }; // base
        d[X2] = { S { 1 }, S { 2 } }; // exponent
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // pow([2,3],[1,2]) = [2^1, 3^2] = [2, 9]
        REQUIRE(r.inf() <= 2.0 + 1e-3);
        REQUIRE(r.sup() + 1e-2 >= 9.0);
    }

    SECTION("powabs(base=[-3,2], exp=[2,3]) -> contains [0, 27]")
    {
        // {X2, X1, Powabs}: abs(X1) is base, X2 is exponent.
        Operon::Vector<Operon::Node> ns { Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Powabs>() };
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -3 }, S { 2 } }; // abs → [0, 3]
        d[X2] = { S { 2 }, S { 3 } }; // exponent
        IE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        // pow(abs([-3,2]), [2,3]) = pow([0,3], [2,3]) ⊇ [0, 27]
        REQUIRE(r.inf() <= 0.0 + 1e-3);
        REQUIRE(r.sup() + 1e-2 >= 27.0);
    }
}

TEST_CASE("Interval backend: pow with negative-base domain and constant even exponent", "[pappus][interval]")
{
    // A degenerate exponent interval must dispatch to pow(interval, Scalar),
    // not the general interval^interval overload, which restricts the base
    // to non-negative even for a literal even-integer exponent.
    constexpr Operon::Hash X1 { 1 };
    Operon::Vector<Operon::Node> ns { Const(2.0), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { -3 }, S { -1 } }; // always-negative base
    IE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients());
    // pow([-3,-1], 2) = [1, 9], not empty
    REQUIRE(r.inf() <= 1.0 + 1e-3);
    REQUIRE(r.sup() + 1e-2 >= 9.0);
}

TEST_CASE("Affine backend: pow/powabs child ordering", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 };

    // The general affine pow(affine_base, affine_exp) approximation (which
    // linearises both dimensions jointly) can fail to produce conservative bounds
    // for some domains. We use a constant-node exponent so the evaluator takes
    // the simpler affine::pow(T) path, which is conservative.
    //
    // Tree layout: {Const(exp), Var(X1), Pow/Powabs}
    //   j = i-1 = index of Var (LEFT semantic = base)
    //   k = j - (nodes[j].Length+1) = index of Const (RIGHT semantic = exponent)
    //
    // If j/k were swapped: pow(const, var) instead of pow(var, const).
    //   Correct: pow([2,3], 2)   -> [4, ~9.6], sup > 9
    //   Swapped: pow(2, [2,3])   -> [4, 8],    sup <= 8

    SECTION("pow(base=[2,3], exp=const_2) -> enclosure contains [4, 9]")
    {
        Operon::Vector<Operon::Node> ns { Const(2.0), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>() };
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { 2 }, S { 3 } };
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 4.0, 9.0, 1e-2));
        // Key: upper bound must reach 9, not just 8 (the swapped result).
        REQUIRE(r.to_interval().sup() + 1e-2 >= 9.0);
    }

    // Powabs: base is X1 (abs applied), exponent is const.
    // Use X1=[-3,-1] (all negative) so abs(X1)=-X1=[1,3] is exact (negation,
    // no approximation error) and min(abs(X1))=1 > 0 so pow cannot throw.
    //   Correct: pow(abs([-3,-1])=[1,3], 3) -> [1, 27]
    //   Swapped: pow(abs(const_3)=3, [-3,-1]) -> 3^[-3,-1] ≈ [1/27, 1/3]

    SECTION("powabs(base=[-3,-1], exp=const_3) -> enclosure contains [1, 27]")
    {
        Operon::Vector<Operon::Node> ns { Const(3.0), Var(X1), Util::MakeOp<Operon::BuiltinOp::Powabs>() };
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = { S { -3 }, S { -1 } }; // all-negative: abs = exact negation, min(abs) = 1 > 0
        AE eval(&tree, std::move(d));
        auto const r = Eval(eval, tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 27.0, 1e-2));
    }
}

TEST_CASE("Affine backend: pow with non-degenerate exponent falls back to interval enclosure", "[pappus][affine]")
{
    constexpr Operon::Hash X1 { 1 }, X2 { 2 };
    Operon::Vector<Operon::Node> ns { Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = { S { 2 }, S { 3 } };
    d[X2] = { S { 1 }, S { 2 } };

    AE eval(&tree, std::move(d));
    auto const r = Eval(eval, tree.GetCoefficients());
    REQUIRE(Contains(r, 2.0, 9.0, 1e-2));
}

// ---------------------------------------------------------------------------
// Phase 8: Performance benchmarks (nanobench, [pappus][performance])
// ---------------------------------------------------------------------------
// Tree: exp(X1 + X2) * X3 — mixes n-ary add, a transcendental, and mul.
//
// Phase 8a: Throughput of each backend per single Evaluate() call.
// Phase 8b: Affine term-growth profile — how many noise terms the root form
//           carries after each evaluation under max_terms=0 vs max_terms=4.
// Phase 8c: max_terms policy decision (see comment below).
// Phase 8d: aq/sqrtabs/logabs now use dedicated ops:: wrappers — 2-3 finalize
//           calls → 1 per composite node.  The tree below doesn't exercise
//           these directly; the Aq test above is the relevant regression.
//
// Scalar baseline: hand-written formula evaluated 1000× to anchor the cost
// of the arithmetic itself vs pappus overhead.

TEST_CASE("Pappus backends: evaluation throughput", "[pappus][performance]")
{
    namespace nb = ankerl::nanobench;

    constexpr Operon::Hash X1 { 1 }, X2 { 2 }, X3 { 3 };
    Operon::Vector<Operon::Node> ns { Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(),
        Util::MakeOp<Operon::BuiltinOp::Exp>(), Var(X3), Util::MakeOp<Operon::BuiltinOp::Mul>() };
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto coeffs = tree.GetCoefficients();

    auto d = Domains();
    d[X1] = { S { 0 }, S { 1 } };
    d[X2] = { S { 0 }, S { 1 } };
    d[X3] = { S { 1 }, S { 2 } };

    nb::Bench b;
    b.timeUnit(std::chrono::microseconds(1), "µs").minEpochIterations(1000);

    // Scalar baseline: exp(x1 + x2) * x3 evaluated at the domain midpoints.
    // Shows the floor cost of the arithmetic itself before pappus overhead.
    b.run("scalar midpoint (baseline)", [&]() {
        S x1 { 0.5 }, x2 { 0.5 }, x3 { 1.5 };
        nb::doNotOptimizeAway(std::exp(x1 + x2) * x3);
    });

    IE iev(&tree, IE::DomainMap { d });
    b.run("interval Evaluate", [&]() { nb::doNotOptimizeAway(iev(coeffs)); });

    // Constructing from an existing map compiles scalar bounds directly into
    // node slots. Keep the avoided input-map copy visible in this benchmark.
    b.run("interval construct (pre-copied map)", [&]() {
        IE eval(&tree, IE::DomainMap { d });
        nb::doNotOptimizeAway(eval.GetTree());
    });
    b.run("interval construct (domain snapshot)", [&]() {
        IE eval(&tree, d);
        nb::doNotOptimizeAway(eval.GetTree());
    });

    AE aev(&tree, AE::DomainMap { d });
    b.run("affine Evaluate (max_terms=0)", [&]() { nb::doNotOptimizeAway(aev(coeffs)); });

    AE aev4(&tree, AE::DomainMap { d }, /*maxTerms=*/4);
    b.run("affine Evaluate (max_terms=4)", [&]() { nb::doNotOptimizeAway(aev4(coeffs)); });

    // Phase 8b: term-growth report — call Evaluate once per config and print.
    // Not timed; just measures state after one evaluation.
    {
        AE ev_unbounded(&tree, AE::DomainMap { d }, /*maxTerms=*/0);
        auto const r_u = Eval(ev_unbounded, coeffs);
        auto const terms_u = ev_unbounded.TermCount();

        AE ev_4(&tree, AE::DomainMap { d }, /*maxTerms=*/4);
        auto const r_4 = Eval(ev_4, coeffs);
        auto const terms_4 = ev_4.TermCount();

        // Sanity: both enclosures must agree (one is tighter, one is looser).
        REQUIRE(r_u.to_interval().inf() <= r_4.to_interval().sup() + S { 1e-2 });
        // The condensed form must not have more terms than its budget.
        REQUIRE(terms_4 <= 4);
        // The unbounded form should have at least as many terms (same or more).
        // (Equal when the tree is shallow enough to stay within 4 terms anyway.)
        REQUIRE(terms_u >= terms_4);
        // Make term counts visible in verbose output.
        INFO("term count (max_terms=0): " << terms_u);
        INFO("term count (max_terms=4): " << terms_4);
    }
}

} // namespace Operon::Test
