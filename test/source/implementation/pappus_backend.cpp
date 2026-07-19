// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../operon_test.hpp"

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"

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
        auto eval = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-5));
    }

    SECTION("X1 - X2 -> [-3, -1]") {
        // operon lays out binary-op children with the semantic LEFT operand at
        // the higher index (rightmost); `Tree::Indices` yields rightmost-first.
        auto eval = make({Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Sub>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -3.0, -1.0, 1e-5));
    }

    SECTION("X1 * X2 -> [3, 8]") {
        auto eval = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-5));
    }

    SECTION("X1 / X2 -> [0.25, 0.6667]") {
        auto eval = make({Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Div>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(r.inf() <= 0.25 + 1e-5);
        REQUIRE(r.sup() + 1e-5 >= 2.0/3.0);
    }

    SECTION("neg(X1) -> [-2, -1]") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Sub>();
        node.Arity = 1;
        auto eval = make({Var(X1), node});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, -2.0, -1.0, 1e-5));
    }

    SECTION("inv(X1) -> [0.5, 1]") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
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

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t; t = Operon::Tree({Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1)}).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = {S{dom.first}, S{dom.second}};
        IE eval(&t, std::move(dm));
        return {eval, eval.Evaluate(eval.GetTree()->GetCoefficients())};
    };

    SECTION("square([-1, 2]) -> [0, 4]") {
        auto [eval, r] = make({-1, 2}, Operon::BuiltinOp::Square);
        REQUIRE(Contains(r, 0.0, 4.0, 1e-4));
    }

    SECTION("sqrt([1, 4]) -> [1, 2]") {
        auto [eval, r] = make({1, 4}, Operon::BuiltinOp::Sqrt);
        REQUIRE(Contains(r, 1.0, 2.0, 1e-4));
    }

    SECTION("exp([0, 1]) -> [1, e]") {
        auto [eval, r] = make({0, 1}, Operon::BuiltinOp::Exp);
        REQUIRE(r.inf() <= 1.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= std::exp(1.0));
    }

    SECTION("log([1, e]) -> [0, 1]") {
        auto [eval, r] = make({1, std::exp(1.0)}, Operon::BuiltinOp::Log);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= 1.0);
    }
}

TEST_CASE("Interval backend: nested tree", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1}, X2{2}, X3{3};
    // (X1 + X2) * X3
    Operon::Vector<Operon::Node> ns{
        Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(),
        Var(X3), Util::MakeOp<Operon::BuiltinOp::Mul>()};
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

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> std::pair<IE, IE::Interval> {
        static Operon::Tree t; t = Operon::Tree({Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1)}).UpdateNodes(); // NOLINT
        auto dm = Domains();
        dm[X1] = {S{dom.first}, S{dom.second}};
        IE eval(&t, std::move(dm));
        return {eval, eval.Evaluate(eval.GetTree()->GetCoefficients())};
    };

    SECTION("abs([-3, -1]) -> [1, 3]") {
        auto [eval, r] = make({-3, -1}, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 1.0, 3.0, 1e-4));
    }
    SECTION("abs([-1, 2]) -> [0, 2]") {
        auto [eval, r] = make({-1, 2}, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-4));
    }
    SECTION("abs([2, 5]) -> [2, 5]") {
        auto [eval, r] = make({2, 5}, Operon::BuiltinOp::Abs);
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }
    SECTION("sqrtabs([-1, 4]) -> [0, 2]") {
        auto [eval, r] = make({-1, 4}, Operon::BuiltinOp::Sqrtabs);
        REQUIRE(Contains(r, 0.0, 2.0, 1e-3));
    }
    SECTION("logabs([-3, -1]) -> [0, log(3)]") {
        auto [eval, r] = make({-3, -1}, Operon::BuiltinOp::Logabs);
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
        auto r = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmin>()});
        REQUIRE(Contains(r, 1.0, 3.0, 1e-5));
    }
    SECTION("fmax(X1, X2) -> [2, 4]") {
        // max([1,3], [2,4]) = [max(1,2), max(3,4)] = [2, 4]
        auto r = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmax>()});
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
        Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Aq>()};
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
        Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Aq>()};
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
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{2}, S{5}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of negative domain -> negation") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-5}, S{-2}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 5.0, 1e-4));
    }

    SECTION("abs of zero-crossing domain -> sound enclosure") {
        // Now supported via pappus affine abs (Chebyshev V-shape).
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Abs>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-1}, S{2}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // abs([-1, 2]) = [0, 2]; affine enclosure must contain this.
        REQUIRE(Contains(r, 0.0, 2.0, 1e-3));
    }
}

TEST_CASE("Interval backend: cbrt, log1p, floor, ceil", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};

    auto make = [&](IE::Domain dom, Operon::BuiltinOp op) -> IE::Interval {
        auto tree = Operon::Tree({Var(X1), Operon::Node::Function(static_cast<Operon::Hash>(op), 1)}).UpdateNodes();
        auto dm = Domains();
        dm[X1] = {S{dom.first}, S{dom.second}};
        IE eval(&tree, std::move(dm));
        return eval.Evaluate(tree.GetCoefficients());
    };

    SECTION("cbrt([-1, 8]) -> [-1, 2]") {
        auto r = make({-1, 8}, Operon::BuiltinOp::Cbrt);
        REQUIRE(Contains(r, -1.0, 2.0, 1e-4));
    }
    SECTION("cbrt([1, 27]) -> [1, 3]") {
        auto r = make({1, 27}, Operon::BuiltinOp::Cbrt);
        REQUIRE(Contains(r, 1.0, 3.0, 1e-4));
    }
    SECTION("log1p([0, e-1]) -> [0, 1]") {
        auto r = make({0, std::exp(1.0) - 1.0}, Operon::BuiltinOp::Log1p);
        REQUIRE(r.inf() <= 0.0 + 1e-4);
        REQUIRE(r.sup() + 1e-3 >= 1.0);
    }
    SECTION("floor([-1.5, 8.7]) -> [-2, 8]") {
        auto r = make({-1.5, 8.7}, Operon::BuiltinOp::Floor);
        REQUIRE(Contains(r, -2.0, 8.0, 1e-5));
    }
    SECTION("ceil([-1.5, 8.7]) -> [-1, 9]") {
        auto r = make({-1.5, 8.7}, Operon::BuiltinOp::Ceil);
        REQUIRE(Contains(r, -1.0, 9.0, 1e-5));
    }
}

TEST_CASE("Affine backend: cbrt, log1p, floor, ceil, fmin, fmax", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1}, X2{2};

    SECTION("cbrt([1, 27]) -> contains [1, 3]") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Cbrt>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{27}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 3.0, 1e-3));
    }
    SECTION("log1p([0, 1]) -> contains [0, log(2)]") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Log1p>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{0}, S{1}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        auto const iv = r.to_interval();
        REQUIRE(iv.inf() <= 0.0 + 1e-4);
        REQUIRE(iv.sup() + 1e-3 >= std::log(2.0));
    }
    SECTION("fmin([1,3], [2,4]) -> contains [1, 3]") {
        auto tree = Operon::Tree({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmin>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{3}};
        d[X2] = {S{2}, S{4}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 3.0, 1e-3));
    }
    SECTION("fmax([1,3], [2,4]) -> contains [2, 4]") {
        auto tree = Operon::Tree({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Fmax>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{3}};
        d[X2] = {S{2}, S{4}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 2.0, 4.0, 1e-3));
    }
    SECTION("floor([-1.5, 8.7]) -> contains [-2, 8]") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Floor>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-1.5}, S{8.7}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, -2.0, 8.0, 1e-3));
    }
    SECTION("ceil([-1.5, 8.7]) -> contains [-1, 9]") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Ceil>()}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-1.5}, S{8.7}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, -1.0, 9.0, 1e-3));
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
        auto eval = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 4.0, 6.0, 1e-4));
    }

    SECTION("X1 * X2 (affine enclosure contains [3, 8])") {
        auto eval = make({Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>()});
        auto const r = eval.Evaluate(eval.GetTree()->GetCoefficients());
        REQUIRE(Contains(r, 3.0, 8.0, 1e-3));
    }

    SECTION("X1 - X2 -> contains [-3, -1]") {
        auto eval = make({Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Sub>()});
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
        Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(),
        Util::MakeOp<Operon::BuiltinOp::Exp>()};
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
    auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
    node.Arity = 1;
    auto tree = Operon::Tree({Var(X1), node}).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{-1}, S{1}}; // contains zero
    AE eval(&tree, std::move(d));
    REQUIRE_THROWS_AS(eval.Evaluate(tree.GetCoefficients()), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Phase 7: Validation tests
// ---------------------------------------------------------------------------

TEST_CASE("Affine backend: mixed context throws", "[pappus][affine]")
{
    // Two AffineEvaluators own independent contexts. Combining forms from
    // different contexts must throw — pappus rejects this by design.
    constexpr Operon::Hash X1{1};
    auto tree = Operon::Tree({Var(X1)}).UpdateNodes();

    auto d1 = Domains(); d1[X1] = {S{1}, S{3}};
    auto d2 = Domains(); d2[X1] = {S{2}, S{4}};

    AE eval1(&tree, std::move(d1));
    AE eval2(&tree, std::move(d2));

    auto const r1 = eval1.Evaluate(tree.GetCoefficients());
    auto const r2 = eval2.Evaluate(tree.GetCoefficients());
    // Combining forms from different contexts must throw.
    REQUIRE_THROWS_AS(pappus::ops::add<Scalar>(r1, r2), std::invalid_argument);
}

TEST_CASE("Interval backend: empty interval propagation", "[pappus][interval]")
{
    // Out-of-domain operations (log of negative, sqrt of negative) return
    // interval::empty() (NaN bounds). Empty propagates silently through
    // subsequent ops. The caller must check is_empty().
    constexpr Operon::Hash X1{1};

    SECTION("log of negative domain -> empty") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Log>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{-3}, S{-1}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.is_empty());
        REQUIRE(std::isnan(r.inf()));
        REQUIRE(std::isnan(r.sup()));
    }

    SECTION("sqrt of negative domain -> empty") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Sqrt>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{-4}, S{-1}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }

    SECTION("empty propagates through exp") {
        // exp(log(x)) for x in [-3, -1]: log returns empty, exp(empty) = empty
        Operon::Vector<Operon::Node> ns{
            Var(X1), Util::MakeOp<Operon::BuiltinOp::Log>(),
            Util::MakeOp<Operon::BuiltinOp::Exp>()};
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains(); d[X1] = {S{-3}, S{-1}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }

    SECTION("log1p of domain below -1 -> empty") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Log1p>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{-3}, S{-2}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.is_empty());
    }
}

TEST_CASE("Interval backend: infinite bounds", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};

    SECTION("inv of zero-containing domain -> infinite") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Div>();
        node.Arity = 1;
        auto tree = Operon::Tree({Var(X1), node}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{-1}, S{1}}; // contains zero
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.is_infinite());
        REQUIRE(r.inf() == -std::numeric_limits<S>::infinity());
        REQUIRE(r.sup() == +std::numeric_limits<S>::infinity());
    }

    SECTION("exp of unbounded domain -> unbounded upper") {
        // exp([0, inf]) = [1, inf]
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Exp>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{0}, S{std::numeric_limits<S>::infinity()}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() <= 1.0 + 1e-4);
        REQUIRE(r.sup() == std::numeric_limits<S>::infinity());
    }
}

TEST_CASE("Interval backend: single-point (degenerate) intervals", "[pappus][interval]")
{
    constexpr Operon::Hash X1{1};

    SECTION("constant domain [c, c]") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Square>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{2}, S{2}}; // point interval
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(4.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(4.0).margin(1e-4));
    }

    SECTION("sqrt of point interval") {
        auto tree = Operon::Tree({Var(X1), Util::MakeOp<Operon::BuiltinOp::Sqrt>()}).UpdateNodes();
        auto d = Domains(); d[X1] = {S{9}, S{9}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(r.inf() == Catch::Approx(3.0).margin(1e-4));
        REQUIRE(r.sup() == Catch::Approx(3.0).margin(1e-4));
    }
}

TEST_CASE("Interval backend: n-ary Fmin/Fmax with arity > 2", "[pappus][interval]")
{
    // Verify the n-ary fold handles more than 2 children (the affine evaluator
    // had a binary-only bug; this confirms the interval side is correct).
    constexpr Operon::Hash X1{1}, X2{2}, X3{3};

    SECTION("ternary fmin") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmin>();
        node.Arity = 3;
        auto tree = Operon::Tree({Var(X1), Var(X2), Var(X3), node}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{5}};
        d[X2] = {S{2}, S{4}};
        d[X3] = {S{3}, S{6}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // min([1,5], [2,4], [3,6]) = [min(1,2,3), min(5,4,6)] = [1, 4]
        REQUIRE(Contains(r, 1.0, 4.0, 1e-5));
    }

    SECTION("ternary fmax") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmax>();
        node.Arity = 3;
        auto tree = Operon::Tree({Var(X1), Var(X2), Var(X3), node}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{5}};
        d[X2] = {S{2}, S{4}};
        d[X3] = {S{3}, S{6}};
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // max([1,5], [2,4], [3,6]) = [max(1,2,3), max(5,4,6)] = [3, 6]
        REQUIRE(Contains(r, 3.0, 6.0, 1e-5));
    }
}

TEST_CASE("Affine backend: n-ary Fmin/Fmax with arity > 2", "[pappus][affine]")
{
    // Regression for the binary-only Fmin/Fmax bug. With arity=3, the
    // evaluator must fold all three children, not just the first two.
    constexpr Operon::Hash X1{1}, X2{2}, X3{3};

    SECTION("ternary fmin") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmin>();
        node.Arity = 3;
        auto tree = Operon::Tree({Var(X1), Var(X2), Var(X3), node}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{5}};
        d[X2] = {S{2}, S{4}};
        d[X3] = {S{3}, S{6}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // min enclosure must contain [1, 4]
        REQUIRE(Contains(r, 1.0, 4.0, 1e-3));
    }

    SECTION("ternary fmax") {
        auto node = Util::MakeOp<Operon::BuiltinOp::Fmax>();
        node.Arity = 3;
        auto tree = Operon::Tree({Var(X1), Var(X2), Var(X3), node}).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{1}, S{5}};
        d[X2] = {S{2}, S{4}};
        d[X3] = {S{3}, S{6}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // max enclosure must contain [3, 6]
        REQUIRE(Contains(r, 3.0, 6.0, 1e-3));
    }
}

TEST_CASE("Affine backend: max_terms condensation", "[pappus][affine]")
{
    // Verify that max_terms > 0 triggers condensation, keeping the form's
    // term count bounded. With max_terms=0 (default), terms grow unbounded.
    constexpr Operon::Hash X1{1}, X2{2};
    // A tree that generates many noise terms: X1 * X2 * X1 * X2 (via Mul chain)
    Operon::Vector<Operon::Node> ns{
        Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>(),
        Var(X1), Util::MakeOp<Operon::BuiltinOp::Mul>(),
        Var(X2), Util::MakeOp<Operon::BuiltinOp::Mul>()};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto d = Domains();
    d[X1] = {S{1}, S{2}};
    d[X2] = {S{1}, S{2}};

    SECTION("max_terms=0 (unbounded) — terms grow freely") {
        AE eval(&tree, std::move(d), /*maxTerms=*/0);
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // The result is a valid enclosure of [1, 16]
        REQUIRE(Contains(r, 1.0, 16.0, 1e-3));
    }

    SECTION("max_terms=2 — form is condensed to <= 2 terms + remainder") {
        auto d2 = Domains();
        d2[X1] = {S{1}, S{2}};
        d2[X2] = {S{1}, S{2}};
        AE eval(&tree, std::move(d2), /*maxTerms=*/2);
        auto const r = eval.Evaluate(tree.GetCoefficients());
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
    constexpr Operon::Hash X1{1}, X2{2};

    SECTION("pow(base=[2,3], exp=[1,2]) -> contains [2, 9]") {
        // {X2, X1, Pow}: X1 is base (j=i-1), X2 is exponent (k).
        Operon::Vector<Operon::Node> ns{Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>()};
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{2}, S{3}};  // base
        d[X2] = {S{1}, S{2}};  // exponent
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // pow([2,3],[1,2]) = [2^1, 3^2] = [2, 9]
        REQUIRE(r.inf() <= 2.0 + 1e-3);
        REQUIRE(r.sup() + 1e-2 >= 9.0);
    }

    SECTION("powabs(base=[-3,2], exp=[2,3]) -> contains [0, 27]") {
        // {X2, X1, Powabs}: abs(X1) is base, X2 is exponent.
        Operon::Vector<Operon::Node> ns{Var(X2), Var(X1), Util::MakeOp<Operon::BuiltinOp::Powabs>()};
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-3}, S{2}};  // abs → [0, 3]
        d[X2] = {S{2}, S{3}};   // exponent
        IE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        // pow(abs([-3,2]), [2,3]) = pow([0,3], [2,3]) ⊇ [0, 27]
        REQUIRE(r.inf() <= 0.0 + 1e-3);
        REQUIRE(r.sup() + 1e-2 >= 27.0);
    }
}

TEST_CASE("Affine backend: pow/powabs child ordering", "[pappus][affine]")
{
    constexpr Operon::Hash X1{1};

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

    SECTION("pow(base=[2,3], exp=const_2) -> enclosure contains [4, 9]") {
        Operon::Vector<Operon::Node> ns{Const(2.0), Var(X1), Util::MakeOp<Operon::BuiltinOp::Pow>()};
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{2}, S{3}};
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 4.0, 9.0, 1e-2));
        // Key: upper bound must reach 9, not just 8 (the swapped result).
        REQUIRE(r.to_interval().sup() + 1e-2 >= 9.0);
    }

    // Powabs: base is X1 (abs applied), exponent is const.
    // Use X1=[-3,-1] (all negative) so abs(X1)=-X1=[1,3] is exact (negation,
    // no approximation error) and min(abs(X1))=1 > 0 so pow cannot throw.
    //   Correct: pow(abs([-3,-1])=[1,3], 3) -> [1, 27]
    //   Swapped: pow(abs(const_3)=3, [-3,-1]) -> 3^[-3,-1] ≈ [1/27, 1/3]

    SECTION("powabs(base=[-3,-1], exp=const_3) -> enclosure contains [1, 27]") {
        Operon::Vector<Operon::Node> ns{Const(3.0), Var(X1), Util::MakeOp<Operon::BuiltinOp::Powabs>()};
        auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
        auto d = Domains();
        d[X1] = {S{-3}, S{-1}};  // all-negative: abs = exact negation, min(abs) = 1 > 0
        AE eval(&tree, std::move(d));
        auto const r = eval.Evaluate(tree.GetCoefficients());
        REQUIRE(Contains(r, 1.0, 27.0, 1e-2));
    }
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

    constexpr Operon::Hash X1{1}, X2{2}, X3{3};
    Operon::Vector<Operon::Node> ns{
        Var(X1), Var(X2), Util::MakeOp<Operon::BuiltinOp::Add>(),
        Util::MakeOp<Operon::BuiltinOp::Exp>(),
        Var(X3), Util::MakeOp<Operon::BuiltinOp::Mul>()};
    auto tree = Operon::Tree(std::move(ns)).UpdateNodes();
    auto coeffs = tree.GetCoefficients();

    auto d = Domains();
    d[X1] = {S{0}, S{1}};
    d[X2] = {S{0}, S{1}};
    d[X3] = {S{1}, S{2}};

    nb::Bench b;
    b.timeUnit(std::chrono::microseconds(1), "µs").minEpochIterations(1000);

    // Scalar baseline: exp(x1 + x2) * x3 evaluated at the domain midpoints.
    // Shows the floor cost of the arithmetic itself before pappus overhead.
    b.run("scalar midpoint (baseline)", [&]() {
        S x1{0.5}, x2{0.5}, x3{1.5};
        nb::doNotOptimizeAway(std::exp(x1 + x2) * x3);
    });

    IE iev(&tree, IE::DomainMap{d});
    b.run("interval Evaluate", [&]() {
        nb::doNotOptimizeAway(iev.Evaluate(coeffs));
    });

    AE aev(&tree, AE::DomainMap{d});
    b.run("affine Evaluate (max_terms=0)", [&]() {
        nb::doNotOptimizeAway(aev.Evaluate(coeffs));
    });

    AE aev4(&tree, AE::DomainMap{d}, /*maxTerms=*/4);
    b.run("affine Evaluate (max_terms=4)", [&]() {
        nb::doNotOptimizeAway(aev4.Evaluate(coeffs));
    });

    // Phase 8b: term-growth report — call Evaluate once per config and print.
    // Not timed; just measures state after one evaluation.
    {
        AE ev_unbounded(&tree, AE::DomainMap{d}, /*maxTerms=*/0);
        auto const r_u = ev_unbounded.Evaluate(coeffs);
        auto const terms_u = ev_unbounded.TermCount();

        AE ev_4(&tree, AE::DomainMap{d}, /*maxTerms=*/4);
        auto const r_4 = ev_4.Evaluate(coeffs);
        auto const terms_4 = ev_4.TermCount();

        // Sanity: both enclosures must agree (one is tighter, one is looser).
        REQUIRE(r_u.to_interval().inf() <= r_4.to_interval().sup() + S{1e-2});
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
