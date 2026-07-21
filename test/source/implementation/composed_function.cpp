// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

// Exercises MakeComposedCallable/MakeComposedCallableDiff/
// ValidateSymbolicDiffCoverage (composed-function derivation, steps 2+3 of
// operon-planning/designs/composed-functions.md) directly against
// DispatchTable::RegisterFunction — the higher-level RegisterComposedFunction
// orchestration (wiring all pieces + the other 3 backends together) is a
// later step and not exercised here.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fmt/format.h>

#include "../operon_test.hpp"
#include "operon/core/composed_function.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

namespace {
    auto MakeComposedNode(std::string const& name, uint16_t arity) -> Operon::Node
    {
        auto const hash = Operon::Hasher{}(name);
        return Operon::Node::Function(hash, arity);
    }

    // Local copy of tree_diff.cpp's own test-file helper (EvalDagJacobian in
    // test/source/implementation/tree_diff.cpp) — evaluates one Jacobian
    // column via the interpreter over the sub-tree ending at its root.
    constexpr std::size_t kNoGrad = std::numeric_limits<std::size_t>::max();

    auto EvalJacColumn(Operon::JacobianDag const& dag, std::size_t k,
        Operon::Span<Operon::Scalar const> coeff, Operon::Dataset const& ds, Operon::Range range,
        Operon::DispatchTable<Operon::Scalar> const& dtable) -> Operon::Vector<Operon::Scalar>
    {
        auto const r = dag.Roots[k];
        if (r == kNoGrad) { return Operon::Vector<Operon::Scalar>(range.Size(), Operon::Scalar{0}); }
        Operon::Vector<Operon::Node> subnodes(dag.Nodes.cbegin(), dag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(r) + 1);
        Operon::Tree t{std::move(subnodes)};
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interp{&dtable, &ds, &t};
        return interp.Evaluate(coeff, range);
    }
} // namespace

TEST_CASE("Composed function: Callable derivation", "[composed-function]")
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const xHash = ds.GetVariable("x")->Hash;
    auto const yHash = ds.GetVariable("y")->Hash;

    SECTION("Unary body: numeric eval matches the manually-expanded expression") {
        auto body = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("logistic", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 1.0 / (1.0 + std::exp(-static_cast<double>(xs[i])));
            CHECK(static_cast<double>(result[i]) == Catch::Approx(expected).margin(1e-5));
        }
    }

    SECTION("Binary body: argument order matches Tree::Indices binding (order-sensitive op)") {
        // Sub is order-sensitive — this would fail silently (wrong sign/value)
        // if BindArgIndices bound params in the wrong order.
        auto body = InfixParser::ParseFunctionBody("a - b", std::vector<std::string>{"a", "b"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 2);
        auto composedNode = MakeComposedNode("sub2", 2);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Node vy(Operon::NodeType::Variable);
        vy.HashValue = vy.CalculatedHashValue = yHash;
        vy.Value = 1.0F;
        // Postfix: first-arg subtree, second-arg subtree, then the call —
        // i.e. sub2(x, y).
        Operon::Vector<Operon::Node> nodes{vx, vy, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        auto const ys = ds.GetValues("y");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(result[i]) == Catch::Approx(static_cast<double>(xs[i]) - static_cast<double>(ys[i])).margin(1e-5));
        }
    }

    SECTION("Composed node's own weight is applied once (not baked into the body)") {
        auto body = InfixParser::ParseFunctionBody("sin(x)", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("sinComposed", 1);
        composedNode.Value = 3.7F;
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 3.7 * std::sin(static_cast<double>(xs[i]));
            CHECK(static_cast<double>(result[i]) == Catch::Approx(expected).margin(1e-5));
        }
    }
}

TEST_CASE("Composed function: CallableDiff derivation", "[composed-function]")
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const xHash = ds.GetVariable("x")->Hash;

    SECTION("Weighted composed node: d/dx(w*sin(x)) = w*cos(x) — the exact case the double-weighting bug would break") {
        auto body = InfixParser::ParseFunctionBody("sin(x)", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("sinComposedW", 1);
        composedNode.Value = 3.7F;
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 3.7 * std::cos(static_cast<double>(xs[i]));
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(expected).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(expected).margin(1e-4));
        }
    }

    SECTION("Repeated parameter occurrence, nonzero derivative: d/dx(x + x) = 2") {
        auto body = InfixParser::ParseFunctionBody("x + x", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("doubleX", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(2.0).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(2.0).margin(1e-4));
        }
    }

    SECTION("Repeated parameter occurrence, cancelling derivative: d/dx(x - x) = 0") {
        auto body = InfixParser::ParseFunctionBody("x - x", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("zeroX", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(0.0).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(0.0).margin(1e-4));
        }
    }
}

TEST_CASE("Composed function: symbolic-diff coverage validation", "[composed-function]")
{
    SECTION("Body using only fully-covered built-ins passes") {
        auto body = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", std::vector<std::string>{"x"});
        CHECK_NOTHROW(Operon::ValidateSymbolicDiffCoverage(body));
    }

    SECTION("Unary non-diff built-in (abs) is rejected") {
        auto body = InfixParser::ParseFunctionBody("abs(x)", std::vector<std::string>{"x"});
        CHECK_THROWS_AS(Operon::ValidateSymbolicDiffCoverage(body), std::invalid_argument);
    }

    SECTION("Binary non-diff built-in (fmin, spelled 'min' in the parser vocabulary) is rejected") {
        auto body = InfixParser::ParseFunctionBody("min(x, y)", std::vector<std::string>{"x", "y"});
        CHECK_THROWS_AS(Operon::ValidateSymbolicDiffCoverage(body), std::invalid_argument);
    }

    SECTION("Binary non-diff built-in (aq) is rejected") {
        auto body = InfixParser::ParseFunctionBody("aq(x, y)", std::vector<std::string>{"x", "y"});
        CHECK_THROWS_AS(Operon::ValidateSymbolicDiffCoverage(body), std::invalid_argument);
    }

    SECTION("Hardcoded arity>=2 ops (add/mul/sub/div/pow) are accepted") {
        auto body = InfixParser::ParseFunctionBody("(x + y) * (x - y) / x ^ 2", std::vector<std::string>{"x", "y"});
        CHECK_NOTHROW(Operon::ValidateSymbolicDiffCoverage(body));
    }
}

TEST_CASE("Composed function: unary symbolic-diff rule (JIT/BuildJacobianDag path)", "[composed-function]")
{
    // Differentiate w.r.t. a tunable *Constant* argument, not a Variable's
    // own weight — BuildJacobianDag's leaf rule for a Variable is
    // d(w*X)/dw = X (an extra factor unrelated to what's being tested
    // here), whereas a Constant's derivative w.r.t. itself is exactly 1
    // (GetConst's own leaf case), keeping the expected value exactly the
    // composed rule's raw output with no extra scaling to account for.
    //
    // Also deliberately does NOT give the composed node itself a non-unit
    // weight: confirmed by inspection (only GetVar's leaf case reads
    // Node::Value anywhere in tree_diff.cpp) that Deriv() ignores every
    // Function node's own weight entirely, for every built-in, not just
    // composed ones — real GP-evolved trees never coefficient-optimize a
    // Function node's weight (Node::Function always sets Optimize=false),
    // so this isn't a gap composed functions need to special-case.
    Operon::Dataset ds(std::vector<std::string>{"x"}, std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};

    SECTION("d/dc logistic(c) = logistic(c) * (1 - logistic(c))") {
        auto body = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", std::vector<std::string>{"x"});
        auto composedNode = MakeComposedNode("logisticJit", 1);
        dtable.RegisterFunction<Operon::Scalar>(
            composedNode.HashValue,
            MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1));
        Operon::RegisterUnarySymbolicDeriv(composedNode.HashValue, Operon::MakeComposedUnarySymbolicDerivRule(body));

        auto c = Operon::Node::Constant(1.3);
        Operon::Vector<Operon::Node> nodes{c, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        auto dag = Operon::BuildJacobianDag(tree);
        REQUIRE(dag.Roots.size() == 1);
        REQUIRE(dag.Roots[0] != kNoGrad);

        auto jac = EvalJacColumn(dag, 0, coeff, ds, range, dtable);
        auto const lg = 1.0 / (1.0 + std::exp(-1.3));
        auto const expected = lg * (1.0 - lg);
        CHECK(static_cast<double>(jac[0]) == Catch::Approx(expected).margin(1e-4));
    }

    SECTION("d/dc sin_composed(c) = cos(c)") {
        auto body = InfixParser::ParseFunctionBody("sin(x)", std::vector<std::string>{"x"});
        auto composedNode = MakeComposedNode("sinJit", 1);
        dtable.RegisterFunction<Operon::Scalar>(
            composedNode.HashValue,
            MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1));
        Operon::RegisterUnarySymbolicDeriv(composedNode.HashValue, Operon::MakeComposedUnarySymbolicDerivRule(body));

        auto c = Operon::Node::Constant(1.3);
        Operon::Vector<Operon::Node> nodes{c, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        auto dag = Operon::BuildJacobianDag(tree);
        REQUIRE(dag.Roots.size() == 1);
        REQUIRE(dag.Roots[0] != kNoGrad);

        auto jac = EvalJacColumn(dag, 0, coeff, ds, range, dtable);
        auto const expected = std::cos(1.3);
        CHECK(static_cast<double>(jac[0]) == Catch::Approx(expected).margin(1e-4));
    }
}

TEST_CASE("Composed function: unary interval/affine mini-evaluator", "[composed-function]")
{
    auto const xHash = Operon::Hasher{}("x");

    SECTION("Interval: logistic(x) encloses within (0,1) for a wide domain") {
        auto body = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", std::vector<std::string>{"x"});
        auto composedNode = MakeComposedNode("logisticInterval", 1);
        Operon::RegisterUnaryInterval(composedNode.HashValue, Operon::MakeComposedIntervalUnaryFn(body));

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        vx.Optimize = false;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        Operon::IntervalEvaluator::DomainMap domains{{xHash, {-5.0F, 5.0F}}};
        Operon::IntervalEvaluator eval(&tree, domains);
        auto result = eval.Evaluate({});
        CHECK(result.inf() >= -1e-3);
        CHECK(result.sup() <= 1.0 + 1e-3);
        // Must be a real (non-degenerate) enclosure, not [0,0] from a bug
        // that dropped the argument entirely.
        CHECK(result.sup() > result.inf());
    }

    SECTION("Affine: identity(x) - x encloses to exactly 0 via structural sharing (correlation preserved)") {
        // The exact motivating case for the "reuse the caller's already-
        // computed form" fix (Fix 1): a nested-pass approach that rebinds
        // the param as a *fresh* affine variable would lose the shared
        // noise symbol and wrongly enclose this as [-w,+w] instead of 0.
        // Structural sharing (Ref, not two independent Variable nodes —
        // two separate Variable occurrences of the same hash each get
        // their OWN fresh noise symbol in AffineEvaluator and would NOT
        // correlate regardless of what the composed function does) is
        // required to actually exercise this: node0 is the one real "x"
        // leaf; node1 is a Ref to it, feeding identity() the *same*
        // affine_form object (same symbols) that node0 itself carries.
        auto body = InfixParser::ParseFunctionBody("x", std::vector<std::string>{"x"});
        auto composedNode = MakeComposedNode("identityAffine", 1);
        Operon::RegisterUnaryAffine(composedNode.HashValue, Operon::MakeComposedAffineUnaryFn(body));

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        vx.Optimize = false;
        auto ref = Operon::Node::Ref(0);
        auto subNode = Operon::Node::Function(Operon::Hash(Operon::BuiltinOp::Sub), 2);
        // identity(x) - x: [x(real), Ref(0), identity(Ref(0)), Sub]
        Operon::Vector<Operon::Node> nodes{vx, ref, composedNode, subNode};
        Operon::Tree tree{nodes};

        Operon::AffineEvaluator::DomainMap domains{{xHash, {-5.0F, 5.0F}}};
        Operon::AffineEvaluator eval(&tree, domains);
        auto result = eval.Evaluate({});
        auto const iv = result.to_interval();
        CHECK(iv.inf() == Catch::Approx(0.0).margin(1e-4));
        CHECK(iv.sup() == Catch::Approx(0.0).margin(1e-4));
    }
}

TEST_CASE("Composed function: chained-Add symbolic diff (hash-consing regression)", "[composed-function]")
{
    // Regression test for a real hash-consing collision found via code
    // review (2026-07-21): tree_diff.cpp assigns h[i]=i for the original
    // tree's own nodes (h[0]=0 is normal there), and BuiltinOp::Add==0 too
    // — DiffMix(Add, h[0]=0) collapsed to exactly 0 and stayed 0 through a
    // second Add combining with it, making DiffMakeBinary's memo spuriously
    // treat Add(innerSum, x) as identical to Add(x, x). Fixed by salting
    // DiffMix so (0,0) is no longer a fixed point. Without the fix, this
    // computed 3*exp(2x) instead of 3*exp(3x).
    Operon::Dataset ds(std::vector<std::string>{"x"}, std::vector<std::vector<Operon::Scalar>>{{1.3F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};

    auto body = InfixParser::ParseFunctionBody("exp(x + x + x)", std::vector<std::string>{"x"});
    auto composedNode = MakeComposedNode("chainedAddExp", 1);
    dtable.RegisterFunction<Operon::Scalar>(
        composedNode.HashValue,
        MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1));
    Operon::RegisterUnarySymbolicDeriv(composedNode.HashValue, Operon::MakeComposedUnarySymbolicDerivRule(body));

    auto c = Operon::Node::Constant(1.3);
    Operon::Vector<Operon::Node> nodes{c, composedNode};
    Operon::Tree tree{nodes};

    auto coeff = tree.GetCoefficients();
    auto dag = Operon::BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 1);
    REQUIRE(dag.Roots[0] != kNoGrad);

    auto jac = EvalJacColumn(dag, 0, coeff, ds, range, dtable);
    auto const expected = 3.0 * std::exp(3.0 * 1.3); // d/dc exp(3c) = 3*exp(3c)
    CHECK(static_cast<double>(jac[0]) == Catch::Approx(expected).margin(1e-3));
}

TEST_CASE("Composed function: chained-Sub symbolic diff (x - x - x)", "[composed-function]")
{
    // d/dc [c - c - c] = -1 (nested binary Sub: (c-c)-c, so d/dc = 1-1-1 = -1).
    Operon::Dataset ds(std::vector<std::string>{"x"}, std::vector<std::vector<Operon::Scalar>>{{1.3F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};

    auto body = InfixParser::ParseFunctionBody("x - x - x", std::vector<std::string>{"x"});
    auto composedNode = MakeComposedNode("chainedSub", 1);
    dtable.RegisterFunction<Operon::Scalar>(
        composedNode.HashValue,
        MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1));
    Operon::RegisterUnarySymbolicDeriv(composedNode.HashValue, Operon::MakeComposedUnarySymbolicDerivRule(body));

    auto c = Operon::Node::Constant(1.3);
    Operon::Vector<Operon::Node> nodes{c, composedNode};
    Operon::Tree tree{nodes};

    // Sanity: numeric value should also be c - c - c = -c.
    auto coeffEval = tree.GetCoefficients();
    Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interp{&dtable, &ds, &tree};
    auto val = interp.Evaluate(coeffEval, range);
    CHECK(static_cast<double>(val[0]) == Catch::Approx(-1.3).margin(1e-4));

    auto coeff = tree.GetCoefficients();
    auto dag = Operon::BuildJacobianDag(tree);
    REQUIRE(dag.Roots.size() == 1);
    REQUIRE(dag.Roots[0] != kNoGrad);

    auto jac = EvalJacColumn(dag, 0, coeff, ds, range, dtable);
    CHECK(static_cast<double>(jac[0]) == Catch::Approx(-1.0).margin(1e-3));
}

} // namespace Operon::Test
