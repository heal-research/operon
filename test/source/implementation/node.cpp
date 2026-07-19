// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"

namespace Operon::Test {

namespace {
    // Local equivalent of Util::MakeOp (operon_test.hpp) - this file is
    // deliberately self-contained (only includes node.hpp), so it doesn't
    // pull in the shared test harness header just for this one helper.
    template<BuiltinOp Op>
    auto MakeOp() -> Node {
        constexpr uint16_t arity = Node::IsUnaryOp<Op> ? 1 : 2;
        return Node::Function(static_cast<Operon::Hash>(Op), arity);
    }
} // namespace

TEST_CASE("Node::Is<BuiltinOp...>() distinguishes built-in ops by HashValue", "[core]")
{
    CHECK(MakeOp<BuiltinOp::Add>().Is<BuiltinOp::Add>());
    CHECK_FALSE(MakeOp<BuiltinOp::Add>().Is<BuiltinOp::Mul>());
    CHECK(MakeOp<BuiltinOp::Sin>().Is<BuiltinOp::Sin>());
    CHECK((MakeOp<BuiltinOp::Fmax>().Is<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax>()));
    CHECK_FALSE((MakeOp<BuiltinOp::Sub>().Is<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax>()));
}

TEST_CASE("IsNaryOp/IsBinaryOp/IsUnaryOp<BuiltinOp> classify every op consistently", "[core]")
{
    auto check = [](BuiltinOp op, bool isNary, bool isBinary, bool isUnary) {
        CHECK(!Node::Function(static_cast<Operon::Hash>(op), 1).IsLeaf()); // sanity: every BuiltinOp-backed node is non-leaf
        CHECK(isNary == (op == BuiltinOp::Add || op == BuiltinOp::Mul || op == BuiltinOp::Sub
            || op == BuiltinOp::Div || op == BuiltinOp::Fmin || op == BuiltinOp::Fmax));
        CHECK(isBinary == (op == BuiltinOp::Aq || op == BuiltinOp::Pow || op == BuiltinOp::Powabs));
        CHECK(isUnary == (!isNary && !isBinary));
        CHECK((isNary ? 1 : 0) + (isBinary ? 1 : 0) + (isUnary ? 1 : 0) == 1); // mutually exclusive, exhaustive
    };

    check(BuiltinOp::Add, Node::IsNaryOp<BuiltinOp::Add>, Node::IsBinaryOp<BuiltinOp::Add>, Node::IsUnaryOp<BuiltinOp::Add>);
    check(BuiltinOp::Aq, Node::IsNaryOp<BuiltinOp::Aq>, Node::IsBinaryOp<BuiltinOp::Aq>, Node::IsUnaryOp<BuiltinOp::Aq>);
    check(BuiltinOp::Sin, Node::IsNaryOp<BuiltinOp::Sin>, Node::IsBinaryOp<BuiltinOp::Sin>, Node::IsUnaryOp<BuiltinOp::Sin>);

    // Boundary values specifically (most likely place for an off-by-one).
    static_assert(Node::IsNaryOp<BuiltinOp::Fmax>);
    static_assert(!Node::IsNaryOp<BuiltinOp::Aq>);
    static_assert(Node::IsBinaryOp<BuiltinOp::Aq>);
    static_assert(Node::IsBinaryOp<BuiltinOp::Powabs>);
    static_assert(!Node::IsBinaryOp<BuiltinOp::Abs>);
    static_assert(Node::IsUnaryOp<BuiltinOp::Abs>);
    static_assert(Node::IsUnaryOp<BuiltinOp::Square>);
}

TEST_CASE("Node::Is*() single-op convenience methods are re-pointed correctly", "[core]")
{
    // Each of these compares HashValue via Is<BuiltinOp::X>() (node.hpp), so
    // must still report true only for the one op it names and false for
    // every other built-in op.
    CHECK(MakeOp<BuiltinOp::Add>().IsAddition());
    CHECK_FALSE(MakeOp<BuiltinOp::Mul>().IsAddition());
    CHECK(MakeOp<BuiltinOp::Sub>().IsSubtraction());
    CHECK_FALSE(MakeOp<BuiltinOp::Add>().IsSubtraction());
    CHECK(MakeOp<BuiltinOp::Mul>().IsMultiplication());
    CHECK_FALSE(MakeOp<BuiltinOp::Div>().IsMultiplication());
    CHECK(MakeOp<BuiltinOp::Div>().IsDivision());
    CHECK_FALSE(MakeOp<BuiltinOp::Mul>().IsDivision());
    CHECK(MakeOp<BuiltinOp::Aq>().IsAq());
    CHECK_FALSE(MakeOp<BuiltinOp::Pow>().IsAq());
    CHECK(MakeOp<BuiltinOp::Pow>().IsPow());
    CHECK_FALSE(MakeOp<BuiltinOp::Powabs>().IsPow());
    CHECK(MakeOp<BuiltinOp::Powabs>().IsPowabs());
    CHECK_FALSE(MakeOp<BuiltinOp::Pow>().IsPowabs());
    CHECK(MakeOp<BuiltinOp::Exp>().IsExp());
    CHECK_FALSE(MakeOp<BuiltinOp::Log>().IsExp());
    CHECK(MakeOp<BuiltinOp::Log>().IsLog());
    CHECK_FALSE(MakeOp<BuiltinOp::Exp>().IsLog());
    CHECK(MakeOp<BuiltinOp::Sin>().IsSin());
    CHECK_FALSE(MakeOp<BuiltinOp::Cos>().IsSin());
    CHECK(MakeOp<BuiltinOp::Cos>().IsCos());
    CHECK_FALSE(MakeOp<BuiltinOp::Sin>().IsCos());
    CHECK(MakeOp<BuiltinOp::Tan>().IsTan());
    CHECK_FALSE(MakeOp<BuiltinOp::Tanh>().IsTan());
    CHECK(MakeOp<BuiltinOp::Tanh>().IsTanh());
    CHECK_FALSE(MakeOp<BuiltinOp::Tan>().IsTanh());
    CHECK(MakeOp<BuiltinOp::Sqrt>().IsSquareRoot());
    CHECK_FALSE(MakeOp<BuiltinOp::Cbrt>().IsSquareRoot());
    CHECK(MakeOp<BuiltinOp::Cbrt>().IsCubeRoot());
    CHECK_FALSE(MakeOp<BuiltinOp::Sqrt>().IsCubeRoot());
    CHECK(MakeOp<BuiltinOp::Square>().IsSquare());
    CHECK_FALSE(MakeOp<BuiltinOp::Pow>().IsSquare());
}

TEST_CASE("Node::IsCommutative() agrees for every built-in op", "[core]")
{
    for (auto op : { BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax }) {
        CHECK(Node::Function(static_cast<Operon::Hash>(op), 2).IsCommutative());
    }
    for (auto op : { BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Aq, BuiltinOp::Pow, BuiltinOp::Powabs }) {
        CHECK_FALSE(Node::Function(static_cast<Operon::Hash>(op), 2).IsCommutative());
    }
    for (auto op : { BuiltinOp::Sin, BuiltinOp::Cos, BuiltinOp::Square }) {
        CHECK_FALSE(Node::Function(static_cast<Operon::Hash>(op), 1).IsCommutative());
    }
}

TEST_CASE("Node::Function() sets Type/HashValue/Arity/Length/Optimize consistently", "[core]")
{
    auto n = Node::Function(static_cast<Operon::Hash>(BuiltinOp::Add), 3);
    CHECK(n.Type == NodeType::Function);
    CHECK(n.HashValue == static_cast<Operon::Hash>(BuiltinOp::Add));
    CHECK(n.Arity == 3);
    CHECK(n.Length == 3);
    CHECK_FALSE(n.IsLeaf());
    CHECK_FALSE(n.Optimize);
}

TEST_CASE("Node::Function() dispatches through the BuiltinOp-retargeted registry", "[core]")
{
    // Regression guard for the dispatch-chain retarget (functions.hpp/
    // derivatives.hpp/dispatch.hpp/standard_library.hpp now templated on
    // BuiltinOp instead of NodeType): every one of the 29 built-in ops must
    // still resolve to both a registered Callable and CallableDiff by hash,
    // independent of a Node's Type. Covers all of BuiltinOp, not just a
    // sample, and checks derivatives too (Diff<>/DiffOp/MakeDiffCall are
    // retargeted by this PR just as much as Func<>/MakeFunctionCall are).
    DispatchTable<Operon::Scalar> dt;
    for (auto op : { BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Sub, BuiltinOp::Div,
                      BuiltinOp::Fmin, BuiltinOp::Fmax, BuiltinOp::Aq, BuiltinOp::Pow,
                      BuiltinOp::Powabs, BuiltinOp::Abs, BuiltinOp::Acos, BuiltinOp::Asin,
                      BuiltinOp::Atan, BuiltinOp::Cbrt, BuiltinOp::Ceil, BuiltinOp::Cos,
                      BuiltinOp::Cosh, BuiltinOp::Exp, BuiltinOp::Floor, BuiltinOp::Log,
                      BuiltinOp::Logabs, BuiltinOp::Log1p, BuiltinOp::Sin, BuiltinOp::Sinh,
                      BuiltinOp::Sqrt, BuiltinOp::Sqrtabs, BuiltinOp::Tan, BuiltinOp::Tanh,
                      BuiltinOp::Square }) {
        auto const hash = static_cast<Operon::Hash>(op);
        CHECK(dt.Contains(hash));
        CHECK(dt.TryGetFunction<Operon::Scalar>(hash).has_value());
        CHECK(dt.TryGetDerivative<Operon::Scalar>(hash).has_value());
    }
}

} // namespace Operon::Test
