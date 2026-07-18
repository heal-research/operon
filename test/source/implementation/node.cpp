// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include "operon/core/node.hpp"

namespace Operon::Test {

TEST_CASE("BuiltinOp::X shares NodeType::X's ordinal and a built-in Node's HashValue", "[core]")
{
    // Node(NodeType) sets HashValue = static_cast<Hash>(Type), so for every
    // built-in math op, Is<NodeType::X>() and Is<BuiltinOp::X>() must agree.
    auto check = [](NodeType type, BuiltinOp op) {
        Node const n(type);
        CHECK(static_cast<Operon::Hash>(type) == static_cast<Operon::Hash>(op));
        CHECK(n.HashValue == static_cast<Operon::Hash>(op));
    };

    check(NodeType::Add, BuiltinOp::Add);
    check(NodeType::Mul, BuiltinOp::Mul);
    check(NodeType::Sub, BuiltinOp::Sub);
    check(NodeType::Div, BuiltinOp::Div);
    check(NodeType::Fmin, BuiltinOp::Fmin);
    check(NodeType::Fmax, BuiltinOp::Fmax);
    check(NodeType::Aq, BuiltinOp::Aq);
    check(NodeType::Pow, BuiltinOp::Pow);
    check(NodeType::Powabs, BuiltinOp::Powabs);
    check(NodeType::Abs, BuiltinOp::Abs);
    check(NodeType::Acos, BuiltinOp::Acos);
    check(NodeType::Asin, BuiltinOp::Asin);
    check(NodeType::Atan, BuiltinOp::Atan);
    check(NodeType::Cbrt, BuiltinOp::Cbrt);
    check(NodeType::Ceil, BuiltinOp::Ceil);
    check(NodeType::Cos, BuiltinOp::Cos);
    check(NodeType::Cosh, BuiltinOp::Cosh);
    check(NodeType::Exp, BuiltinOp::Exp);
    check(NodeType::Floor, BuiltinOp::Floor);
    check(NodeType::Log, BuiltinOp::Log);
    check(NodeType::Logabs, BuiltinOp::Logabs);
    check(NodeType::Log1p, BuiltinOp::Log1p);
    check(NodeType::Sin, BuiltinOp::Sin);
    check(NodeType::Sinh, BuiltinOp::Sinh);
    check(NodeType::Sqrt, BuiltinOp::Sqrt);
    check(NodeType::Sqrtabs, BuiltinOp::Sqrtabs);
    check(NodeType::Tan, BuiltinOp::Tan);
    check(NodeType::Tanh, BuiltinOp::Tanh);
    check(NodeType::Square, BuiltinOp::Square);
}

TEST_CASE("Node::Is<BuiltinOp...>() agrees with Is<NodeType...>() for built-in nodes", "[core]")
{
    CHECK(Node(NodeType::Add).Is<BuiltinOp::Add>());
    CHECK_FALSE(Node(NodeType::Add).Is<BuiltinOp::Mul>());
    CHECK(Node(NodeType::Sin).Is<BuiltinOp::Sin>());
    CHECK((Node(NodeType::Fmax).Is<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax>()));
    CHECK_FALSE((Node(NodeType::Sub).Is<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax>()));
}

TEST_CASE("IsNaryOp/IsBinaryOp/IsUnaryOp<BuiltinOp> agree with IsNary/IsBinary/IsUnary<NodeType>", "[core]")
{
    auto check = [](NodeType type, bool isNary, bool isBinary, bool isUnary) {
        CHECK(Node(type).IsLeaf() == false); // sanity: all BuiltinOp-backed types are non-leaf
        CHECK(isNary == (type == NodeType::Add || type == NodeType::Mul || type == NodeType::Sub
            || type == NodeType::Div || type == NodeType::Fmin || type == NodeType::Fmax));
        CHECK(isBinary == (type == NodeType::Aq || type == NodeType::Pow || type == NodeType::Powabs));
        CHECK(isUnary == (!isNary && !isBinary));
    };

    // NodeType-keyed boundaries (existing, unchanged)
    check(NodeType::Add, Node::IsNary<NodeType::Add>, Node::IsBinary<NodeType::Add>, Node::IsUnary<NodeType::Add>);
    check(NodeType::Aq, Node::IsNary<NodeType::Aq>, Node::IsBinary<NodeType::Aq>, Node::IsUnary<NodeType::Aq>);
    check(NodeType::Sin, Node::IsNary<NodeType::Sin>, Node::IsBinary<NodeType::Sin>, Node::IsUnary<NodeType::Sin>);

    // BuiltinOp-keyed boundaries must classify every op identically to their
    // NodeType counterparts.
    CHECK(Node::IsNaryOp<BuiltinOp::Add> == Node::IsNary<NodeType::Add>);
    CHECK(Node::IsNaryOp<BuiltinOp::Mul> == Node::IsNary<NodeType::Mul>);
    CHECK(Node::IsNaryOp<BuiltinOp::Sub> == Node::IsNary<NodeType::Sub>);
    CHECK(Node::IsNaryOp<BuiltinOp::Div> == Node::IsNary<NodeType::Div>);
    CHECK(Node::IsNaryOp<BuiltinOp::Fmin> == Node::IsNary<NodeType::Fmin>);
    CHECK(Node::IsNaryOp<BuiltinOp::Fmax> == Node::IsNary<NodeType::Fmax>);

    CHECK(Node::IsBinaryOp<BuiltinOp::Aq> == Node::IsBinary<NodeType::Aq>);
    CHECK(Node::IsBinaryOp<BuiltinOp::Pow> == Node::IsBinary<NodeType::Pow>);
    CHECK(Node::IsBinaryOp<BuiltinOp::Powabs> == Node::IsBinary<NodeType::Powabs>);

    CHECK(Node::IsUnaryOp<BuiltinOp::Abs> == Node::IsUnary<NodeType::Abs>);
    CHECK(Node::IsUnaryOp<BuiltinOp::Sin> == Node::IsUnary<NodeType::Sin>);
    CHECK(Node::IsUnaryOp<BuiltinOp::Square> == Node::IsUnary<NodeType::Square>);

    // Boundary values specifically (most likely place for an off-by-one).
    static_assert(Node::IsNaryOp<BuiltinOp::Fmax>);
    static_assert(!Node::IsNaryOp<BuiltinOp::Aq>);
    static_assert(Node::IsBinaryOp<BuiltinOp::Aq>);
    static_assert(Node::IsBinaryOp<BuiltinOp::Powabs>);
    static_assert(!Node::IsBinaryOp<BuiltinOp::Abs>);
    static_assert(Node::IsUnaryOp<BuiltinOp::Abs>);
    static_assert(Node::IsUnaryOp<BuiltinOp::Square>);
}

} // namespace Operon::Test
