// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "operon/core/node.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/affine_evaluator.hpp"

#ifdef HAVE_ASMJIT
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#endif

// Cross-registry coverage: nothing structurally guarantees the symbolic-diff,
// interval, affine, and JIT-codegen registries all cover the same set of
// BuiltinOps. A BuiltinOp missing from one of them silently degrades (Zero
// for diff, a throw for interval/affine/JIT) — indistinguishable from a
// legitimately-unsupported op unless asserted explicitly. This test is the
// deliverable flagged (but not built) in the registry redesign's planning
// notes: assert every BuiltinOp is either registered, or on the explicit
// "deliberately absent" allowlist below, for each registry.

namespace Operon::Test {

namespace {

auto OpName(BuiltinOp op) -> std::string
{
    return Node::Function(static_cast<Operon::Hash>(op), 1).Name();
}

// n-ary reductions (Add/Mul/Sub/Div/Fmin/Fmax) are handled directly as
// structural cases in each consumer (Deriv(), Evaluate(), EmitNodesAvx2) —
// never through a single-hash unary/binary registry entry. Shared by the
// interval/affine/JIT checks below (all three have the identical boundary).
auto NaryFolds() -> std::vector<BuiltinOp> const&
{
    static std::vector<BuiltinOp> const ops {
        BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Sub, BuiltinOp::Div,
        BuiltinOp::Fmin, BuiltinOp::Fmax,
    };
    return ops;
}

} // namespace

TEST_CASE("Cross-registry coverage: symbolic-diff registry", "[registry][coverage]")
{
    // Structural (n-ary, or Pow's two-term chain rule — handled directly in
    // Deriv(), never through the unary registry) or genuinely not-yet-
    // implemented — see tree_diff.cpp's Deriv() and
    // RegisterBuiltinSymbolicDerivs().
    std::vector<BuiltinOp> deliberatelyAbsent = NaryFolds();
    deliberatelyAbsent.insert(deliberatelyAbsent.end(), {
        BuiltinOp::Aq, BuiltinOp::Pow, BuiltinOp::Powabs, // Pow: structural (two terms); Aq/Powabs: not yet differentiated
        BuiltinOp::Abs, BuiltinOp::Sqrtabs, BuiltinOp::Floor, BuiltinOp::Ceil, // non-smooth
    });

    for (std::size_t i = 0; i < Operon::BuiltinOpCount; ++i) {
        auto const op = static_cast<BuiltinOp>(i);
        auto const hash = static_cast<Operon::Hash>(op);
        bool const present = HasUnarySymbolicDeriv(hash);
        bool const expectAbsent = std::ranges::find(deliberatelyAbsent, op) != deliberatelyAbsent.end();
        INFO("op: " << OpName(op));
        CHECK(present == !expectAbsent);
    }
}

TEST_CASE("Cross-registry coverage: interval/affine registries", "[registry][coverage]")
{
    // Force built-in registration (each evaluator populates its registries
    // lazily on first Evaluate() call). Optimize=false so Evaluate({}) needs
    // no coefficient span (Node::Constant() defaults Optimize=true as a leaf).
    auto constNode = Node::Constant(1.0F);
    constNode.Optimize = false;
    Operon::Tree dummy{{constNode}};
    Operon::IntervalEvaluator::DomainMap noDomains;
    static_cast<void>(IntervalEvaluator{&dummy, noDomains}.Evaluate({}));
    static_cast<void>(AffineEvaluator{&dummy, noDomains}.Evaluate({}));

    auto const& deliberatelyAbsent = NaryFolds();

    for (std::size_t i = 0; i < Operon::BuiltinOpCount; ++i) {
        auto const op = static_cast<BuiltinOp>(i);
        auto const hash = static_cast<Operon::Hash>(op);
        bool const expectAbsent = std::ranges::find(deliberatelyAbsent, op) != deliberatelyAbsent.end();

        bool const inInterval = Operon::IntervalUnaryRules().Contains(hash)
            || Operon::IntervalBinaryRules().Contains(hash);
        bool const inAffine = Operon::AffineUnaryRules().Contains(hash)
            || Operon::AffineBinaryRules().Contains(hash);

        INFO("op: " << OpName(op));
        CHECK(inInterval == !expectAbsent);
        CHECK(inAffine == !expectAbsent);
    }
}

#ifdef HAVE_ASMJIT
TEST_CASE("Cross-registry coverage: JIT codegen registry", "[registry][coverage][jit]")
{
    auto const& deliberatelyAbsent = NaryFolds();

    for (std::size_t i = 0; i < Operon::BuiltinOpCount; ++i) {
        auto const op = static_cast<BuiltinOp>(i);
        auto const hash = static_cast<Operon::Hash>(op);
        bool const expectAbsent = std::ranges::find(deliberatelyAbsent, op) != deliberatelyAbsent.end();

        bool const inJit = Operon::JIT::HasUnaryJitCodegen(hash)
            || Operon::JIT::HasBinaryJitCodegen(hash);

        INFO("op: " << OpName(op));
        CHECK(inJit == !expectAbsent);
    }
}
#endif

} // namespace Operon::Test
