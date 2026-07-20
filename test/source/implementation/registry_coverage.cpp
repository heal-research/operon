// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include <string>
#include <vector>

#include "operon/core/hash_registry.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/affine_evaluator.hpp"

#ifdef HAVE_ASMJIT
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#endif

#include "../operon_test.hpp"

// Cross-registry coverage: nothing structurally guarantees the symbolic-diff,
// interval, affine, and JIT-codegen registries all cover the same set of
// BuiltinOps. A BuiltinOp missing from one of them silently degrades (Zero
// for diff, a throw for interval/affine/JIT) — indistinguishable from a
// legitimately-unsupported op unless asserted explicitly. These tests assert
// every BuiltinOp is either registered, or on the explicit "deliberately
// absent" allowlist below, for each registry — plus behavioral coverage for
// the public Register*()/Has*() API surface itself: round-trip lookup,
// write-once duplicate rejection, the register-before-first-evaluate
// ordering fix, and each registry's end-to-end miss behavior.

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

// --- HashRegistry<Fn> itself: the write-once contract every domain above
// builds on. Tested directly (not through a domain's Fn type) so the
// contract is verified independent of any one consumer.
TEST_CASE("HashRegistry: round-trip and write-once contract", "[registry][hash_registry]")
{
    Operon::HashRegistry<int> reg;
    constexpr Operon::Hash hash{0x1111111111111111ULL};

    CHECK_FALSE(reg.Contains(hash));
    CHECK(reg.TryGet(hash) == nullptr);

    reg.Register(hash, 42);
    REQUIRE(reg.Contains(hash));
    REQUIRE(reg.TryGet(hash) != nullptr);
    CHECK(*reg.TryGet(hash) == 42);

    // Write-once: a second Register() for the same hash throws rather than
    // overwriting, even with a different value.
    CHECK_THROWS_AS(reg.Register(hash, 99), std::invalid_argument);
    CHECK(*reg.TryGet(hash) == 42); // unchanged by the failed second write
}

// --- Regression coverage for the TOCTOU fix: every public Register*() now
// triggers its domain's built-in lazy-init before writing, so registering a
// hash that collides with a built-in always throws immediately at the call
// site — regardless of whether anything in this process happened to already
// force that domain's built-in registration. Pre-fix, this was
// order-dependent (silently accepted if the built-ins hadn't been touched
// yet, only surfacing as a confusing throw later, from a different call,
// the first time the domain was actually used). Split one per domain
// (rather than one TEST_CASE with a SECTION each) so each domain's own PR
// carries its own regression coverage independently.
TEST_CASE("RegisterUnarySymbolicDeriv: colliding with a built-in hash always throws", "[registry][toctou]")
{
    CHECK_THROWS_AS(
        Operon::RegisterUnarySymbolicDeriv(Operon::Hash(BuiltinOp::Log),
            [](Operon::Vector<Operon::Node>&, Operon::Map<Operon::Hash, std::size_t>&,
               Operon::Vector<Operon::Hash>&, std::size_t, std::size_t) -> std::size_t { return 0; }),
        std::invalid_argument);
}

TEST_CASE("RegisterUnaryInterval/RegisterUnaryAffine: colliding with a built-in hash always throws", "[registry][toctou]")
{
    auto const logHash = Operon::Hash(BuiltinOp::Log);

    CHECK_THROWS_AS(
        Operon::RegisterUnaryInterval(logHash, [](Operon::IntervalEvaluator::Interval const& v) { return v; }),
        std::invalid_argument);
    CHECK_THROWS_AS(
        Operon::RegisterUnaryAffine(logHash,
            [](Operon::AffineEvaluator::Context const&, Operon::AffineEvaluator::Affine const& v) { return v; }),
        std::invalid_argument);
}

#ifdef HAVE_ASMJIT
TEST_CASE("RegisterUnaryJitCodegen: colliding with a built-in hash always throws", "[registry][toctou][jit]")
{
    CHECK_THROWS_AS(
        Operon::JIT::RegisterUnaryJitCodegen(Operon::Hash(BuiltinOp::Log),
            [](asmjit::x86::Compiler&, asmjit::x86::Vec const& a) { return a; }),
        std::invalid_argument);
}
#endif

// --- Round-trip: a genuinely new (non-built-in) hash registers cleanly,
// is invoked by the consumer, and a second registration under the same
// hash throws.
TEST_CASE("IntervalEvaluator: user-registered unary op round-trips and rejects duplicates", "[registry][interval]")
{
    constexpr Operon::Hash customHash{0x2222222222222222ULL};
    // static: the lambda is stored permanently in the global IntervalUnaryRules()
    // registry (there's no way to unregister it), so it must not capture a
    // reference to something with the TEST_CASE's own (shorter) lifetime.
    static bool invoked = false;
    invoked = false;
    Operon::RegisterUnaryInterval(customHash, [](Operon::IntervalEvaluator::Interval const& v) {
        invoked = true;
        return v;
    });
    REQUIRE(Operon::IntervalUnaryRules().Contains(customHash));

    CHECK_THROWS_AS(
        Operon::RegisterUnaryInterval(customHash, [](Operon::IntervalEvaluator::Interval const& v) { return v; }),
        std::invalid_argument);

    auto constNode = Node::Constant(1.0F);
    constNode.Optimize = false;
    Tree const tree = Tree({ constNode, Node::Function(customHash, 1) }).UpdateNodes();
    Operon::IntervalEvaluator::DomainMap noDomains;
    static_cast<void>(IntervalEvaluator{&tree, noDomains}.Evaluate({}));
    CHECK(invoked);
}

// --- End-to-end miss behavior: a genuinely unmapped hash (not registered
// anywhere) must still throw for interval/affine, matching the unchanged
// contract documented on Register{Unary,Binary}{Interval,Affine}.
TEST_CASE("IntervalEvaluator/AffineEvaluator: unmapped op throws at Evaluate()", "[registry][miss]")
{
    constexpr Operon::Hash unmappedHash{0x3333333333333333ULL};
    auto constNode = Node::Constant(1.0F);
    constNode.Optimize = false;
    Tree const tree = Tree({ constNode, Node::Function(unmappedHash, 1) }).UpdateNodes();
    Operon::IntervalEvaluator::DomainMap noDomains;

    CHECK_THROWS_AS(IntervalEvaluator(&tree, noDomains).Evaluate({}), std::runtime_error);
    CHECK_THROWS_AS(AffineEvaluator(&tree, noDomains).Evaluate({}), std::runtime_error);
}

#ifdef HAVE_ASMJIT
// --- The JIT-specific miss behavior this commit changes: CompileAVX2()
// returns nullptr (not a propagating throw) for an unmapped op, so
// JitEvaluator can fall back to the interpreter. Also confirms the fix
// doesn't regress into the retry-forever bug: a second CompileAVX2() call
// (of an all-built-in tree) still succeeds afterward.
TEST_CASE("JIT CompileAVX2: unmapped op degrades to nullptr, doesn't brick later compiles", "[registry][jit][miss]")
{
    constexpr Operon::Hash unmappedHash{0x4444444444444444ULL};
    JIT::JitRuntimePool pool;
    if (!pool.HasAVX2()) { return; } // matches CompileAVX2's own guard

    auto constNode = Node::Constant(1.0F);
    constNode.Optimize = false;
    Tree const badTree = Tree({ constNode, Node::Function(unmappedHash, 1) }).UpdateNodes();

    JIT::TreeCompiler compiler(&pool);
    auto compiled = compiler.CompileAVX2(badTree);
    CHECK(compiled == nullptr);

    auto expNode = Util::MakeOp<BuiltinOp::Exp>();
    Tree const goodTree = Tree({ constNode, expNode }).UpdateNodes();
    auto compiledGood = compiler.CompileAVX2(goodTree);
    CHECK(compiledGood != nullptr);
}
#endif

} // namespace Operon::Test
