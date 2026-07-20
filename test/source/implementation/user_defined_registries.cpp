// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

// Exercises the "non-built-in functions participate in diff/interval-affine/
// JIT for the first time" claim from an actual external-caller perspective,
// not just by reading the migrated built-in rules: registers a single
// user-defined function ("recip", f(x) = 1/x) through all four registries
// (numeric DispatchTable, symbolic diff, interval, affine, and — if built —
// JIT codegen) and checks each path independently against a known reference.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/symbol_library.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"

#ifdef HAVE_ASMJIT
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#endif

namespace Operon::Test {

namespace {

// Hash-consing convention private to tree_diff.cpp (MakeUnary/MakeBinary/
// GetConst are anonymous-namespace, not exported) hand-rolled here using
// only the public Node factories — this is exactly the amount of protocol
// knowledge an external caller needs: children are appended as Ref nodes
// (never inlined subtrees, so Node::Function(hash, arity)'s arity == the
// node's Length with no further bookkeeping), a binary op's farther child
// is appended before its nearer child, and any literal constant must have
// Optimize forced to false (a leaf's default ctor sets Optimize=true,
// which would make the derivative-DAG's own synthetic constant look like a
// tunable coefficient to anything counting them). The parallel hash vector
// `h` is read only by Deriv()'s own memo lookups within the same call
// (verified: `grep h\[ source/core/tree_diff.cpp` shows exactly three
// reads, all feeding Mix() for a memo key) — never compared against
// anything outside this one Deriv() invocation — so any distinct
// placeholder value here (dag.size() at push time) is safe; it costs
// missed CSE opportunities, not correctness.
struct HandRolledDagBuilder {
    Operon::Vector<Node>& dag;
    Operon::Vector<Operon::Hash>& h;

    auto PushRef(std::size_t target) -> std::size_t {
        dag.push_back(Node::Ref(static_cast<uint16_t>(target)));
        h.push_back(static_cast<Operon::Hash>(dag.size()));
        return dag.size() - 1;
    }
    auto PushConst(Scalar v) -> std::size_t {
        auto n = Node::Constant(v);
        n.Optimize = false;
        dag.push_back(n);
        h.push_back(static_cast<Operon::Hash>(dag.size()));
        return dag.size() - 1;
    }
    auto PushUnary(BuiltinOp op, std::size_t a) -> std::size_t {
        PushRef(a);
        dag.push_back(Node::Function(static_cast<Operon::Hash>(op), 1));
        h.push_back(static_cast<Operon::Hash>(dag.size()));
        return dag.size() - 1;
    }
    auto PushBinary(BuiltinOp op, std::size_t a, std::size_t b) -> std::size_t {
        PushRef(b); // farther child first — matches MakeBinary's documented layout
        PushRef(a); // nearer child second
        dag.push_back(Node::Function(static_cast<Operon::Hash>(op), 2));
        h.push_back(static_cast<Operon::Hash>(dag.size()));
        return dag.size() - 1;
    }
};

} // namespace

TEST_CASE("User-defined function via registries: recip(x) = 1/x", "[registry][user-defined]")
{
    auto const hash = Operon::Hasher{}("recip");
    REQUIRE(hash >= Operon::BuiltinOpCount); // sanity: lands outside the built-in range

    // 1. Numeric evaluation — RegisterUnaryFunction (existing DispatchTable
    // API, pre-dates this PR); registers with no explicit derivative, so
    // Jet<T,1> auto-diff backs coefficient optimization if this function is
    // ever tuned — independent of the symbolic-diff registry tested below.
    Operon::ScalarDispatch dtable;
    PrimitiveSet pset;
    pset.SetConfig(NodeType::Constant | NodeType::Variable | BuiltinOp::Add | BuiltinOp::Mul);
    RegisterUnaryFunction<Operon::ScalarDispatch, Operon::Scalar>(
        dtable, pset, { .Name = "recip", .Desc = "f(x) = 1/x", .Arity = 1, .Frequency = 1 },
        [](auto v) { return decltype(v){1} / v; });

    auto tree = Tree({ Node::Constant(2.0), Node::Function(hash, 1) }).UpdateNodes();
    tree.Nodes()[0].Optimize = true;

    Dataset const ds(std::vector<std::string>{ "dummy" }, std::vector<std::vector<Operon::Scalar>>{ { 0.0F } });
    Interpreter<Operon::Scalar, Operon::ScalarDispatch> interp(&dtable, &ds, &tree);
    auto result = interp.Evaluate(tree.GetCoefficients(), Range{ 0, 1 });
    CHECK(result[0] == Catch::Approx(0.5).epsilon(1e-5)); // 1/2

    // 2. Symbolic differentiation — RegisterUnarySymbolicDeriv, the
    // registry this PR adds. d(1/x)/dx = -1/x^2, built by hand per the
    // HandRolledDagBuilder rationale above (no private helper needed).
    RegisterUnarySymbolicDeriv(hash,
        [](Operon::Vector<Node>& dag, Operon::Map<Operon::Hash, std::size_t>& /*memo*/,
           Operon::Vector<Operon::Hash>& h, std::size_t /*i*/, std::size_t j) -> std::size_t {
            HandRolledDagBuilder b{ dag, h };
            auto negOne = b.PushConst(Scalar{ -1 });
            auto sq     = b.PushUnary(BuiltinOp::Square, j);
            return b.PushBinary(BuiltinOp::Div, negOne, sq); // -1 / x^2, non-commutative — exercises binary child ordering
        });
    REQUIRE(HasUnarySymbolicDeriv(hash));

    auto jdag = BuildJacobianDag(tree);
    REQUIRE(jdag.Roots.size() == 1);
    REQUIRE(jdag.Roots[0] != std::numeric_limits<std::size_t>::max()); // not Zero

    // The dag's flat Node array follows the same postfix/Ref convention a
    // plain Tree does (original tree at [0, OriginalSize), derivative
    // subtrees appended after) — sliceable into a standalone, directly
    // evaluable Tree.
    Operon::Vector<Node> sliced(jdag.Nodes.begin(), jdag.Nodes.begin() + static_cast<std::ptrdiff_t>(jdag.Roots[0]) + 1);
    Tree derivTree(std::move(sliced));
    derivTree.UpdateNodes();
    auto derivCoeff = derivTree.GetCoefficients(); // the constant's Optimize flag was preserved from the original tree
    Interpreter<Operon::Scalar, Operon::ScalarDispatch> derivInterp(&dtable, &ds, &derivTree);
    auto derivResult = derivInterp.Evaluate(derivCoeff, Range{ 0, 1 });
    CHECK(derivResult[0] == Catch::Approx(-0.25).epsilon(1e-4)); // -1/2^2

    // 3. Interval bound propagation — RegisterUnaryInterval.
    RegisterUnaryInterval(hash, [](IntervalEvaluator::Interval const& v) {
        return IntervalEvaluator::Interval{ Scalar{1} } / v;
    });
    {
        // Interval/affine evaluators don't consult a Dataset for hash
        // assignment (unlike the JIT section below) — any self-chosen
        // hash is fine as long as the Variable node and DomainMap key agree.
        auto varHash = Operon::Hash{ 1 };
        Node var(NodeType::Variable, varHash); var.Value = 1.0F;
        auto ivTree = Tree({ var, Node::Function(hash, 1) }).UpdateNodes();
        IntervalEvaluator::DomainMap dm;
        dm[varHash] = { Scalar{ 1 }, Scalar{ 4 } };
        IntervalEvaluator ivEval(&ivTree, std::move(dm));
        auto iv = ivEval.Evaluate(ivTree.GetCoefficients());
        CHECK(iv.inf() <= 0.25 + 1e-4);
        CHECK(iv.sup() + 1e-4 >= 1.0);
    }

    // 4. Affine bound propagation — RegisterUnaryAffine. Unlike interval's
    // pappus::interval (which has a scalar-value ctor), affine_form has no
    // context-free literal constructor (every affine_form is tied to the
    // affine_context that allocated its epsilon terms) — but its friend
    // `operator/(T, affine_form const&)` exists precisely to avoid needing
    // one here, so the ctx parameter genuinely goes unused for this rule.
    RegisterUnaryAffine(hash,
        [](AffineEvaluator::Context const&, AffineEvaluator::Affine const& v) {
            return Scalar{1} / v;
        });
    {
        auto varHash = Operon::Hash{ 2 };
        Node var(NodeType::Variable, varHash); var.Value = 1.0F;
        auto afTree = Tree({ var, Node::Function(hash, 1) }).UpdateNodes();
        AffineEvaluator::DomainMap dm;
        dm[varHash] = { Scalar{ 1 }, Scalar{ 4 } };
        AffineEvaluator afEval(&afTree, std::move(dm));
        auto af = afEval.Evaluate(afTree.GetCoefficients());
        auto ivFromAffine = af.to_interval();
        CHECK(ivFromAffine.inf() <= 0.25 + 1e-2);
        CHECK(ivFromAffine.sup() + 1e-2 >= 1.0);
    }

#ifdef HAVE_ASMJIT
    // 5. JIT codegen — RegisterUnaryJitCodegen. jit_compiler.cpp has its
    // own private BroadcastFloat() helper for exactly this pattern
    // (mov bit-pattern -> GP32, vmovd -> xmm, vbroadcastss -> ymm) but it's
    // anonymous-namespace, not exported — replicated here with only public
    // asmjit::x86::Compiler calls (the virtual-register API only, never
    // raw Assembler, per the registry's documented safety contract).
    JIT::RegisterUnaryJitCodegen(hash, [](asmjit::x86::Compiler& cc, asmjit::x86::Vec const& a) {
        uint32_t bits{};
        float const one = 1.0F;
        std::memcpy(&bits, &one, sizeof bits);
        auto tmp = cc.new_gp32();
        cc.mov(tmp, bits);
        auto xmm = cc.new_xmm_ss();
        cc.vmovd(xmm, tmp);
        auto ones = cc.new_ymm_ps();
        cc.vbroadcastss(ones, xmm);
        auto res = cc.new_ymm_ps();
        cc.vdivps(res, ones, a);
        return res;
    });
    REQUIRE(JIT::HasUnaryJitCodegen(hash));

    {
        // End-to-end: compile a tree using the custom op and compare its
        // AVX2-JIT output against the interpreter, across a small batch.
        JIT::JitRuntimePool pool;
        if (pool.HasAVX2()) {
            // Dataset assigns each named variable's hash itself (Hasher{}
            // applied to the name) — read it back rather than assume one,
            // so the tree's Variable node and the dataset's column lookup
            // agree.
            Dataset const jitDs(std::vector<std::string>{ "V" },
                std::vector<std::vector<Operon::Scalar>>{ { 1.0F, 2.0F, 4.0F, 0.5F } });
            auto varHash = jitDs.GetVariable("V").value().Hash;
            Node var(NodeType::Variable, varHash); var.Value = 1.0F;
            auto jitTree = Tree({ var, Node::Function(hash, 1) }).UpdateNodes();

            Interpreter<Operon::Scalar, Operon::ScalarDispatch> refInterp(&dtable, &jitDs, &jitTree);
            auto ref = refInterp.Evaluate(jitTree.GetCoefficients(), Range{ 0, jitDs.Rows<std::size_t>() });

            JIT::TreeCompiler compiler(&pool);
            auto compiled = compiler.CompileAVX2(jitTree);
            REQUIRE(compiled != nullptr);
            REQUIRE(compiled->fn != nullptr);

            // Invoke the compiled function directly (same calling
            // convention jit.cpp's own tests use — nRows padded to a
            // multiple of 8, columns in JIT::VarOrder(tree)'s order).
            Range const jitRange{ 0, jitDs.Rows<std::size_t>() };
            auto const nRows    = static_cast<int32_t>(jitRange.Size());
            auto const nRowsPad = (nRows + 7) & ~7;
            auto const varOrder = JIT::VarOrder(jitTree);
            std::vector<float const*> colPtrs(varOrder.size());
            for (std::size_t i = 0; i < varOrder.size(); ++i) {
                colPtrs[i] = jitDs.GetPaddedValues(varOrder[i]) + jitRange.Start();
            }
            auto jitCoeff = jitTree.GetCoefficients();
            std::vector<float> scratch(static_cast<std::size_t>(nRowsPad));
            compiled->fn(scratch.data(), colPtrs.data(), nRowsPad,
                         jitCoeff.empty() ? nullptr : jitCoeff.data());

            REQUIRE(ref.size() == static_cast<std::size_t>(nRows));
            for (std::size_t i = 0; i < ref.size(); ++i) {
                INFO("row " << i << " ref=" << ref[i] << " jit=" << scratch[i]);
                CHECK(scratch[i] == Catch::Approx(ref[i]).epsilon(1e-4));
            }
        }
    }
#endif
}

} // namespace Operon::Test
