// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#ifdef HAVE_ASMJIT

#include <asmjit/asmjit.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "operon/core/hash_registry.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon::JIT {

// Registered AVX2 codegen callbacks for unary/binary built-in or user-defined
// functions, keyed by Node::HashValue. Each callback emits asmjit
// instructions for one op's result into a fresh Vec (a ymm_ps register),
// given already-emitted operand Vec(s) — the same "emit into a stack slot"
// shape EmitNodesAvx2's old switch cases had, just as standalone callables.
// Scoped to asmjit's x86::Compiler virtual-register API only (never raw
// x86::Assembler), matching the codegen this file already emits everywhere
// else — a structural safety measure, not just a style choice: Compiler's
// virtual-register allocator (as opposed to Assembler's fixed physical
// registers) is what makes composing independently-written callbacks safe
// without them fighting over concrete registers.
//
// AVX2-only, no scalar/SSE counterpart: operon's own build unconditionally
// requires AVX2 hardware (-march=x86-64-v3 on Linux, /arch:AVX2 on the
// Windows preset), so the scalar JIT path this file used to also carry was
// dead code on every officially-built platform — deleted rather than
// ported to a registry.
using JitUnaryCodegenFn  = std::function<asmjit::x86::Vec(asmjit::x86::Compiler&, asmjit::x86::Vec const&)>;
using JitBinaryCodegenFn = std::function<asmjit::x86::Vec(asmjit::x86::Compiler&, asmjit::x86::Vec const&, asmjit::x86::Vec const&)>;

using JitUnaryCodegenRegistry  = HashRegistry<JitUnaryCodegenFn>;
using JitBinaryCodegenRegistry = HashRegistry<JitBinaryCodegenFn>;

// Register an AVX2 codegen callback for a unary function (built-in or
// user-defined), keyed by the same hash the function's Node::HashValue
// carries. A miss at compile time means that tree isn't JIT-compilable —
// CompileAVX2/CompileJacobian fall back to returning nullptr (the existing
// JitEvaluator caller already treats a nullptr compile result as "use the
// interpreter for this tree"), not a thrown exception reaching the caller.
// Throws if `hash` is already registered (write-once).
OPERON_EXPORT void RegisterUnaryJitCodegen(Operon::Hash hash, JitUnaryCodegenFn fn);

// Register an AVX2 codegen callback for a binary function. See
// RegisterUnaryJitCodegen for the miss-behavior note.
OPERON_EXPORT void RegisterBinaryJitCodegen(Operon::Hash hash, JitBinaryCodegenFn fn);

// Query whether an AVX2 codegen callback is registered for `hash` (built-in
// or user-defined), forcing built-in registration first. Mainly useful for
// coverage checks (e.g. asserting every BuiltinOp is either registered here
// or deliberately excluded as a structural n-ary case handled directly in
// EmitNodesAvx2).
OPERON_EXPORT auto HasUnaryJitCodegen(Operon::Hash hash) -> bool;
OPERON_EXPORT auto HasBinaryJitCodegen(Operon::Hash hash) -> bool;

// Compiled forward-pass signature.
//   out:    float[nRows]
//   cols:   float const*[nVars]  — indexed by VarOrder(tree)
//   nRows:  int32_t
//   consts: float const[nConsts]
using EvalFn = void(*)(float* out, float const* const* cols, int32_t nRows, float const* consts);

// Compiled Jacobian signature. outs[k][row] = ∂f/∂c_k(row).
//   outs:   float*[nConsts]
//   cols:   float const*[nVars]  — indexed by VarOrder(tree)
//   nRows:  int32_t
//   consts: float const[nConsts]
using EvalJacFn = void(*)(float* const* outs, float const* const* cols, int32_t nRows, float const* consts);

// Returns the unique variables of a tree in first-occurrence (postfix) order.
// This is the column ordering that compiled functions expect for their cols[] argument.
// Derivable from the tree at any time — no need to store it alongside the compiled code.
inline auto VarOrder(Operon::Tree const& tree) -> std::vector<Operon::Hash> {
    std::vector<Operon::Hash> order;
    for (auto const& n : tree.Nodes()) {
        if (!n.IsVariable()) { continue; }
        if (std::ranges::find(order, n.HashValue) == order.end()) {
            order.push_back(n.HashValue);
        }
    }
    return order;
}

// Holds one or both compiled functions for a single structural hash.
// fn is compiled first (by GetOrCompile); jacFn is added lazily (by GetOrCompileJacobian).
// rt* point into the JitRuntimePool owned by JitZobrist — must outlive this object.
struct CompileMeta {
    asmjit::JitRuntime* rtTree = nullptr;
    asmjit::JitRuntime* rtJac  = nullptr;
    EvalFn    fn    = nullptr;
    EvalJacFn jacFn = nullptr;
    int nVars   = 0;
    int nConsts = 0;

    CompileMeta() = default;

    ~CompileMeta() {
        if (fn    != nullptr && rtTree != nullptr) { rtTree->release(reinterpret_cast<void*>(fn));    } // NOLINT(*reinterpret-cast*)
        if (jacFn != nullptr && rtJac  != nullptr) { rtJac ->release(reinterpret_cast<void*>(jacFn)); } // NOLINT(*reinterpret-cast*)
    }

    CompileMeta(CompileMeta const&)            = delete;
    CompileMeta& operator=(CompileMeta const&) = delete;

    CompileMeta(CompileMeta&& o) noexcept
        : rtTree(o.rtTree), rtJac(o.rtJac), fn(o.fn), jacFn(o.jacFn)
        , nVars(o.nVars), nConsts(o.nConsts)
    { o.rtTree = o.rtJac = nullptr; o.fn = nullptr; o.jacFn = nullptr; }

    CompileMeta& operator=(CompileMeta&&) = delete;
};

// Pool of K independent JitRuntimes. Each runtime has its own JitAllocator mutex.
// Lifetime rule: must outlive every CompileMeta that was produced from it.
// JitZobrist declares pool_ before cache_, guaranteeing correct destruction order.
struct JitRuntimePool {
    static constexpr int PoolSize = 8;

    mutable std::array<asmjit::JitRuntime, PoolSize> runtimes;
    mutable std::atomic<unsigned> next{0};

    auto pick() const noexcept -> asmjit::JitRuntime& {
        return runtimes[next.fetch_add(1U, std::memory_order_relaxed) % static_cast<unsigned>(PoolSize)];
    }

    [[nodiscard]] auto HasAVX2() const noexcept -> bool {
        return runtimes[0].cpu_features().x86().has(asmjit::CpuFeatures::X86::kAVX2);
    }
};

// Stateless JIT code generator. All runtime state lives in the JitRuntimePool
// provided at construction; TreeCompiler can be destroyed before the pool.
//
// Every transcendental op it compiles (Exp/Log/Pow/etc., see the
// JitUnaryCodegenFn/JitBinaryCodegenFn registrations in jit_compiler.cpp)
// calls the Eve backend's approximate math functions directly, regardless
// of which MATH_BACKEND the rest of the library was built with. A tree
// compiled here always computes Eve-equivalent results, even when
// MATH_BACKEND is Stl, MadEve, or Eigen — comparing its output against a
// reference Interpreter built with a different MATH_BACKEND will show
// small, expected numeric differences, not a JIT bug.
class OPERON_EXPORT TreeCompiler {
public:
    explicit TreeCompiler(JitRuntimePool const* pool) : pool_(pool) {}

    // AVX2 vectorized path (8 rows/iter). No scalar/SSE fallback path
    // (deleted along with EmitNodesScalar — dead code given operon's own
    // build-wide AVX2 requirement). Returns nullptr if AVX2 unavailable, an
    // op has no registered codegen, or compile fails for any other reason —
    // all three degrade the same way, to interpreter fallback at the
    // JitEvaluator layer.
    auto CompileAVX2(Operon::Tree const& tree) -> std::unique_ptr<CompileMeta>;

    // Compiles all ∂f/∂c_k. Returns nullptr if AVX2 unavailable, no roots, or compile fails.
    auto CompileJacobian(JacobianDag const& dag) -> std::unique_ptr<CompileMeta>;

    [[nodiscard]] auto HasAVX2() const noexcept -> bool { return pool_->HasAVX2(); }

private:
    JitRuntimePool const* pool_;

    auto pick() const noexcept -> asmjit::JitRuntime& { return pool_->pick(); }
};

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
