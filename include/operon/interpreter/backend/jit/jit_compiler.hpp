// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#ifdef HAVE_ASMJIT

#include <asmjit/asmjit.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/types.hpp"

namespace Operon::JIT {

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

    CompileMeta() = default;

    ~CompileMeta() {
        if (fn    != nullptr && rtTree != nullptr) { rtTree->release(reinterpret_cast<void*>(fn));    } // NOLINT(*reinterpret-cast*)
        if (jacFn != nullptr && rtJac  != nullptr) { rtJac ->release(reinterpret_cast<void*>(jacFn)); } // NOLINT(*reinterpret-cast*)
    }

    CompileMeta(CompileMeta const&)            = delete;
    CompileMeta& operator=(CompileMeta const&) = delete;

    CompileMeta(CompileMeta&& o) noexcept
        : rtTree(o.rtTree), rtJac(o.rtJac), fn(o.fn), jacFn(o.jacFn)
    { o.rtTree = o.rtJac = nullptr; o.fn = nullptr; o.jacFn = nullptr; }

    CompileMeta& operator=(CompileMeta&&) = delete;

    // Steal the jacFn/rtJac from another CompileMeta (which must have fn==nullptr).
    void AcceptJac(CompileMeta& other) noexcept {
        if (jacFn == nullptr && other.jacFn != nullptr) {
            jacFn  = other.jacFn;  other.jacFn  = nullptr;
            rtJac  = other.rtJac;  other.rtJac  = nullptr;
        }
    }
};

// Pool of K independent JitRuntimes. Each runtime has its own JitAllocator mutex.
// Lifetime rule: must outlive every CompileMeta that was produced from it.
// JitZobrist declares pool_ before cache_, guaranteeing correct destruction order.
struct JitRuntimePool {
    static constexpr int kPoolSize = 8;

    mutable std::array<asmjit::JitRuntime, kPoolSize> runtimes;
    mutable std::atomic<unsigned> next{0};

    auto pick() const noexcept -> asmjit::JitRuntime& {
        return runtimes[next.fetch_add(1U, std::memory_order_relaxed) % static_cast<unsigned>(kPoolSize)];
    }

    [[nodiscard]] auto HasAVX2() const noexcept -> bool {
        return runtimes[0].cpu_features().x86().has(asmjit::CpuFeatures::X86::kAVX2);
    }
};

// Stateless JIT code generator. All runtime state lives in the JitRuntimePool
// provided at construction; TreeCompiler can be destroyed before the pool.
class TreeCompiler {
public:
    explicit TreeCompiler(JitRuntimePool const* pool) : pool_(pool) {}

    // Scalar path. Returns nullptr on failure.
    auto Compile(Operon::Tree const& tree) -> std::unique_ptr<CompileMeta>;

    // AVX2 vectorized path (8 rows/iter). Returns nullptr if AVX2 unavailable or compile fails.
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
