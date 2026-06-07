// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#ifdef HAVE_ASMJIT

#include <asmjit/asmjit.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon::JIT {

// Compiled function signature.
//   out:    float[nRows]           -- output buffer
//   cols:   float const*[nVars]    -- column pointers, indexed by varOrder
//   nRows:  int32_t
//   consts: float const[nConsts]   -- optimizable coefficients, postfix order
using EvalFn = void(*)(float* out, float const* const* cols, int32_t nRows, float const* consts);

// A compiled tree function.  Non-copyable; owns the JIT-allocated code page.
struct CompiledTree {
    asmjit::JitRuntime& rt;
    EvalFn fn = nullptr;
    // varOrder[i] = hash of the variable whose column pointer is cols[i].
    // Caller builds cols[] using dataset_.GetValues(hash).data() + range.Start().
    std::vector<Operon::Hash> varOrder;

    explicit CompiledTree(asmjit::JitRuntime& rt_) noexcept : rt(rt_) {}
    ~CompiledTree() { if (fn) { rt.release(fn); } }
    CompiledTree(CompiledTree const&) = delete;
    CompiledTree& operator=(CompiledTree const&) = delete;
    CompiledTree(CompiledTree&&) = delete;
    CompiledTree& operator=(CompiledTree&&) = delete;
};

// JIT-compiles Operon trees to scalar native row-loops (Phase 1).
// One JitRuntime is shared across all trees compiled by this instance.
class TreeCompiler {
public:
    TreeCompiler() = default;

    // Returns nullptr on compilation failure.
    auto Compile(Operon::Tree const& tree) -> std::unique_ptr<CompiledTree>;

    auto Runtime() noexcept -> asmjit::JitRuntime& { return rt_; }

private:
    asmjit::JitRuntime rt_;
};

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
