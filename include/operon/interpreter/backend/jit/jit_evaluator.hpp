// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#pragma once

#ifdef HAVE_ASMJIT

#include <atomic>
#include <memory>

#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operon_export.hpp"

namespace Operon::JIT {

// Evaluator that replaces the interpreter forward pass with a JIT-compiled
// function, keyed by Zobrist structural hash.  Structurally identical trees
// share the same compiled function; only the coefficient array (consts[]) and
// column pointers differ per call.
//
// Compilation is lazy: first call for a new structure compiles and inserts;
// subsequent calls with the same structural hash are O(1) map lookups.
// Thread-safe: uses gtl::parallel_flat_hash_map_m internally.
//
// AVX2 is attempted first; falls back to the scalar Phase-1 path on machines
// without AVX2.
class OPERON_EXPORT JitEvaluator final : public EvaluatorBase {
public:
    JitEvaluator(gsl::not_null<Problem const*> problem,
                 gsl::not_null<Zobrist const*>  zobrist,
                 ErrorMetric                    error         = MSE{},
                 bool                           linearScaling = true);

    ~JitEvaluator() override;

    JitEvaluator(JitEvaluator const&)            = delete;
    JitEvaluator& operator=(JitEvaluator const&) = delete;
    JitEvaluator(JitEvaluator&&)                 = delete;
    JitEvaluator& operator=(JitEvaluator&&)      = delete;

    auto operator()(RandomGenerator& rng, Individual const& ind, Span<Scalar> buf) const -> ReturnType override;
    auto operator()(RandomGenerator& rng, Individual const& ind)                  const -> ReturnType override;

    [[nodiscard]] auto CacheSize() const -> std::size_t;
    [[nodiscard]] auto CacheHits() const -> std::size_t { return hits_.load(); }

    void ClearCache();

    // Returns the compiled forward function for `tree` (compiling on first call).
    // Public so that JitLevenbergMarquardtOptimizer can share the same cache.
    [[nodiscard]] auto GetOrCompile(Tree const& tree) const -> std::shared_ptr<CompiledTree>;

    // Returns the compiled Jacobian function for `tree` (compiling on first call).
    // Keyed by the same structural hash as GetOrCompile. Thread-safe.
    [[nodiscard]] auto GetOrCompileJacobian(Tree const& tree) const -> std::shared_ptr<CompiledJacobian>;

private:
    [[nodiscard]] auto GetOrCompile(Tree const& tree, Hash hash) const -> std::shared_ptr<CompiledTree>;

    gsl::not_null<Zobrist const*> zobrist_;
    ErrorMetric                   error_;
    bool                          scaling_;

    mutable TreeCompiler compiler_;

    struct CacheImpl;
    mutable std::unique_ptr<CacheImpl> cache_;
    mutable std::atomic<std::size_t> hits_{0};
    mutable std::atomic<std::size_t> avx2Fails_{0};    // AVX2 compile failed, scalar used
    mutable std::atomic<std::size_t> compileFails_{0}; // both paths failed, ErrMax used

public:
    [[nodiscard]] auto Avx2Fails()    const -> std::size_t { return avx2Fails_.load(); }
    [[nodiscard]] auto CompileFails() const -> std::size_t { return compileFails_.load(); }
};

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
