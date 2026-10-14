// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#ifdef HAVE_ASMJIT

#include <atomic>
#include <memory>

#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operon_export.hpp"

namespace Operon::JIT {

// ---- JIT-specific cache entry components -----------------------------------

// Per-hash visit counter used by the frequency gate (compile only after N visits).
struct VisitData {
    std::size_t Visits { 0 };
};

// Holds the compiled fn + jacFn for one structural hash.
struct MetaData {
    std::unique_ptr<CompileMeta> meta;
};

using JitEntry = Operon::CacheEntry<VisitData, MetaData>;

// Extends Zobrist with a compiled-function cache (JitEntry map) and a pool
// of JitRuntimes whose code pages back all cached CompileMeta objects.
//
// Declaration order is load-bearing: pool_ is declared BEFORE cache_ so that
// it is destroyed AFTER cache_ (C++ destroys members in reverse order).
// This guarantees CompileMeta::~CompileMeta() can safely call rt->release()
// during cache teardown.
class OPERON_EXPORT JitZobrist final : public Operon::Zobrist {
    JIT::JitRuntimePool pool_; // destroyed last
    mutable Operon::ZobristCache<JitEntry> cache_; // destroyed first
public:
    JitZobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes,
        std::size_t maxAge = 0);
    ~JitZobrist() override = default;

    [[nodiscard]] auto JitCache() const -> Operon::ZobristCache<JitEntry>& { return cache_; }
    [[nodiscard]] auto Pool() const noexcept -> JIT::JitRuntimePool const& { return pool_; }
};

// Evaluator that replaces the interpreter forward pass with a JIT-compiled
// function, keyed by Zobrist structural hash.  Structurally identical trees
// share the same compiled function; only the coefficient array and column
// pointers differ per call.  Column pointers are rebuilt from the tree at
// call time (VarOrder) — no per-entry varOrder storage needed.
//
// It is an Evaluator<ScalarDispatch> that overrides only the prediction step
// (`Predict`): evaluation counting, the linear-scaling fit (under the
// problem's declared Problem::LinearScalingOmitsNonFinite policy), the error
// metric, and skip-nonfinite scoring are the interpreter evaluator's own, so
// JIT and interpreter results follow identical scoring and fit policy.
//
// Thread-safe for concurrent evaluation and compilation. ClearCache() and
// configuration changes require all evaluations using this instance to stop.
// AVX2 is attempted first; falls back to the scalar path on older CPUs.
//
// Not sealed: it is a value evaluator that ShapeConstrainedValueEvaluator derives from. The derived adapter is
// itself a JitEvaluator, so JitLevenbergMarquardtOptimizer keeps using the same JIT instance and code cache.
class OPERON_EXPORT JitEvaluator : public Operon::Evaluator<Operon::ScalarDispatch> {
public:
    JitEvaluator(
        gsl::not_null<Problem const*> problem, gsl::not_null<JitZobrist const*> zobrist, ErrorMetric error = MSE {});

    ~JitEvaluator() override;

    JitEvaluator(JitEvaluator const&) = delete;
    JitEvaluator& operator=(JitEvaluator const&) = delete;
    JitEvaluator(JitEvaluator&&) = delete;
    JitEvaluator& operator=(JitEvaluator&&) = delete;

    [[nodiscard]] auto CacheSize() const -> std::size_t;
    // Counts only lookups that find an already-compiled forward function.
    [[nodiscard]] auto CacheHits() const -> std::size_t { return cacheHits_.load(); }
    // Counts JIT-eligible forward lookups with no compiled forward function.
    [[nodiscard]] auto CacheMisses() const -> std::size_t { return cacheMisses_.load(); }
    // Counts successful forward compilations initiated by this evaluator.
    [[nodiscard]] auto CompileSuccesses() const -> std::size_t { return compileSuccesses_.load(); }

    void ResetCounters(); // not thread-safe; call only when no evaluations are in flight
    // Releases cached code. Call only when no returned CompileMeta pointer is
    // executing or otherwise retained.
    void ClearCache();

    void SetMaxLength(int maxLength) { maxLength_ = maxLength; }
    [[nodiscard]] auto MaxLength() const -> int { return maxLength_; }

    void SetMinVisits(std::size_t minVisits) { minVisits_ = minVisits; }
    [[nodiscard]] auto MinVisits() const -> std::size_t { return minVisits_; }

    // Returns a CompileMeta with fn set (compiling on first call past the frequency gate).
    // Returns nullptr if the tree is above maxLength_, below minVisits_, or compilation fails.
    // Valid for the lifetime of JitZobrist.
    [[nodiscard]] auto GetOrCompile(Tree const& tree) const -> CompileMeta const*;

    // Compiles the Jacobian and stores it in the existing cache entry (or creates one).
    // Does NOT guarantee that fn is set — call GetOrCompile separately if the forward pass
    // is also needed.  Returns nullptr only if CompileJacobian fails (e.g. non-AVX2 CPU).
    [[nodiscard]] auto GetOrCompileJacobian(Tree const& tree) const -> CompileMeta const*;

    [[nodiscard]] auto Avx2Fails() const -> std::size_t { return avx2Fails_.load(); }
    [[nodiscard]] auto CompileFails() const -> std::size_t { return compileFails_.load(); }

protected:
    // JIT-compiled (or interpreter-fallback) forward pass filling `values` with the genotype's raw output.
    auto Predict(Operon::Individual const& ind, Operon::Span<Operon::Scalar> values) const
        -> tl::expected<void, Operon::InterpreterError> override;

private:
    [[nodiscard]] auto GetOrCompile(Tree const& tree, Hash hash) const -> CompileMeta const*;

    gsl::not_null<JitZobrist const*> zobrist_;

    mutable TreeCompiler compiler_;

    int maxLength_ { 0 };
    std::size_t minVisits_ { 1 };

    mutable std::atomic<std::size_t> cacheHits_ { 0 };
    mutable std::atomic<std::size_t> cacheMisses_ { 0 };
    mutable std::atomic<std::size_t> compileSuccesses_ { 0 };
    mutable std::atomic<std::size_t> avx2Fails_ { 0 };
    mutable std::atomic<std::size_t> compileFails_ { 0 };
};

static_assert(Operon::Concepts::ValueEvaluator<JitEvaluator>);

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
