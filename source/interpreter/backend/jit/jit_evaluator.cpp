// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/operators/evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>

#include <gtl/phmap.hpp>

namespace Operon::JIT {

struct CacheEntry {
    std::shared_ptr<CompiledTree>     Tree;
    std::shared_ptr<CompiledJacobian> Jac;
};

struct JitEvaluator::CacheImpl {
    gtl::parallel_flat_hash_map_m<Operon::Hash, CacheEntry> Map;
};

JitEvaluator::JitEvaluator(gsl::not_null<Problem const*> problem,
                             gsl::not_null<Zobrist const*>  zobrist,
                             ErrorMetric                    error,
                             bool                           linearScaling)
    : EvaluatorBase(problem)
    , zobrist_(zobrist)
    , error_(std::move(error))
    , scaling_(linearScaling)
    , cache_(std::make_unique<CacheImpl>())
{}

JitEvaluator::~JitEvaluator() = default;

auto JitEvaluator::GetOrCompile(Tree const& tree) const -> std::shared_ptr<CompiledTree>
{
    return GetOrCompile(tree, zobrist_->ComputeHash(tree));
}

auto JitEvaluator::GetOrCompile(Tree const& tree, Hash hash) const -> std::shared_ptr<CompiledTree>
{
    CacheEntry entry;

    // Fast path: already in cache.
    cache_->Map.if_contains(hash, [&](auto const& kv) { entry = kv.second; });
    if (entry.Tree) { ++hits_; return entry.Tree; }

    // Compile outside the map lock so that cc.finalize() (the expensive part) runs
    // in parallel across threads.  rt_.add() is internally mutex-protected in asmjit,
    // so concurrent calls are safe.  A rare double-compile race is harmless: we just
    // keep whichever result arrives first in the map.
    std::shared_ptr<CompiledTree> compiled(compiler_.CompileAVX2(tree));
    if (!compiled) { ++avx2Fails_; compiled.reset(compiler_.Compile(tree).release()); }
    if (!compiled) { ++compileFails_; }

    cache_->Map.lazy_emplace_l(
        hash,
        [&](auto& kv) { entry = kv.second; ++hits_; },   // another thread beat us
        [&](auto const& ctor) { entry.Tree = compiled; ctor(hash, entry); }
    );
    return entry.Tree;
}

auto JitEvaluator::GetOrCompileJacobian(Tree const& tree) const -> std::shared_ptr<CompiledJacobian>
{
    auto const hash = zobrist_->ComputeHash(tree);

    // Fast path: Jacobian already compiled.
    std::shared_ptr<CompiledJacobian> jac;
    cache_->Map.if_contains(hash, [&](auto const& kv) { jac = kv.second.Jac; });
    if (jac) { return jac; }

    // Ensure the primal entry exists (compile it if needed).
    static_cast<void>(GetOrCompile(tree, hash));

    // Compile the Jacobian outside any lock (deterministic — a double-compile on
    // a very rare race is harmless).
    auto dag    = Operon::BuildJacobianDag(tree);
    auto newJac = compiler_.CompileJacobian(dag);

    // Store it; if another thread beat us, use their result.
    cache_->Map.modify_if(hash, [&](auto& kv) {
        if (!kv.second.Jac) { kv.second.Jac = std::move(newJac); }
        jac = kv.second.Jac;
    });
    return jac;
}

auto JitEvaluator::operator()(RandomGenerator& /*rng*/, Individual const& ind,
                               Span<Scalar> buf) const -> ReturnType
{
    ++CallCount;

    auto const* problem       = GetProblem();
    auto const* dataset       = problem->GetDataset();
    auto const  range         = problem->TrainingRange();
    auto const  targetValues  = problem->TargetValues(range);
    auto const  weightsOpt    = problem->Weights(range);
    auto const  weights       = weightsOpt.value_or(Span<Scalar const>{});

    auto const& tree    = ind.Genotype;
    auto const  hash    = zobrist_->ComputeHash(tree);
    auto const  compiled = GetOrCompile(tree, hash);

    ENSURE(buf.size() >= range.Size());
    ++ResidualEvaluations;

    if (compiled) {
        auto const  nVars      = compiled->varOrder.size();
        auto const  nRows      = static_cast<int32_t>(range.Size());
        auto const  nRowsPad   = (nRows + 7) & ~7;

        thread_local std::vector<float const*> colPtrs;
        colPtrs.resize(nVars);
        for (std::size_t i = 0; i < nVars; ++i) {
            colPtrs[i] = dataset->GetPaddedValues(compiled->varOrder[i])
                         + static_cast<std::ptrdiff_t>(range.Start());
        }

        // JIT function processes nRowsPad rows (always a multiple of 8, no scalar tail).
        // Use a thread-local scratch buffer so the extra 0-7 writes don't touch buf.
        thread_local std::vector<Scalar> scratch;
        scratch.resize(static_cast<std::size_t>(nRowsPad));

        thread_local std::vector<Scalar> coeff;
        tree.GetCoefficients(coeff);
        compiled->fn(scratch.data(), colPtrs.data(), nRowsPad,
                     coeff.empty() ? nullptr : coeff.data());
        std::copy_n(scratch.data(), nRows, buf.data());
    } else {
        // Compilation failed — fill with a large constant so the individual is culled.
        std::ranges::fill(buf, EvaluatorBase::ErrMax);
    }

    if (scaling_) {
        auto [a, b] = FitLeastSquares(Span<Scalar const>(buf.data(), buf.size()), targetValues, weights);
        std::ranges::transform(buf, buf.begin(),
            [a=a, b=b](auto x) -> Scalar { return static_cast<Scalar>(a * x + b); });
    }

    auto fit = static_cast<Scalar>(weights.empty()
        ? error_(buf, targetValues)
        : error_(buf, targetValues, weights));

    if (!std::isfinite(fit)) { fit = EvaluatorBase::ErrMax; }
    return ReturnType{ fit };
}

auto JitEvaluator::operator()(RandomGenerator& rng, Individual const& ind) const -> ReturnType
{
    return EvaluatorBase::Evaluate(this, rng, ind);
}

auto JitEvaluator::CacheSize() const -> std::size_t
{
    return cache_->Map.size();
}

void JitEvaluator::ClearCache()
{
    cache_->Map.clear();
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
