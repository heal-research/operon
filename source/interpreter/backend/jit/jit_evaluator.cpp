// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/operators/evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>

#include <gtl/phmap.hpp>

namespace Operon::JIT {

struct JitEvaluator::JitCacheImpl {
    gtl::parallel_flat_hash_map_m<Operon::Hash, std::shared_ptr<CompiledTree>> Map;
};

JitEvaluator::JitEvaluator(gsl::not_null<Problem const*> problem,
                             gsl::not_null<Zobrist const*>  zobrist,
                             ErrorMetric                    error,
                             bool                           linearScaling)
    : EvaluatorBase(problem)
    , zobrist_(zobrist)
    , error_(std::move(error))
    , scaling_(linearScaling)
    , cache_(std::make_unique<JitCacheImpl>())
{}

JitEvaluator::~JitEvaluator() = default;

auto JitEvaluator::GetOrCompile(Tree const& tree) const -> std::shared_ptr<CompiledTree>
{
    return GetOrCompile(tree, zobrist_->ComputeHash(tree));
}

auto JitEvaluator::GetOrCompile(Tree const& tree, Hash hash) const -> std::shared_ptr<CompiledTree>
{
    std::shared_ptr<CompiledTree> ptr;

    // Fast path: already in cache.
    cache_->Map.if_contains(hash, [&](auto const& kv) { ptr = kv.second; });
    if (ptr) { ++hits_; return ptr; }

    // Slow path: compile then insert.  lazy_emplace_l holds the segment lock,
    // so only one thread compiles for a given hash simultaneously.
    cache_->Map.lazy_emplace_l(
        hash,
        [&](auto& kv) { ptr = kv.second; ++hits_; },   // another thread beat us
        [&](auto const& ctor) {
            auto compiled = compiler_.CompileAVX2(tree);
            if (!compiled) { compiled = compiler_.Compile(tree); }
            ptr = std::move(compiled);
            ctor(hash, ptr);
        }
    );
    return ptr;
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
        auto const  nVars = compiled->varOrder.size();
        auto const  nRows = static_cast<int32_t>(range.Size());

        std::vector<float const*> colPtrs(nVars);
        for (std::size_t i = 0; i < nVars; ++i) {
            colPtrs[i] = dataset->GetValues(compiled->varOrder[i]).data()
                         + static_cast<std::ptrdiff_t>(range.Start());
        }

        auto coeff = tree.GetCoefficients();
        compiled->fn(buf.data(), colPtrs.data(), nRows,
                     coeff.empty() ? nullptr : coeff.data());
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
