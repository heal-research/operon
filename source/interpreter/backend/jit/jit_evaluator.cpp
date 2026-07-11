// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/interpreter/interpreter.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace Operon::JIT {

JitZobrist::JitZobrist(Operon::RandomGenerator& rng, int maxLength,
                       Operon::Span<Operon::Hash const> variableHashes)
    : Zobrist(rng, maxLength, variableHashes)
{}


JitEvaluator::JitEvaluator(gsl::not_null<Problem const*>    problem,
                             gsl::not_null<JitZobrist const*> zobrist,
                             ErrorMetric                      error,
                             bool                             linearScaling)
    : EvaluatorBase(problem)
    , zobrist_(zobrist)
    , error_(error)
    , scaling_(linearScaling)
    , compiler_(&zobrist->Pool())
{}

JitEvaluator::~JitEvaluator() = default;

auto JitEvaluator::GetOrCompile(Tree const& tree) const -> CompileMeta const*
{
    return GetOrCompile(tree, zobrist_->ComputeHash(tree));
}

auto JitEvaluator::GetOrCompile(Tree const& tree, Hash hash) const -> CompileMeta const*
{
    if (maxLength_ > 0 && std::cmp_greater(tree.Length(), maxLength_)) { ++misses_; return nullptr; }

    // Fast path: already compiled.
    CompileMeta const* result{};
    if (zobrist_->JitCache().IfContains(hash, [&](JitEntry const& e) -> void {
            if (e.meta && e.meta->fn) { result = e.meta.get(); }
        }) && result != nullptr) {
        ++hits_;
        return result;
    }

    // Increment visit counter; return nullptr until frequency threshold is met.
    std::size_t visits{};
    zobrist_->JitCache().LazyEmplace(hash,
        [&](JitEntry& e) -> void { visits = ++e.Visits; },
        [&](JitEntry& e) -> void { visits = ++e.Visits; }
    );
    if (visits < minVisits_) { ++misses_; return nullptr; }

    // Compile outside the map lock so cc.finalize() runs in parallel.
    auto compiled = compiler_.CompileAVX2(tree);
    if (!compiled) { ++avx2Fails_; compiled = compiler_.Compile(tree); }
    if (!compiled) { ++compileFails_; }

    zobrist_->JitCache().ModifyIf(hash, [&](JitEntry& e) -> void {
        if (!e.meta) {
            e.meta = std::move(compiled);
        } else if (e.meta->fn == nullptr && compiled) {
            e.meta->fn      = compiled->fn;      compiled->fn      = nullptr;
            e.meta->rtTree  = compiled->rtTree;  compiled->rtTree  = nullptr;
            e.meta->nVars   = compiled->nVars;
            e.meta->nConsts = compiled->nConsts;
        }
        if (e.meta && e.meta->fn) { result = e.meta.get(); }
    });
    if (result != nullptr) { ++hits_; } else { ++misses_; }
    return result;
}

auto JitEvaluator::GetOrCompileJacobian(Tree const& tree) const -> CompileMeta const*
{
    auto const hash = zobrist_->ComputeHash(tree);

    // Fast path: Jacobian already compiled.
    CompileMeta const* meta{};
    if (zobrist_->JitCache().IfContains(hash, [&](JitEntry const& e) -> void {
            if (e.meta && e.meta->jacFn) { meta = e.meta.get(); }
        }) && meta != nullptr) {
        return meta;
    }

    // Ensure an entry exists and count this as a visit (consistent with GetOrCompile).
    // Jacobian compilation is not frequency-gated — it is only requested by the optimizer
    // for trees that have already passed selection, so compiling unconditionally is correct.
    zobrist_->JitCache().LazyEmplace(hash,
        [](JitEntry& e) -> void { ++e.Visits; },
        [](JitEntry& e) -> void { e.Visits = 1; });

    auto dag    = Operon::BuildJacobianDag(tree);
    auto newJac = compiler_.CompileJacobian(dag);

    zobrist_->JitCache().ModifyIf(hash, [&](JitEntry& e) -> void {
        if (!e.meta) {
            e.meta = std::move(newJac);
        } else if (e.meta->jacFn == nullptr && newJac && newJac->jacFn != nullptr) {
            e.meta->jacFn  = newJac->jacFn;  newJac->jacFn  = nullptr;
            e.meta->rtJac  = newJac->rtJac;  newJac->rtJac  = nullptr;
        }
        if (e.meta) { meta = e.meta.get(); }
    });
    return meta;
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

    auto const& tree = ind.Genotype;
    auto const  hash = zobrist_->ComputeHash(tree);
    CompileMeta const* compiled = GetOrCompile(tree, hash);

    ENSURE(buf.size() >= range.Size());
    ++ResidualEvaluations;

    if (compiled != nullptr) {
        auto const  nRows    = static_cast<int32_t>(range.Size());
        auto const  nRowsPad = (nRows + 7) & ~7; // NOLINT(hicpp-signed-bitwise)

        // Rebuild column pointers from the tree (VarOrder is re-derivable since
        // the Zobrist hash now structurally identifies each unique tree).
        thread_local std::vector<Hash>         varOrderBuf;
        thread_local std::vector<float const*> colPtrs;
        varOrderBuf = VarOrder(tree);
        colPtrs.resize(varOrderBuf.size());
        for (std::size_t i = 0; i < varOrderBuf.size(); ++i) {
            colPtrs[i] = dataset->GetPaddedValues(varOrderBuf[i])
                         + static_cast<std::ptrdiff_t>(range.Start());
        }

        thread_local std::vector<Scalar> scratch;
        scratch.resize(static_cast<std::size_t>(nRowsPad));

        thread_local std::vector<Scalar> coeff;
        tree.GetCoefficients(coeff);
        ENSURE(static_cast<int>(varOrderBuf.size()) == compiled->nVars);
        ENSURE(static_cast<int>(coeff.size()) == compiled->nConsts);
        compiled->fn(scratch.data(), colPtrs.data(), nRowsPad,
                     coeff.empty() ? nullptr : coeff.data());
        std::copy_n(scratch.data(), nRows, buf.data());
    } else {
        thread_local ScalarDispatch fallbackDtable;
        thread_local std::vector<Scalar> coeffBuf;
        tree.GetCoefficients(coeffBuf);
        Interpreter<Scalar, ScalarDispatch> const interp{&fallbackDtable, dataset, &tree};
        interp.Evaluate(Span<Scalar const>(coeffBuf.data(), coeffBuf.size()), range, buf);
    }

    if (scaling_) {
        auto [a, b] = FitLeastSquares(Span<Scalar const>(buf.data(), buf.size()), targetValues, weights);
        std::ranges::transform(buf, buf.begin(),
            [a=a, b=b](auto x) -> Scalar { return static_cast<Scalar>((a * x) + b); });
    }

    auto fit = static_cast<Scalar>(weights.empty()
        ? error_(buf, targetValues)
        : error_(buf, targetValues, weights));

    if (!std::isfinite(fit)) { fit = EvaluatorBase::ErrMax; }
    return ReturnType{ fit };
}

auto JitEvaluator::operator()(RandomGenerator& rng, Individual const& ind) const -> ReturnType
{
    return Evaluate(rng, ind);
}

auto JitEvaluator::CacheSize() const -> std::size_t
{
    return zobrist_->JitCache().Size();
}

void JitEvaluator::ClearCache()
{
    zobrist_->JitCache().Clear();
}

void JitEvaluator::ResetCounters()
{
    hits_.store(0);
    misses_.store(0);
    avx2Fails_.store(0);
    compileFails_.store(0);
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
