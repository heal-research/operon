// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#ifdef HAVE_ASMJIT

#include <vector>

#include <gsl/pointers>

#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/backend/jit/jit_compiler.hpp"
#include "operon/optimizer/lm_cost_function_base.hpp"

namespace Operon {

// LM cost function that replaces residual and Jacobian evaluation with JIT-compiled
// functions.  Residuals use the compiled forward pass; Jacobian uses the compiled
// derivative DAG (EvalJacFn) when available, falling back to interpreter JacRev.
//
// colPtrs[i] and jacColPtrs[i] must follow the ordering returned by VarOrder(tree),
// each already offset to range.Start().
template<typename T = Operon::Scalar, int StorageOrder = Eigen::ColMajor>
struct JitLMCostFunction : public LMCostFunctionBase<JitLMCostFunction<T, StorageOrder>, StorageOrder> {
    using Base = LMCostFunctionBase<JitLMCostFunction<T, StorageOrder>, StorageOrder>;
    using Scalar = typename Base::Scalar;

    static_assert(std::is_same_v<T, float>,
        "JitLMCostFunction requires float precision (EvalFn operates on float arrays)");

    // `target` and `weights` must span the *whole* dataset column (absolute,
    // dataset-row-indexed), same contract as GaussianLoss/PoissonLoss/LMCostFunction
    // - not a slice pre-cut to `range`. JIT LM never mini-batches (range_ is fixed
    // for the object's lifetime), so the local range.Size()-sized slice this cost
    // function actually reads is computed once here in the ctor, not per Evaluate().
    JitLMCostFunction(gsl::not_null<InterpreterBase<T> const*> interpreter,
                      JIT::EvalFn                              fn,
                      std::vector<float const*>                colPtrs,
                      Operon::Span<Operon::Scalar const>       target,
                      Operon::Range                            range,
                      JIT::EvalJacFn                           jacFn      = nullptr,
                      std::vector<float const*>                jacColPtrs = {},
                      int                                      nVars      = -1,
                      int                                      nConsts    = -1,
                      Operon::Span<Operon::Scalar const>       weights    = {})
        : Base{range.Size(), static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount())}
        , interpreter_(interpreter)
        , fn_(fn)
        , colPtrs_(std::move(colPtrs))
        , jacFn_(jacFn)
        , jacColPtrs_(std::move(jacColPtrs))
        , target_(target.subspan(range.Start(), range.Size()))
        , range_(range)
        , weights_(weights.empty() ? weights : weights.subspan(range.Start(), range.Size()))
        , nRowsPad_(static_cast<std::size_t>((static_cast<int>(range.Size()) + 7) & ~7))
        , scratchResiduals_(nRowsPad_)
        , scratchJac_(nRowsPad_ * this->numParameters_)
        , nVars_(nVars)
        , nConsts_(nConsts)
    {
        ValidateLMWeights(weights_, this->numResiduals_);
    }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == this->numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, this->numParameters_ };

        auto const nRowsPad = static_cast<int32_t>(nRowsPad_);

        if (jacobian != nullptr) {
            ++this->jacobianCallCount_;
            if (jacFn_ != nullptr) {
                ENSURE(nVars_   < 0 || static_cast<int>(jacColPtrs_.size()) == nVars_);
                ENSURE(nConsts_ < 0 || static_cast<int>(this->numParameters_) == nConsts_);
                // Write into padded per-column scratch, then copy valid rows to jacobian.
                std::vector<float*> outs(this->numParameters_);
                for (std::size_t k = 0; k < this->numParameters_; ++k) {
                    outs[k] = scratchJac_.data() + k * nRowsPad_;
                }
                jacFn_(outs.data(), jacColPtrs_.data(), nRowsPad, parameters);
                for (std::size_t k = 0; k < this->numParameters_; ++k) {
                    std::copy_n(scratchJac_.data() + k * nRowsPad_, this->numResiduals_,
                                jacobian + k * static_cast<std::ptrdiff_t>(this->numResiduals_));
                }
            } else {
                Operon::Span<Operon::Scalar> jac{jacobian, this->numResiduals_ * this->numParameters_};
                interpreter_->JacRev(params, range_, jac);
            }
            ApplyLMJacobianWeights(weights_, jacobian, this->numResiduals_, this->numParameters_);
        }

        if (residuals != nullptr) {
            ++this->residualCallCount_;
            Operon::Span<Operon::Scalar> res{residuals, this->numResiduals_};
            if (fn_ != nullptr) {
                ENSURE(nVars_   < 0 || static_cast<int>(colPtrs_.size()) == nVars_);
                ENSURE(nConsts_ < 0 || static_cast<int>(this->numParameters_) == nConsts_);
                fn_(scratchResiduals_.data(), colPtrs_.data(), nRowsPad, parameters);
                std::copy_n(scratchResiduals_.data(), this->numResiduals_, residuals);
            } else {
                Operon::Span<Operon::Scalar const> params{parameters, this->numParameters_};
                interpreter_->Evaluate(params, range_, res);
            }
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, static_cast<Eigen::Index>(this->numResiduals_));
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), static_cast<Eigen::Index>(this->numResiduals_));
            x -= y;
            ApplyLMResidualWeights(weights_, residuals, this->numResiduals_);
        }
        return true;
    }

private:
    gsl::not_null<InterpreterBase<T> const*> interpreter_;
    JIT::EvalFn                              fn_;
    std::vector<float const*>                colPtrs_;
    JIT::EvalJacFn                           jacFn_ = nullptr;
    std::vector<float const*>                jacColPtrs_;
    Operon::Span<Operon::Scalar const>       target_;
    Operon::Range const                      range_;   // NOLINT
    Operon::Span<Operon::Scalar const>       weights_;
    std::size_t                              nRowsPad_;
    mutable std::vector<Scalar>              scratchResiduals_;
    mutable std::vector<Scalar>              scratchJac_;

    int                                      nVars_   = -1;
    int                                      nConsts_ = -1;
};

} // namespace Operon

#endif // HAVE_ASMJIT
