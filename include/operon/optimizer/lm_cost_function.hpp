// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_COST_FUNCTION_HPP
#define OPERON_LM_COST_FUNCTION_HPP

#include <gsl/pointers>
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/lm_cost_function_base.hpp"

namespace Operon {

template<typename T = Operon::Scalar, int StorageOrder = Eigen::ColMajor>
struct LMCostFunction : public LMCostFunctionBase<LMCostFunction<T, StorageOrder>, StorageOrder> {
    using Base = LMCostFunctionBase<LMCostFunction<T, StorageOrder>, StorageOrder>;
    using Scalar = typename Base::Scalar;

    explicit LMCostFunction(gsl::not_null<InterpreterBase<T> const*> interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, Operon::Span<Operon::Scalar const> weights = {})
        : Base{range.Size(), static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount())}
        , interpreter_(interpreter)
        , target_{target}
        , range_{range}
        , weights_{weights}
    {
        // Validated once here rather than per-Evaluate() call: EXPECT (libassert)
        // is always-on, not debug-only, so an O(N) scan per residual evaluation
        // would be a real per-LM-iteration cost across a GP run.
        ValidateLMWeights(weights_, this->numResiduals_);
    }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == this->numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, this->numParameters_ };

        if (jacobian != nullptr) {
            ++this->jacobianCallCount_;
            Operon::Span<Operon::Scalar> jac{jacobian, this->numResiduals_ * this->numParameters_};
            interpreter_->JacRev(params, range_, jac);
            ApplyLMJacobianWeights(weights_, jacobian, this->numResiduals_, this->numParameters_);
        }

        if (residuals != nullptr) {
            ++this->residualCallCount_;
            Operon::Span<Operon::Scalar> res{ residuals, this->numResiduals_ };
            interpreter_->Evaluate(params, range_, res);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, static_cast<Eigen::Index>(this->numResiduals_));
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), static_cast<Eigen::Index>(this->numResiduals_));
            x -= y;
            ApplyLMResidualWeights(weights_, residuals, this->numResiduals_);
        }
        return true;
    }

private:
    gsl::not_null<InterpreterBase<T> const*> interpreter_;
    Operon::Span<Operon::Scalar const> target_;
    Operon::Range const range_; // NOLINT
    Operon::Span<Operon::Scalar const> weights_;
};
} // namespace Operon

#endif
