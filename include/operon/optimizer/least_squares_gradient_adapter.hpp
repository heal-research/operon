// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_GRADIENT_ADAPTER_HPP
#define OPERON_LEAST_SQUARES_GRADIENT_ADAPTER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <gsl/pointers>

#include "operon/optimizer/gradient_cost.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

namespace detail {
    [[nodiscard]] inline auto ToGradientError(LeastSquaresError const& error) -> GradientError
    {
        auto code = GradientErrorCode::EvaluationFailure;
        switch (error.Code) {
        case LeastSquaresErrorCode::InvalidShape:
        case LeastSquaresErrorCode::InvalidView:
            code = GradientErrorCode::InvalidShape;
            break;
        case LeastSquaresErrorCode::NonFiniteEvaluation:
            code = GradientErrorCode::NonFiniteEvaluation;
            break;
        case LeastSquaresErrorCode::NumericalFailure:
            code = GradientErrorCode::NumericalFailure;
            break;
        case LeastSquaresErrorCode::EvaluationFailure:
            code = GradientErrorCode::EvaluationFailure;
            break;
        }
        return GradientError { .Code = code, .Expected = error.Expected, .Actual = error.Actual, .Cause = error.Cause };
    }
} // namespace detail

/**
 * Reduces a LeastSquaresCostFunction's residuals/Jacobian to the objective
 * plus gradient a GradientCostFunction consumer needs: Cost = 0.5 *
 * sum(w_i * r_i^2), gradient = J^T (w .* r). weights are numerical WLS
 * weights, not statistical sigma. The adapter owns only scratch buffers;
 * the wrapped cost is borrowed. First error wins; on failure the gradient
 * is filled with NaN.
 */
class LeastSquaresGradientAdapter final : public GradientCostFunction {
public:
    using Scalar = Operon::Scalar;

    explicit LeastSquaresGradientAdapter(gsl::not_null<LeastSquaresCostFunction const*> cost, ConstScalarSpan weights = {})
        : cost_(cost)
        , weights_(weights)
        , residualScratch_(cost->NumResiduals())
        , jacobianScratch_(cost->NumResiduals() * cost->NumParameters())
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return cost_->NumParameters(); }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError> override
    {
        auto const n = cost_->NumResiduals();
        auto const p = cost_->NumParameters();
        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        ScalarMatrixView jacobianView { jacobianScratch_.data(), Mapping { Extents { n, p }, std::array<MemoryIndex, 2> { p, 1 } } };

        auto evalResult = cost_->Evaluate(parameters, residualScratch_, jacobianView);
        if (!evalResult) {
            return Fail(evalResult.error(), gradient);
        }

        auto gradResult = ComputeGradient(residualScratch_, jacobianView, gradient, weights_);
        if (!gradResult) {
            return Fail(gradResult.error(), gradient);
        }

        auto const cost = static_cast<Scalar>(*gradResult);
        if (!std::isfinite(static_cast<double>(cost))) {
            return Fail(LeastSquaresError { .Code = LeastSquaresErrorCode::NumericalFailure }, gradient);
        }
        return cost;
    }

    [[nodiscard]] auto Error() const -> std::optional<GradientError> const& { return error_; }

private:
    auto Fail(LeastSquaresError const& error, ScalarSpan gradient) const -> tl::expected<Scalar, GradientError>
    {
        auto gradientError = detail::ToGradientError(error);
        if (!error_) {
            error_ = gradientError;
        }
        std::fill(gradient.begin(), gradient.end(), std::numeric_limits<Scalar>::quiet_NaN());
        return tl::unexpected(gradientError);
    }

    gsl::not_null<LeastSquaresCostFunction const*> cost_;
    ConstScalarSpan weights_;
    mutable std::vector<Scalar> residualScratch_;
    mutable std::vector<Scalar> jacobianScratch_;
    mutable std::optional<GradientError> error_;
};

} // namespace Operon

#endif
