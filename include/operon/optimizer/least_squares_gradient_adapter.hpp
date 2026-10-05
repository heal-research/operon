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
#include <utility>
#include <vector>

#include <gsl/pointers>

#include "operon/optimizer/gradient_cost.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

/**
 * Total, location-preserving conversion: every LeastSquaresErrorCode maps to
 * the GradientErrorCode of the same name, and Expected/Actual/Row/Column/Cause
 * are carried over unchanged.
 */
[[nodiscard]] inline auto ToGradientError(LeastSquaresError const& error) -> GradientError
{
    auto const code = [&error]() -> GradientErrorCode {
        switch (error.Code) {
        case LeastSquaresErrorCode::InvalidShape:
            return GradientErrorCode::InvalidShape;
        case LeastSquaresErrorCode::InvalidView:
            return GradientErrorCode::InvalidView;
        case LeastSquaresErrorCode::InvalidWeights:
            return GradientErrorCode::InvalidWeights;
        case LeastSquaresErrorCode::NonFiniteEvaluation:
            return GradientErrorCode::NonFiniteEvaluation;
        case LeastSquaresErrorCode::NumericalFailure:
            return GradientErrorCode::NumericalFailure;
        case LeastSquaresErrorCode::EvaluationFailure:
            return GradientErrorCode::EvaluationFailure;
        }
        // Unreachable for a valid enumerator; an out-of-range value is a
        // generic evaluation failure rather than undefined behavior.
        return GradientErrorCode::EvaluationFailure;
    }();
    return GradientError { .Code = code,
        .Expected = error.Expected,
        .Actual = error.Actual,
        .Row = error.Row,
        .Column = error.Column,
        .Cause = error.Cause };
}

/** Weight violations surface as GradientErrorCode::InvalidWeights with Expected/Actual/Row preserved. */
[[nodiscard]] inline auto ToGradientError(WeightError const& error) -> GradientError
{
    return ToGradientError(ToLeastSquaresError(error));
}

namespace detail {
    /**
     * Validates a whole-dataset-column weight/exposure span for the batch
     * [start, start + count) and returns the span a cost should use for it:
     * the column itself when it is empty or a broadcast scalar, otherwise the
     * batch-local slice. A per-row column must have exactly columnRows entries;
     * only the batch slice is domain-checked (rows outside the training range
     * may legitimately hold placeholder values). Violations are
     * GradientErrorCode::InvalidWeights; for a per-row column Row is the
     * absolute column row of the first offending entry.
     */
    [[nodiscard]] inline auto ValidatedBatchWeights(ConstScalarSpan column, std::size_t columnRows, std::size_t start,
        std::size_t count) -> tl::expected<ConstScalarSpan, GradientError>
    {
        if (column.empty() || column.size() == 1) {
            if (auto valid = ValidateWeights(column, count); !valid) {
                return tl::unexpected(ToGradientError(valid.error()));
            }
            return column;
        }
        if (column.size() != columnRows) {
            return tl::unexpected(ToGradientError(WeightError {
                .Code = WeightErrorCode::SizeMismatch, .Expected = columnRows, .Actual = column.size() }));
        }
        auto const slice = column.subspan(start, count);
        if (auto valid = ValidateWeights(slice, count); !valid) {
            auto error = ToGradientError(valid.error());
            error.Row += start;
            return tl::unexpected(std::move(error));
        }
        return slice;
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

    explicit LeastSquaresGradientAdapter(
        gsl::not_null<LeastSquaresCostFunction const*> cost, ConstScalarSpan weights = {})
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
        ScalarMatrixView jacobianView { jacobianScratch_.data(),
            Mapping { Extents { n, p }, std::array<MemoryIndex, 2> { p, 1 } } };

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
        auto gradientError = ToGradientError(error);
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
