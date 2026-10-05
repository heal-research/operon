// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_LEAST_SQUARES_HPP
#define OPERON_INTERPRETER_LEAST_SQUARES_HPP

#include <cstddef>
#include <vector>

#include <gsl/pointers>

#include "operon/core/range.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

/**
 * Backend-neutral LeastSquaresCostFunction over an InterpreterBase: raw
 * residuals = prediction - target, raw Jacobian is the interpreter's
 * reverse-mode derivative copied into the caller's (possibly non-column-
 * major) view. Applies no statistical sigma and no numerical weight -- a
 * caller that wants weighting wraps this in LeastSquaresLMAdapter or
 * LeastSquaresGradientAdapter. Interpreter failures are reported as
 * LeastSquaresErrorCode::EvaluationFailure with the original
 * InterpreterError preserved as Cause.
 */
class InterpreterLeastSquaresCostFunction final : public LeastSquaresCostFunction {
public:
    InterpreterLeastSquaresCostFunction(
        gsl::not_null<InterpreterBase<Scalar> const*> interpreter, ConstScalarSpan target, Range range)
        : interpreter_(interpreter)
        , target_(target.subspan(range.Start(), range.Size()))
        , range_(range)
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return numParameters_; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return range_.Size(); }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan residuals,
        std::optional<ScalarMatrixView> jacobian) const -> tl::expected<void, LeastSquaresError> override
    {
        auto const n = range_.Size();
        if (parameters.size() != numParameters_) {
            return tl::unexpected(LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = numParameters_, .Actual = parameters.size() });
        }
        if (residuals.size() != n) {
            return tl::unexpected(LeastSquaresError {
                .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = residuals.size() });
        }
        if (jacobian && (jacobian->extent(0) != n || jacobian->extent(1) != numParameters_)) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape,
                .Expected = n,
                .Actual = jacobian->extent(0),
                .Row = n,
                .Column = numParameters_ });
        }

        auto predicted = interpreter_->Evaluate(parameters, range_, residuals);
        if (!predicted) {
            return tl::unexpected(
                LeastSquaresError { .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = predicted.error() });
        }
        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] -= target_[i];
        }

        if (jacobian) {
            auto const& view = *jacobian;
            // The interpreter writes column-major (stride {1, n}). A view with
            // exactly that layout is filled in place; any other stride pattern
            // goes through scratch and a logical (row, column) copy.
            if (n > 0 && view.stride(0) == 1 && view.stride(1) == n) {
                auto jacResult
                    = interpreter_->JacRev(parameters, range_, ScalarSpan { view.data_handle(), n * numParameters_ });
                if (!jacResult) {
                    return tl::unexpected(LeastSquaresError {
                        .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = jacResult.error() });
                }
                return {};
            }
            jacobianScratch_.resize(n * numParameters_);
            auto jacResult = interpreter_->JacRev(parameters, range_, jacobianScratch_);
            if (!jacResult) {
                return tl::unexpected(
                    LeastSquaresError { .Code = LeastSquaresErrorCode::EvaluationFailure, .Cause = jacResult.error() });
            }
            for (std::size_t j = 0; j < numParameters_; ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                    At(view, i, j) = jacobianScratch_[(j * n) + i];
                }
            }
        }

        return {};
    }

private:
    gsl::not_null<InterpreterBase<Scalar> const*> interpreter_;
    ConstScalarSpan target_;
    Range range_; // NOLINT(readability-identifier-naming)
    std::size_t numParameters_;
    mutable std::vector<Scalar> jacobianScratch_;
};

} // namespace Operon

#endif
