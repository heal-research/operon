// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_HPP
#define OPERON_LEAST_SQUARES_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <tl/expected.hpp>

#include "operon/core/interpreter_error.hpp"
#include "operon/core/memory_view.hpp"

namespace Operon {

enum class LeastSquaresErrorCode : std::uint8_t {
    InvalidShape,
    InvalidView,
    NonFiniteEvaluation,
    NumericalFailure,
};

struct LeastSquaresError {
    LeastSquaresErrorCode Code {LeastSquaresErrorCode::InvalidShape};
    std::size_t Expected {};
    std::size_t Actual {};
    std::size_t Row {};
    std::size_t Column {};
    std::optional<InterpreterError> Cause {};
};

/**
 * Backend-neutral least-squares cost contract.
 *
 * Implementations borrow the model/data and caller-owned output buffers.
 * Parameters and residuals must have exact sizes; a supplied Jacobian must
 * have shape (NumResiduals(), NumParameters()) and arbitrary valid strides.
 * An absent Jacobian requests residual-only evaluation. Outputs are
 * indeterminate after an error unless an implementation documents transactional
 * behavior. Implementations must not expose or require a solver-specific
 * matrix type.
 */
class LeastSquaresCostFunction {
public:
    LeastSquaresCostFunction() = default;
    LeastSquaresCostFunction(LeastSquaresCostFunction const&) = delete;
    auto operator=(LeastSquaresCostFunction const&) -> LeastSquaresCostFunction& = delete;
    LeastSquaresCostFunction(LeastSquaresCostFunction&&) = delete;
    auto operator=(LeastSquaresCostFunction&&) -> LeastSquaresCostFunction& = delete;
    virtual ~LeastSquaresCostFunction() = default;
    [[nodiscard]] virtual auto NumParameters() const noexcept -> std::size_t = 0;
    [[nodiscard]] virtual auto NumResiduals() const noexcept -> std::size_t = 0;
    [[nodiscard]] virtual auto Evaluate(
        std::span<Scalar const> parameters,
        std::span<Scalar> residuals,
        std::optional<ScalarMatrixView> jacobian)
        const -> tl::expected<void, LeastSquaresError> = 0;
};

namespace detail {
    // Finite, and nonnegative when requireNonnegative is set.
    [[nodiscard]] inline auto AllFinite(ConstScalarSpan values, bool requireNonnegative) -> bool
    {
        return std::ranges::all_of(values, [requireNonnegative](Scalar v) -> bool {
            return std::isfinite(static_cast<double>(v)) && (!requireNonnegative || v >= Scalar { 0 });
        });
    }
} // namespace detail

/** Diagnostics for a residual vector and optional Jacobian; GradientNorm is NaN without a Jacobian. */
struct LeastSquaresDiagnostics {
    double Cost {};
    double ResidualNorm {};
    double GradientNorm { std::numeric_limits<double>::quiet_NaN() };
};

/** Cost = 0.5 * sum(w_i * r_i^2); GradientNorm = ||J^T (w .* r)||_2 when jacobian is supplied. */
[[nodiscard]] inline auto ComputeDiagnostics(
    ConstScalarSpan residuals,
    std::optional<ConstScalarMatrixView> jacobian,
    ConstScalarSpan weights = {})
    -> tl::expected<LeastSquaresDiagnostics, LeastSquaresError>
{
    auto const n = residuals.size();
    if (!weights.empty() && weights.size() != 1 && weights.size() != n) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = weights.size() });
    }
    if (jacobian && jacobian->extent(0) != n) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = jacobian->extent(0) });
    }
    if (!detail::AllFinite(residuals, false) || !detail::AllFinite(weights, true)) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }

    auto const weightAt = [&weights](std::size_t i) -> AccumulationScalar {
        if (weights.empty()) { return AccumulationScalar { 1 }; }
        return static_cast<AccumulationScalar>(weights.size() == 1 ? weights[0] : weights[i]);
    };

    AccumulationScalar cost {0};
    AccumulationScalar sumSquares {0};
    for (std::size_t i = 0; i < n; ++i) {
        auto const r = static_cast<AccumulationScalar>(residuals[i]);
        cost += 0.5 * weightAt(i) * r * r;
        sumSquares += r * r;
    }

    LeastSquaresDiagnostics diagnostics;
    diagnostics.Cost = cost;
    diagnostics.ResidualNorm = std::sqrt(sumSquares);

    if (jacobian) {
        auto const p = jacobian->extent(1);
        std::vector<AccumulationScalar> gradient(p, AccumulationScalar { 0 });
        for (std::size_t i = 0; i < n; ++i) {
            auto const wr = weightAt(i) * static_cast<AccumulationScalar>(residuals[i]);
            for (std::size_t j = 0; j < p; ++j) {
                gradient[j] += wr * static_cast<AccumulationScalar>(At(*jacobian, i, j));
            }
        }
        AccumulationScalar gradientSumSquares {0};
        for (auto const g : gradient) { gradientSumSquares += g * g; }
        diagnostics.GradientNorm = std::sqrt(gradientSumSquares);
    }

    if (!std::isfinite(diagnostics.Cost) || !std::isfinite(diagnostics.ResidualNorm)
        || (jacobian && !std::isfinite(diagnostics.GradientNorm))) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }

    return diagnostics;
}

} // namespace Operon

#endif
