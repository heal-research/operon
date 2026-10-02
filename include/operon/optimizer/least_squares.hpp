// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_HPP
#define OPERON_LEAST_SQUARES_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
    InvalidWeights,
    NonFiniteEvaluation,
    NumericalFailure,
    EvaluationFailure,
};

struct LeastSquaresError {
    LeastSquaresErrorCode Code {LeastSquaresErrorCode::InvalidShape};
    std::size_t Expected {};
    std::size_t Actual {};
    std::size_t Row {};
    std::size_t Column {};
    std::optional<InterpreterError> Cause {};
};

enum class WeightErrorCode : std::uint8_t {
    SizeMismatch,
    NegativeValue,
    NotANumber,
    Infinite,
};

/**
 * Typed weight/exposure validation failure. Expected is the accepted
 * per-row size and Actual the supplied size (both set for every code);
 * Row is the index of the first offending entry within the validated span
 * (0 for a size mismatch or a broadcast scalar). The coordinate frame is
 * therefore that of whichever span was validated: the caller's weight span
 * for FitLeastSquares and LeastSquaresLMAdapter, the training-range slice
 * (index 0 = first training row) for the optimizers' FitConfigurationError,
 * and the absolute whole-dataset-column row for a gradient cost's
 * GradientErrorCode::InvalidWeights.
 */
struct WeightError {
    WeightErrorCode Code {WeightErrorCode::SizeMismatch};
    std::size_t Expected {};
    std::size_t Actual {};
    std::size_t Row {};
};

/**
 * The single accepted weight shape convention: empty (all ones), one value
 * (broadcast), or exactly `rows` values (per-row). Every value must be finite
 * and nonnegative. The first offending entry is reported; non-finite values
 * are classified before the sign check, so -inf is Infinite, not NegativeValue.
 * Eigen-free and allocation-free.
 */
[[nodiscard]] inline auto ValidateWeights(ConstScalarSpan weights, std::size_t rows) -> tl::expected<void, WeightError>
{
    if (!weights.empty() && weights.size() != 1 && weights.size() != rows) {
        return tl::unexpected(WeightError { .Code = WeightErrorCode::SizeMismatch, .Expected = rows, .Actual = weights.size() });
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        auto const w = static_cast<double>(weights[i]);
        auto code = WeightErrorCode::NegativeValue;
        if (std::isnan(w)) {
            code = WeightErrorCode::NotANumber;
        } else if (std::isinf(w)) {
            code = WeightErrorCode::Infinite;
        } else if (w >= 0.0) {
            continue;
        }
        return tl::unexpected(WeightError { .Code = code, .Expected = rows, .Actual = weights.size(), .Row = i });
    }
    return {};
}

/** Wraps a WeightError as LeastSquaresErrorCode::InvalidWeights, preserving size and Row. */
[[nodiscard]] inline auto ToLeastSquaresError(WeightError const& error) -> LeastSquaresError
{
    return LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidWeights, .Expected = error.Expected, .Actual = error.Actual, .Row = error.Row };
}

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

namespace Concepts {
    // Structural: NumParameters()/NumResiduals()/Evaluate() only. No
    // likelihood, Fisher, or statistical requirement.
    template <typename T>
    concept LeastSquaresCost = requires(
        T const& cost,
        ConstScalarSpan parameters,
        ScalarSpan residuals,
        std::optional<ScalarMatrixView> jacobian) {
        { cost.NumParameters() } -> std::same_as<std::size_t>;
        { cost.NumResiduals() } -> std::same_as<std::size_t>;
        { cost.Evaluate(parameters, residuals, jacobian) } -> std::same_as<tl::expected<void, LeastSquaresError>>;
    };
} // namespace Concepts

namespace detail {
    [[nodiscard]] inline auto AllFinite(ConstScalarSpan values) -> bool
    {
        return std::ranges::all_of(values, [](Scalar v) -> bool {
            return std::isfinite(static_cast<double>(v));
        });
    }
} // namespace detail

/** Diagnostics for a residual vector and optional Jacobian; GradientNorm is NaN without a Jacobian. */
struct LeastSquaresDiagnostics {
    double Cost {};
    double ResidualNorm {};
    double GradientNorm { std::numeric_limits<double>::quiet_NaN() };
};

/**
 * Computes Cost = 0.5 * sum(w_i * r_i^2) and writes gradient = J^T (w .* r).
 * weights follow ValidateWeights (empty, size 1, or residuals.size();
 * finite and nonnegative). jacobian must have residuals.size() rows;
 * gradient must have jacobian's column count.
 * Returns InvalidShape for a Jacobian/gradient dimension mismatch,
 * InvalidWeights for a weight violation (Row = first offending entry),
 * NonFiniteEvaluation for a non-finite residual or accumulated result,
 * NumericalFailure if an accumulated component does not narrow to a finite
 * Scalar.
 */
[[nodiscard]] inline auto ComputeGradient(
    ConstScalarSpan residuals,
    ConstScalarMatrixView jacobian,
    ScalarSpan gradient,
    ConstScalarSpan weights = {})
    -> tl::expected<AccumulationScalar, LeastSquaresError>
{
    auto const n = residuals.size();
    auto const p = jacobian.extent(1);
    if (jacobian.extent(0) != n) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = jacobian.extent(0) });
    }
    if (gradient.size() != p) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = p, .Actual = gradient.size() });
    }
    if (auto validWeights = ValidateWeights(weights, n); !validWeights) {
        return tl::unexpected(ToLeastSquaresError(validWeights.error()));
    }
    if (!detail::AllFinite(residuals)) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }

    auto const weightAt = [&weights](std::size_t i) -> AccumulationScalar {
        if (weights.empty()) { return AccumulationScalar { 1 }; }
        return static_cast<AccumulationScalar>(weights.size() == 1 ? weights[0] : weights[i]);
    };

    std::vector<AccumulationScalar> accum(p, AccumulationScalar { 0 });
    AccumulationScalar cost {0};
    for (std::size_t i = 0; i < n; ++i) {
        auto const r = static_cast<AccumulationScalar>(residuals[i]);
        auto const w = weightAt(i);
        cost += 0.5 * w * r * r;
        auto const wr = w * r;
        for (std::size_t j = 0; j < p; ++j) {
            accum[j] += wr * static_cast<AccumulationScalar>(At(jacobian, i, j));
        }
    }

    if (!std::isfinite(cost)) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }
    for (std::size_t j = 0; j < p; ++j) {
        if (!std::isfinite(accum[j])) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation, .Column = j });
        }
        auto const value = static_cast<Scalar>(accum[j]);
        if (!std::isfinite(static_cast<double>(value))) {
            return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NumericalFailure, .Column = j });
        }
        gradient[j] = value;
    }

    return cost;
}

/** Cost = 0.5 * sum(w_i * r_i^2); GradientNorm = ||J^T (w .* r)||_2 when jacobian is supplied. */
[[nodiscard]] inline auto ComputeDiagnostics(
    ConstScalarSpan residuals,
    std::optional<ConstScalarMatrixView> jacobian,
    ConstScalarSpan weights = {})
    -> tl::expected<LeastSquaresDiagnostics, LeastSquaresError>
{
    auto const n = residuals.size();
    if (auto validWeights = ValidateWeights(weights, n); !validWeights) {
        return tl::unexpected(ToLeastSquaresError(validWeights.error()));
    }
    if (jacobian && jacobian->extent(0) != n) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape, .Expected = n, .Actual = jacobian->extent(0) });
    }
    if (!detail::AllFinite(residuals)) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }

    AccumulationScalar sumSquares {0};
    for (auto const r : residuals) {
        auto const rr = static_cast<AccumulationScalar>(r);
        sumSquares += rr * rr;
    }

    LeastSquaresDiagnostics diagnostics;
    diagnostics.ResidualNorm = std::sqrt(sumSquares);

    if (jacobian) {
        auto const p = jacobian->extent(1);
        std::vector<Scalar> gradient(p);
        auto result = ComputeGradient(residuals, *jacobian, gradient, weights);
        if (!result) {
            return tl::unexpected(result.error());
        }
        diagnostics.Cost = *result;
        AccumulationScalar gradientSumSquares {0};
        for (auto const g : gradient) {
            auto const gg = static_cast<AccumulationScalar>(g);
            gradientSumSquares += gg * gg;
        }
        diagnostics.GradientNorm = std::sqrt(gradientSumSquares);
    } else {
        auto const weightAt = [&weights](std::size_t i) -> AccumulationScalar {
            if (weights.empty()) { return AccumulationScalar { 1 }; }
            return static_cast<AccumulationScalar>(weights.size() == 1 ? weights[0] : weights[i]);
        };
        AccumulationScalar cost {0};
        for (std::size_t i = 0; i < n; ++i) {
            auto const r = static_cast<AccumulationScalar>(residuals[i]);
            cost += 0.5 * weightAt(i) * r * r;
        }
        diagnostics.Cost = cost;
    }

    if (!std::isfinite(diagnostics.Cost) || !std::isfinite(diagnostics.ResidualNorm)) {
        return tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
    }

    return diagnostics;
}

} // namespace Operon

#endif
