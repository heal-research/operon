// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_GRADIENT_COST_HPP
#define OPERON_GRADIENT_COST_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <tl/expected.hpp>

#include "operon/core/interpreter_error.hpp"
#include "operon/core/memory_view.hpp"

namespace Operon {

enum class GradientErrorCode : std::uint8_t {
    InvalidShape,
    InvalidView,
    InvalidWeights,
    NonFiniteEvaluation,
    NumericalFailure,
    EvaluationFailure,
};

// Row/Column locate the offending residual row, Jacobian entry, or weight
// when the producing cost knows them (zero otherwise); they are set by
// ToGradientError from LeastSquaresError and by the Gaussian/Poisson costs
// for weight violations. For GradientErrorCode::InvalidWeights, Row is the
// index of the first invalid weight in the frame documented on WeightError
// (absolute dataset-column row for a gradient cost's Evaluate).
struct GradientError {
    GradientErrorCode Code { GradientErrorCode::EvaluationFailure };
    std::size_t Expected {};
    std::size_t Actual {};
    std::size_t Row {};
    std::size_t Column {};
    std::optional<InterpreterError> Cause {};
};

/**
 * Backend-neutral objective-plus-gradient contract. Evaluate writes the
 * gradient of exactly the returned numerical objective; parameters and
 * gradient are exact-size borrowed contiguous spans. Outputs are
 * indeterminate after an error unless an implementation documents a NaN
 * fill. Implementations must not expose likelihood, Fisher information, or
 * any other statistical method -- those are a separate, explicit concern.
 */
class GradientCostFunction {
public:
    GradientCostFunction() = default;
    GradientCostFunction(GradientCostFunction const&) = delete;
    auto operator=(GradientCostFunction const&) -> GradientCostFunction& = delete;
    GradientCostFunction(GradientCostFunction&&) = delete;
    auto operator=(GradientCostFunction&&) -> GradientCostFunction& = delete;
    virtual ~GradientCostFunction() = default;

    [[nodiscard]] virtual auto NumParameters() const noexcept -> std::size_t = 0;

    [[nodiscard]] virtual auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError>
        = 0;
};

namespace Concepts {
    // Structural: NumParameters()/Evaluate() only, in Operon::Scalar. No
    // likelihood, Fisher, or interpreter requirement -- a purely numerical
    // cost with no statistical interpretation must satisfy this.
    template <typename T>
    concept GradientCost = requires(T const& cost, ConstScalarSpan parameters, ScalarSpan gradient) {
        typename T::Scalar;
        requires std::same_as<typename T::Scalar, Operon::Scalar>;
        { cost.NumParameters() } -> std::same_as<std::size_t>;
        { cost.Evaluate(parameters, gradient) } -> std::same_as<tl::expected<Operon::Scalar, GradientError>>;
    };
} // namespace Concepts

} // namespace Operon

#endif
