// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_HPP
#define OPERON_LEAST_SQUARES_HPP

#include <cstddef>
#include <optional>
#include <span>

#include <tl/expected.hpp>

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
    [[nodiscard]] virtual auto Evaluate(
        std::span<Scalar const> parameters,
        std::span<Scalar> residuals,
        std::optional<ScalarMatrixView> jacobian)
        const -> tl::expected<void, LeastSquaresError> = 0;
};

} // namespace Operon

#endif
