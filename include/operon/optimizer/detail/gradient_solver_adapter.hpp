// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_GRADIENT_SOLVER_ADAPTER_HPP
#define OPERON_GRADIENT_SOLVER_ADAPTER_HPP

#include <cstddef>
#include <limits>
#include <optional>

#include <gsl/pointers>

#include "operon/optimizer/gradient_cost.hpp"

namespace Operon::detail {

/**
 * Private Eigen bridge: adapts any Concepts::GradientCost to the raw
 * (parameters, gradient) -> Scalar functor shape both lbfgs::solver
 * (Eigen::Matrix vectors) and SGDSolver (Eigen::Array vectors) require.
 * Not part of the canonical interface -- GradientCostFunction and
 * LeastSquaresGradientAdapter never see an Eigen type.
 */
template <typename Cost>
requires Concepts::GradientCost<Cost>
class GradientSolverAdapter {
public:
    using Scalar = Operon::Scalar;
    using scalar_t = Scalar; // NOLINT(readability-identifier-naming) -- required spelling for lbfgs::solver

    explicit GradientSolverAdapter(gsl::not_null<Cost const*> cost)
        : cost_(cost)
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t { return cost_->NumParameters(); }

    template <typename ParameterVector, typename GradientVector>
    auto operator()(ParameterVector const& parameters, GradientVector& gradient) const noexcept -> Scalar
    {
        Operon::ConstScalarSpan paramSpan { parameters.data(), static_cast<std::size_t>(parameters.size()) };
        Operon::ScalarSpan gradSpan { gradient.data(), static_cast<std::size_t>(gradient.size()) };
        auto result = cost_->Evaluate(paramSpan, gradSpan);
        if (!result) {
            if (!error_) {
                error_ = result.error();
            }
            return std::numeric_limits<Scalar>::quiet_NaN();
        }
        return *result;
    }

    [[nodiscard]] auto Error() const -> std::optional<GradientError> const& { return error_; }

private:
    gsl::not_null<Cost const*> cost_;
    mutable std::optional<GradientError> error_;
};

} // namespace Operon::detail

#endif
