// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_GAUSSIAN_GRADIENT_COST_HPP
#define OPERON_GAUSSIAN_GRADIENT_COST_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include <gsl/pointers>

#include "operon/core/contracts.hpp"
#include "operon/core/range.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/gradient_cost.hpp"
#include "operon/optimizer/interpreter_least_squares.hpp"
#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/least_squares_gradient_adapter.hpp"
#include "operon/random/random.hpp"

namespace Operon {

/**
 * Gaussian gradient cost: 0.5*sum(w_i*(prediction_i-target_i)^2) and its exact
 * reverse-Jacobian gradient, computed via ComputeGradient over the canonical
 * InterpreterLeastSquaresCostFunction residual/Jacobian of the selected batch.
 * target and weights are whole-dataset-column spans
 * (absolute row-indexed), the same coordinates as range, since a minibatch is
 * a random subrange of range indexed the same way. Empty weights mean one; a
 * scalar weight broadcasts; per-row weights are numerical WLS weights, never
 * statistical sigma. batchSize==0 is full-range; a nonzero batch size
 * requires a non-null rng and selects a new random subrange of range on every
 * call. Has no likelihood, Fisher, sigma, or Eigen-facing method.
 */
template <typename T = Scalar>
class GaussianGradientCostFunction final : public GradientCostFunction {
public:
    using Scalar = T;

    GaussianGradientCostFunction(
        gsl::not_null<InterpreterBase<T> const*> interpreter,
        ConstScalarSpan target,
        Range range,
        RandomGenerator* rng = nullptr,
        std::size_t batchSize = 0,
        ConstScalarSpan weights = {})
        : interpreter_(interpreter)
        , target_(target)
        , range_(range)
        , rng_(rng)
        , batchSize_(batchSize == 0 ? range.Size() : std::min(batchSize, range.Size()))
        , weights_(weights)
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
    {
        EXPECT(range_.Start() + range_.Size() <= target_.size());
        EXPECT(batchSize == 0 || rng_ != nullptr);
        EXPECT(weights_.empty() || weights_.size() == 1 || weights_.size() == target_.size());
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return numParameters_; }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError> override
    {
        ++feval_;
        auto const batch = SelectBatch();
        auto const n = batch.Size();
        auto const weightSlice = (weights_.empty() || weights_.size() == 1) ? weights_ : weights_.subspan(batch.Start(), n);
        if (!detail::AllFinite(weightSlice, /*requireNonnegative=*/true)) {
            return Fail(GradientError { .Code = GradientErrorCode::NonFiniteEvaluation }, gradient);
        }
        residualScratch_.resize(n);
        jacobianScratch_.resize(n * numParameters_);

        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        ScalarMatrixView jacobianView { jacobianScratch_.data(), Mapping { Extents { n, numParameters_ }, std::array<MemoryIndex, 2> { 1, n } } };

        // The batch is an absolute subrange of the whole-dataset target, so
        // the canonical interpreter cost for exactly this batch yields the
        // raw residual (prediction - target) and Jacobian.
        InterpreterLeastSquaresCostFunction const residualCost { interpreter_, target_, batch };
        auto evaluated = residualCost.Evaluate(parameters, residualScratch_, jacobianView);
        if (!evaluated) {
            return Fail(detail::ToGradientError(evaluated.error()), gradient);
        }
        ++jeval_;
        auto gradResult = ComputeGradient(residualScratch_, jacobianView, gradient, weightSlice);
        if (!gradResult) {
            return Fail(detail::ToGradientError(gradResult.error()), gradient);
        }
        auto const cost = static_cast<Scalar>(*gradResult);
        if (!std::isfinite(static_cast<double>(cost))) {
            return Fail(GradientError { .Code = GradientErrorCode::NonFiniteEvaluation }, gradient);
        }
        return cost;
    }

    [[nodiscard]] auto Error() const -> std::optional<GradientError> const& { return error_; }
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return feval_; }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return jeval_; }

private:
    [[nodiscard]] auto SelectBatch() const -> Range
    {
        if (batchSize_ >= range_.Size()) {
            return range_;
        }
        auto s = std::uniform_int_distribution<std::size_t> { 0UL, range_.Size() - batchSize_ }(*rng_);
        return Range { range_.Start() + s, range_.Start() + s + batchSize_ };
    }

    auto Fail(GradientError error, ScalarSpan gradient) const -> tl::expected<Scalar, GradientError>
    {
        if (!error_) {
            error_ = error;
        }
        std::fill(gradient.begin(), gradient.end(), std::numeric_limits<Scalar>::quiet_NaN());
        return tl::unexpected(error);
    }

    gsl::not_null<InterpreterBase<T> const*> interpreter_;
    ConstScalarSpan target_;
    Range range_; // NOLINT(readability-identifier-naming)
    RandomGenerator* rng_;
    std::size_t batchSize_;
    ConstScalarSpan weights_;
    std::size_t numParameters_;
    mutable std::vector<Scalar> residualScratch_;
    mutable std::vector<Scalar> jacobianScratch_;
    mutable std::size_t feval_ {};
    mutable std::size_t jeval_ {};
    mutable std::optional<GradientError> error_;
};

} // namespace Operon

#endif
