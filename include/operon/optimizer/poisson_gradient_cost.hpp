// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_POISSON_GRADIENT_COST_HPP
#define OPERON_POISSON_GRADIENT_COST_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gsl/pointers>

#include "operon/core/contracts.hpp"
#include "operon/core/range.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/gradient_cost.hpp"
#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/least_squares_gradient_adapter.hpp"
#include "operon/random/random.hpp"

namespace Operon {

namespace detail {
    // gradient = J^T * chain, fixed-order double accumulation with checked
    // narrowing; unlike ComputeGradient this has no quadratic 0.5*r^2 cost
    // term, since chain is already the exact per-row d(nll)/d(prediction).
    [[nodiscard]] inline auto AccumulateChainGradient(ConstScalarSpan chain, ConstScalarMatrixView jacobian, ScalarSpan gradient) -> bool
    {
        auto const n = chain.size();
        auto const p = jacobian.extent(1);
        std::vector<AccumulationScalar> accum(p, AccumulationScalar { 0 });
        for (std::size_t i = 0; i < n; ++i) {
            auto const c = static_cast<AccumulationScalar>(chain[i]);
            for (std::size_t j = 0; j < p; ++j) {
                accum[j] += c * static_cast<AccumulationScalar>(At(jacobian, i, j));
            }
        }
        for (std::size_t j = 0; j < p; ++j) {
            if (!std::isfinite(accum[j])) {
                return false;
            }
            auto const value = static_cast<Scalar>(accum[j]);
            if (!std::isfinite(static_cast<double>(value))) {
                return false;
            }
            gradient[j] = value;
        }
        return true;
    }
} // namespace detail

/**
 * Poisson gradient cost, matching PoissonLikelihood's NLL exactly (including
 * its coefficient-independent lgamma(y_i+1) term):
 *
 * LogInput=true:  eta_i = e_i*z_i, nll_i = exp(eta_i) - y_i*eta_i + lgamma(y_i+1),
 *                 d(nll_i)/d(z_i) = e_i*(exp(eta_i) - y_i).
 * LogInput=false: mu_i = e_i*z_i,  nll_i = mu_i - y_i*log(mu_i) + lgamma(y_i+1),
 *                 d(nll_i)/d(z_i) = e_i*(1 - y_i/mu_i); mu_i must be finite and
 *                 strictly positive whenever the log is evaluated, otherwise a
 *                 typed NonFiniteEvaluation error is returned rather than a
 *                 NaN silently accepted by the solver.
 *
 * z_i is the interpreter prediction; the gradient is this scalar chained
 * through the interpreter's reverse-Jacobian row J_i. exposure is explicit
 * Poisson exposure (not a Gaussian precision): empty means one, one value
 * broadcasts, one value per observation is a whole-dataset-column span in
 * the same absolute row coordinates as target/range. Exposure shape and
 * domain (finite, nonnegative) are user data: a violation is a typed
 * GradientErrorCode::InvalidWeights from Evaluate, never an assertion. Ordinary
 * dataset sample weights are deliberately not applied unless a caller passes
 * them explicitly as exposure; UsesDatasetWeights is therefore false, and the
 * optimizers always construct this cost with empty exposure. batchSize==0 is
 * full-range; a nonzero batch size requires a non-null rng and selects a new
 * random subrange of range on every call. The scalar type is Operon::Scalar.
 */
template <bool LogInput = true>
class PoissonGradientCostFunction final : public GradientCostFunction {
public:
    using Scalar = Operon::Scalar;
    static constexpr bool UsesDatasetWeights { false };

    PoissonGradientCostFunction(
        gsl::not_null<InterpreterBase<Scalar> const*> interpreter,
        ConstScalarSpan target,
        Range range,
        RandomGenerator* rng = nullptr,
        std::size_t batchSize = 0,
        ConstScalarSpan exposure = {})
        : interpreter_(interpreter)
        , target_(target)
        , range_(range)
        , rng_(rng)
        , batchSize_(batchSize == 0 ? range.Size() : std::min(batchSize, range.Size()))
        , exposure_(exposure)
        , numParameters_(static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount()))
    {
        EXPECT(range_.Start() + range_.Size() <= target_.size());
        EXPECT(batchSize == 0 || rng_ != nullptr);
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return numParameters_; }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError> override
    {
        ++feval_;
        auto const batch = SelectBatch();
        auto const n = batch.Size();
        auto validExposure = detail::ValidatedBatchWeights(exposure_, target_.size(), batch.Start(), n);
        if (!validExposure) {
            return Fail(std::move(validExposure.error()), gradient);
        }
        predictionScratch_.resize(n);
        auto predicted = interpreter_->Evaluate(parameters, batch, predictionScratch_);
        if (!predicted) {
            return Fail(GradientError { .Code = GradientErrorCode::EvaluationFailure, .Cause = predicted.error() }, gradient);
        }
        auto const targetSlice = target_.subspan(batch.Start(), n);
        auto const exposureAt = [&](std::size_t i) -> Scalar {
            if (exposure_.empty()) {
                return Scalar { 1 };
            }
            return exposure_.size() == 1 ? exposure_[0] : exposure_[batch.Start() + i];
        };

        chainScratch_.resize(n);
        AccumulationScalar nll { 0 };
        for (std::size_t i = 0; i < n; ++i) {
            auto const z = predictionScratch_[i];
            auto const y = targetSlice[i];
            auto const e = exposureAt(i);
            auto const lgammaTerm = std::lgamma(static_cast<double>(y) + 1.0);
            if constexpr (LogInput) {
                auto const eta = e * z;
                auto const expEta = std::exp(static_cast<double>(eta));
                nll += expEta - (static_cast<double>(y) * static_cast<double>(eta)) + lgammaTerm;
                chainScratch_[i] = e * (static_cast<Scalar>(expEta) - y);
            } else {
                auto const mu = e * z;
                if (!std::isfinite(static_cast<double>(mu)) || mu <= Scalar { 0 }) {
                    return Fail(GradientError { .Code = GradientErrorCode::NonFiniteEvaluation }, gradient);
                }
                nll += static_cast<double>(mu) - (static_cast<double>(y) * std::log(static_cast<double>(mu))) + lgammaTerm;
                chainScratch_[i] = e * (Scalar { 1 } - (y / mu));
            }
        }
        if (!std::isfinite(nll)) {
            return Fail(GradientError { .Code = GradientErrorCode::NonFiniteEvaluation }, gradient);
        }

        ++jeval_;
        jacobianScratch_.resize(n * numParameters_);
        auto jacResult = interpreter_->JacRev(parameters, batch, jacobianScratch_);
        if (!jacResult) {
            return Fail(GradientError { .Code = GradientErrorCode::EvaluationFailure, .Cause = jacResult.error() }, gradient);
        }

        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        ScalarMatrixView jacobianView { jacobianScratch_.data(), Mapping { Extents { n, numParameters_ }, std::array<MemoryIndex, 2> { 1, n } } };

        if (!detail::AccumulateChainGradient(chainScratch_, jacobianView, gradient)) {
            return Fail(GradientError { .Code = GradientErrorCode::NumericalFailure }, gradient);
        }

        return static_cast<Scalar>(nll);
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

    gsl::not_null<InterpreterBase<Scalar> const*> interpreter_;
    ConstScalarSpan target_;
    Range range_; // NOLINT(readability-identifier-naming)
    RandomGenerator* rng_;
    std::size_t batchSize_;
    ConstScalarSpan exposure_;
    std::size_t numParameters_;
    mutable std::vector<Scalar> predictionScratch_;
    mutable std::vector<Scalar> chainScratch_;
    mutable std::vector<Scalar> jacobianScratch_;
    mutable std::size_t feval_ {};
    mutable std::size_t jeval_ {};
    mutable std::optional<GradientError> error_;
};

} // namespace Operon

#endif
