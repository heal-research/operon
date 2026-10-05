// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_LM_ADAPTER_HPP
#define OPERON_LEAST_SQUARES_LM_ADAPTER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <gsl/pointers>

#include "operon/core/contracts.hpp"
#include "operon/optimizer/detail/lm_backend_functor.hpp"
#include "operon/optimizer/least_squares.hpp"

namespace Operon {

/**
 * Adapts a LeastSquaresCostFunction to the raw-pointer Evaluate() interface
 * detail::LMBackendFunctor needs for Eigen::LevenbergMarquardt and
 * ceres::TinySolver. NumResiduals()/NumParameters() are derived from the
 * wrapped cost at construction; no duplicated count is accepted. weights
 * follow ValidateWeights: empty (unweighted), size 1 (uniform), or
 * NumResiduals() (per-row), finite and nonnegative. Weighting scales each
 * residual and Jacobian row by sqrt(weight_i), the standard WLS-via-LM
 * trick; weights are numerical WLS weights, never statistical sigma.
 *
 * Configuration error. The constructor validates the weights once, without
 * asserting or throwing. An invalid weight vector is stored as a typed
 * LeastSquaresErrorCode::InvalidWeights in Error() (Expected/Actual/Row as
 * produced by ValidateWeights) and marks the adapter permanently invalid.
 * Every later Evaluate() then returns false, poisons its outputs, and never
 * calls the wrapped cost or touches the call counters; the stored error is
 * not replaced.
 *
 * Error(). Holds the first error the adapter observed, if any: the
 * configuration error above, else the first error returned by the wrapped
 * cost or the first NonFiniteEvaluation detected here (Code/Row/Column/Cause
 * preserved). First error wins: later failing calls never overwrite it, and
 * nothing resets it. After a runtime (non-configuration) error the adapter
 * keeps evaluating on later calls; backends stop on the first false return.
 *
 * Non-finite outputs. When recoverNonFinite is false, a non-finite residual
 * or Jacobian entry from an otherwise successful cost evaluation is a typed
 * NonFiniteEvaluation error (first offending residual, else first offending
 * Jacobian entry, in row-major scan order). When true, such outputs are
 * weighted and returned as-is with Evaluate() == true and no error recorded,
 * so the solver can reject the trial step and increase damping; errors
 * returned by the cost itself are still recorded.
 *
 * Failure outputs. Whenever Evaluate() returns false, every caller-supplied
 * non-null residual and Jacobian output (all numResiduals * numParameters
 * entries) is filled with quiet NaN, regardless of what the cost wrote. A
 * null residual pointer (Jacobian-only call) is not poisoned: the adapter's
 * internal scratch is overwritten by the next call and is never exposed.
 *
 * Scratch. The residual scratch used by Jacobian-only calls is allocated
 * lazily on the first such call, never at construction, so an adapter whose
 * cost is invalid (or that is only used with residual pointers) allocates
 * nothing.
 *
 * Oversized problems. NumResiduals() above MaxBackendResiduals cannot be
 * represented by the backends' int ABI. ExceedsBackendLimit() reports this and
 * RecordOversized() stores why the problem cannot run (see its comment)
 * without allocating or narrowing; the fit driver calls it before any solver
 * or buffer. A Jacobian-only Evaluate() on such an adapter does the same and
 * returns false instead of allocating scratch. ResidualCount()/ParameterCount()
 * are the exact size_t counts; NumResiduals()/NumParameters() narrow to int and
 * are only meaningful within the limit.
 *
 * Counters. ResidualCalls()/JacobianCalls() count Evaluate() calls that
 * reached the wrapped cost with a non-null residual / Jacobian pointer
 * respectively, including calls that failed; a call with both pointers
 * counts in both. Calls rejected by an invalid configuration or as oversized
 * are not counted. The counters are plain integers (not atomics).
 *
 * Ownership and lifetime. The adapter borrows `cost` and `weights`: both
 * must outlive the adapter and must not be modified while it is in use (the
 * weight span is read on every Evaluate()). Parameter, residual, and
 * Jacobian pointers are borrowed for the duration of one Evaluate() call;
 * their sizes must be numParameters, numResiduals, and
 * numResiduals * numParameters (in StorageOrder). The adapter owns only its
 * scratch buffer, error, and counters. Error() returns a reference into the
 * adapter, valid for the adapter's lifetime. The adapter is neither copyable
 * nor movable.
 *
 * Thread safety and const. Evaluate() is const only because the backend
 * callbacks require it; it mutates the scratch buffer, counters, and
 * error state. There is no locking: an adapter must be used by one thread at
 * a time, and const access does not make concurrent calls safe. Concurrent
 * solves need one adapter each; they may share a cost only if its const
 * Evaluate() is thread-safe.
 */
template <int StorageOrder = Eigen::ColMajor>
struct LeastSquaresLMAdapter final
    : public detail::LMBackendFunctor<LeastSquaresLMAdapter<StorageOrder>, StorageOrder> {
    using Base = detail::LMBackendFunctor<LeastSquaresLMAdapter<StorageOrder>, StorageOrder>;
    using Scalar = typename Base::Scalar;

    // Largest residual count the int-typed backend callback ABI can represent
    // (Eigen's values(), TinySolver's residual count).
    static constexpr std::size_t MaxBackendResiduals = static_cast<std::size_t>(std::numeric_limits<int>::max());

    explicit LeastSquaresLMAdapter(gsl::not_null<LeastSquaresCostFunction const*> cost, ConstScalarSpan weights = {},
        bool recoverNonFinite = false)
        : Base { cost->NumResiduals(), cost->NumParameters() }
        , cost_(cost)
        , weights_(weights)
        , recoverNonFinite_(recoverNonFinite)
    {
        if (auto validWeights = ValidateWeights(weights_, this->numResiduals_); !validWeights) {
            configurationInvalid_ = true;
            error_ = ToLeastSquaresError(validWeights.error());
        }
    }

    LeastSquaresLMAdapter(LeastSquaresLMAdapter const&) = delete;
    auto operator=(LeastSquaresLMAdapter const&) -> LeastSquaresLMAdapter& = delete;
    LeastSquaresLMAdapter(LeastSquaresLMAdapter&&) = delete;
    auto operator=(LeastSquaresLMAdapter&&) -> LeastSquaresLMAdapter& = delete;
    ~LeastSquaresLMAdapter() = default;

    // Backend callback boundary: Eigen::LevenbergMarquardt and Ceres
    // TinySolver invoke this adapter through their raw-pointer callback ABI.
    // The backend-neutral contract remains span/mdspan-like at cost_->Evaluate;
    // these pointers never cross that canonical interface.
    auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        auto const poisonOutputs = [&]() -> void {
            if (residuals != nullptr) {
                std::fill_n(residuals, this->numResiduals_, std::numeric_limits<Scalar>::quiet_NaN());
            }
            if (jacobian != nullptr) {
                std::fill_n(
                    jacobian, this->numResiduals_ * this->numParameters_, std::numeric_limits<Scalar>::quiet_NaN());
            }
        };
        if (configurationInvalid_) {
            poisonOutputs();
            return false;
        }
        // A Jacobian-only call needs internal residual scratch. Beyond the
        // backend limit that scratch would be an unreasonable (or failing)
        // allocation, so reject the shape before evaluating the wrapped cost.
        if (residuals == nullptr && ExceedsBackendLimit()) {
            RecordOversized(parameters);
            poisonOutputs();
            return false;
        }

        Operon::Span<Scalar const> params { parameters, this->numParameters_ };
        if (residuals == nullptr && residualScratch_.size() != this->numResiduals_) {
            residualScratch_.resize(this->numResiduals_);
        }
        auto* residualOut = residuals != nullptr ? residuals : residualScratch_.data();
        Operon::Span<Scalar> residualSpan { residualOut, this->numResiduals_ };

        std::optional<ScalarMatrixView> jacobianView;
        if (jacobian != nullptr) {
            ++this->jacobianCallCount_;
            using Extents = std::dextents<MemoryIndex, 2>;
            using Mapping = std::layout_stride::mapping<Extents>;
            std::array<MemoryIndex, 2> strides {};
            if constexpr (StorageOrder == Eigen::ColMajor) {
                strides = { 1, this->numResiduals_ };
            } else {
                strides = { this->numParameters_, 1 };
            }
            jacobianView = ScalarMatrixView { jacobian,
                Mapping { Extents { this->numResiduals_, this->numParameters_ }, strides } };
        }
        if (residuals != nullptr) {
            ++this->residualCallCount_;
        }
        auto result = cost_->Evaluate(params, residualSpan, jacobianView);
        if (!recoverNonFinite_ && result) {
            if (auto nonFinite = FirstNonFinite(residualSpan, jacobianView); nonFinite) {
                result = tl::unexpected(*nonFinite);
            }
        }
        if (!result) {
            if (!error_) {
                error_ = result.error(); // first error wins
            }
            poisonOutputs();
            return false;
        }

        if (!weights_.empty()) {
            for (std::size_t i = 0; i < this->numResiduals_; ++i) {
                auto const w = weights_.size() == 1 ? weights_[0] : weights_[i];
                auto const sw = std::sqrt(w);
                residualOut[i] *= sw;
                if (jacobianView) {
                    for (std::size_t j = 0; j < this->numParameters_; ++j) {
                        At(*jacobianView, i, j) *= sw;
                    }
                }
            }
        }

        return true;
    }

    // The first error observed (configuration, cost, or non-finite output); never overwritten.
    [[nodiscard]] auto Error() const -> std::optional<LeastSquaresError> const& { return error_; }

    // True when NumResiduals() cannot be represented by the backends' int
    // callback ABI (Eigen's values(), TinySolver's residual count). Such a
    // problem is never handed to a solver or given internal scratch.
    [[nodiscard]] auto ExceedsBackendLimit() const noexcept -> bool
    {
        return this->numResiduals_ > MaxBackendResiduals;
    }

    // Records, as the first error, why an ExceedsBackendLimit() problem cannot
    // run, without allocating, evaluating the wrapped cost, or counting a call.
    // The backend residual limit is the reported expected size. Parameters must
    // hold NumParameters() values.
    void RecordOversized(Scalar const* /*parameters*/) const
    {
        if (!error_) {
            error_ = LeastSquaresError { .Code = LeastSquaresErrorCode::InvalidShape,
                .Expected = MaxBackendResiduals,
                .Actual = this->numResiduals_ };
        }
    }

private:
    gsl::not_null<LeastSquaresCostFunction const*> cost_;
    ConstScalarSpan weights_;
    bool recoverNonFinite_;
    bool configurationInvalid_ { false };
    mutable std::vector<Scalar> residualScratch_;
    mutable std::optional<LeastSquaresError> error_;

    // Locates the first non-finite residual (Row set) or Jacobian entry (Row/Column set).
    static auto FirstNonFinite(ConstScalarSpan residuals, std::optional<ScalarMatrixView> const& jacobian)
        -> std::optional<LeastSquaresError>
    {
        for (std::size_t i = 0; i < residuals.size(); ++i) {
            if (!std::isfinite(static_cast<double>(residuals[i]))) {
                return LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation, .Row = i };
            }
        }
        if (jacobian) {
            for (std::size_t i = 0; i < jacobian->extent(0); ++i) {
                for (std::size_t j = 0; j < jacobian->extent(1); ++j) {
                    if (!std::isfinite(static_cast<double>(At(*jacobian, i, j)))) {
                        return LeastSquaresError {
                            .Code = LeastSquaresErrorCode::NonFiniteEvaluation, .Row = i, .Column = j
                        };
                    }
                }
            }
        }
        return std::nullopt;
    }
};

} // namespace Operon

#endif
