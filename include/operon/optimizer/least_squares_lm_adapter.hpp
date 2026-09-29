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
 * wrapped cost; no duplicated count is accepted. weights is empty
 * (unweighted), size 1 (uniform), or NumResiduals() (per-row); values are
 * validated once (finite, nonnegative) at construction. Weighting scales
 * each residual and Jacobian row by sqrt(weight_i), the standard WLS-via-LM
 * trick that makes the unweighted normal equations solve sum(w_i * r_i^2).
 * weights are numerical WLS weights, never interpreted as statistical sigma.
 */
template <int StorageOrder = Eigen::ColMajor>
struct LeastSquaresLMAdapter final : public detail::LMBackendFunctor<LeastSquaresLMAdapter<StorageOrder>, StorageOrder> {
    using Base = detail::LMBackendFunctor<LeastSquaresLMAdapter<StorageOrder>, StorageOrder>;
    using Scalar = typename Base::Scalar;

    explicit LeastSquaresLMAdapter(
        gsl::not_null<LeastSquaresCostFunction const*> cost,
        ConstScalarSpan weights = {},
        bool recoverNonFinite = false)
        : Base { cost->NumResiduals(), cost->NumParameters() }
        , cost_(cost)
        , weights_(weights)
        , recoverNonFinite_(recoverNonFinite)
        , residualScratch_(cost->NumResiduals())
    {
        EXPECT(weights_.empty() || weights_.size() == 1 || weights_.size() == this->numResiduals_);
        EXPECT(detail::AllFinite(weights_, /*requireNonnegative=*/true));
    }

    // Backend callback boundary: Eigen::LevenbergMarquardt and Ceres
    // TinySolver invoke this adapter through their raw-pointer callback ABI.
    // The backend-neutral contract remains span/mdspan-like at cost_->Evaluate;
    // these pointers never cross that canonical interface.
    auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        Operon::Span<Scalar const> params { parameters, this->numParameters_ };
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
            jacobianView = ScalarMatrixView { jacobian, Mapping { Extents { this->numResiduals_, this->numParameters_ }, strides } };
        }
        if (residuals != nullptr) {
            ++this->residualCallCount_;
        }
        auto result = cost_->Evaluate(params, residualSpan, jacobianView);
        if (!recoverNonFinite_ && result && (!detail::AllFinite(residualSpan, false)
                           || (jacobianView && [&] {
                                  for (std::size_t i = 0; i < this->numResiduals_; ++i) {
                                      for (std::size_t j = 0; j < this->numParameters_; ++j) {
                                          if (!std::isfinite(static_cast<double>(At(*jacobianView, i, j)))) return true;
                                      }
                                  }
                                  return false;
                              }()))) {
            result = tl::unexpected(LeastSquaresError { .Code = LeastSquaresErrorCode::NonFiniteEvaluation });
        }
        if (!result) {
            error_ = result.error();
            if (residuals != nullptr) {
                std::fill_n(residuals, this->numResiduals_, std::numeric_limits<Scalar>::quiet_NaN());
            }
            if (jacobian != nullptr) {
                std::fill_n(jacobian, this->numResiduals_ * this->numParameters_, std::numeric_limits<Scalar>::quiet_NaN());
            }
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

    [[nodiscard]] auto Error() const -> std::optional<LeastSquaresError> const& { return error_; }

private:
    gsl::not_null<LeastSquaresCostFunction const*> cost_;
    ConstScalarSpan weights_;
    bool recoverNonFinite_;
    mutable std::vector<Scalar> residualScratch_;
    mutable std::optional<LeastSquaresError> error_;
};

} // namespace Operon

#endif
