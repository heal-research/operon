// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_LM_ADAPTER_HPP
#define OPERON_LEAST_SQUARES_LM_ADAPTER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <gsl/pointers>

#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/lm_cost_function_base.hpp"

namespace Operon {

/** Adapts a LeastSquaresCostFunction to the raw-pointer Evaluate() interface LMCostFunctionBase needs for Eigen::LevenbergMarquardt and ceres::TinySolver. */
template <int StorageOrder = Eigen::ColMajor>
struct LeastSquaresLMAdapter final : public LMCostFunctionBase<LeastSquaresLMAdapter<StorageOrder>, StorageOrder> {
    using Base = LMCostFunctionBase<LeastSquaresLMAdapter<StorageOrder>, StorageOrder>;
    using Scalar = typename Base::Scalar;

    LeastSquaresLMAdapter(gsl::not_null<LeastSquaresCostFunction const*> cost, std::size_t numResiduals)
        : Base { numResiduals, cost->NumParameters() }
        , cost_(cost)
        , residualScratch_(numResiduals)
    {
    }

    // Both solvers may request the Jacobian alone (residuals == nullptr); the
    // canonical contract always writes residuals, so that case is redirected
    // into residualScratch_ and discarded.
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
        return true;
    }

    [[nodiscard]] auto Error() const -> std::optional<LeastSquaresError> const& { return error_; }

private:
    gsl::not_null<LeastSquaresCostFunction const*> cost_;
    mutable std::vector<Scalar> residualScratch_;
    mutable std::optional<LeastSquaresError> error_;
};

} // namespace Operon

#endif
