// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_COST_FUNCTION_HPP
#define OPERON_LM_COST_FUNCTION_HPP

#include <Eigen/Core>
#include <gsl/pointers>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {


template<typename T = Operon::Scalar, int StorageOrder = Eigen::ColMajor>
struct LMCostFunction {
    static auto constexpr Storage{ StorageOrder };
    using Scalar = Operon::Scalar;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,  // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    explicit LMCostFunction(gsl::not_null<InterpreterBase<T> const*> interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range)
        : interpreter_(interpreter)
        , target_{target}
        , range_{range}
        , numResiduals_{range.Size()}
        , numParameters_{static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount())}
    { }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, numParameters_ };

        if (jacobian != nullptr) {
            ++jacobianCallCount_;
            Operon::Span<Operon::Scalar> jac{jacobian, static_cast<size_t>(numResiduals_ * numParameters_)};
            interpreter_->JacRev(params, range_, jac);
        }

        if (residuals != nullptr) {
            ++residualCallCount_;
            Operon::Span<Operon::Scalar> res{ residuals, static_cast<size_t>(numResiduals_) };
            interpreter_->Evaluate(params, range_, res);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, numResiduals_);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), numResiduals_);
            x -= y;
        }
        return true;
    }

    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    [[nodiscard]] auto NumResiduals() const -> int { return numResiduals_; }
    [[nodiscard]] auto NumParameters() const -> int { return numParameters_; }

    // required by Eigen::LevenbergMarquardt
    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    // there is no real documentation but looking at Eigen unit tests, these functions should return zero
    // see: https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NonLinearOptimization.cpp
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1>& residual) const -> int
    {
        Evaluate(input.data(), residual.data(), nullptr);
        return 0;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1>& jacobian) const -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        Evaluate(input.data(), nullptr, jacobian.data());
        return 0;
    }

    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

    [[nodiscard]] auto ResidualCalls() const -> std::size_t { return residualCallCount_.load(); }
    [[nodiscard]] auto JacobianCalls() const -> std::size_t { return jacobianCallCount_.load(); }

private:
    gsl::not_null<InterpreterBase<T> const*> interpreter_;
    Operon::Span<Operon::Scalar const> target_;
    Operon::Range const range_; // NOLINT
    std::size_t numResiduals_;
    std::size_t numParameters_;

    mutable std::atomic_size_t jacobianCallCount_{0};
    mutable std::atomic_size_t residualCallCount_{0};
};
} // namespace Operon

#endif
