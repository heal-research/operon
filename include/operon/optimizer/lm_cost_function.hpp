// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_COST_FUNCTION_HPP
#define OPERON_LM_COST_FUNCTION_HPP

#include <algorithm>
#include <atomic>
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

    explicit LMCostFunction(gsl::not_null<InterpreterBase<T> const*> interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, Operon::Span<Operon::Scalar const> weights = {})
        : interpreter_(interpreter)
        , target_{target}
        , range_{range}
        , weights_{weights}
        , numResiduals_{range.Size()}
        , numParameters_{static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount())}
    {
        // Validated once here rather than per-Evaluate() call: EXPECT (libassert)
        // is always-on, not debug-only, so an O(N) scan per residual evaluation
        // would be a real per-LM-iteration cost across a GP run.
        EXPECT(weights_.empty() || weights_.size() == numResiduals_);
        EXPECT(std::all_of(weights_.begin(), weights_.end(), [](auto w) { return w >= Operon::Scalar{0}; }));
    }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, numParameters_ };

        // Standard WLS-via-LM trick: scaling both the residual and its
        // Jacobian row by sqrt(w_i) makes the unweighted LM/GN normal
        // equations solve the weighted problem (sum(w_i * r_i^2)) instead.
        if (jacobian != nullptr) {
            ++jacobianCallCount_;
            Operon::Span<Operon::Scalar> jac{jacobian, static_cast<size_t>(numResiduals_ * numParameters_)};
            interpreter_->JacRev(params, range_, jac);
            if (!weights_.empty()) {
                Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, -1>> J(jacobian, static_cast<Eigen::Index>(numResiduals_), static_cast<Eigen::Index>(numParameters_));
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights_.data(), numResiduals_);
                J.array().colwise() *= w.sqrt();
            }
        }

        if (residuals != nullptr) {
            ++residualCallCount_;
            Operon::Span<Operon::Scalar> res{ residuals, static_cast<size_t>(numResiduals_) };
            interpreter_->Evaluate(params, range_, res);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, numResiduals_);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), numResiduals_);
            x -= y;
            if (!weights_.empty()) {
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights_.data(), numResiduals_);
                x *= w.sqrt();
            }
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
    Operon::Span<Operon::Scalar const> weights_;
    std::size_t numResiduals_;
    std::size_t numParameters_;

    mutable std::atomic_size_t jacobianCallCount_{0};
    mutable std::atomic_size_t residualCallCount_{0};
};
} // namespace Operon

#endif
