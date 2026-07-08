// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_COST_FUNCTION_BASE_HPP
#define OPERON_LM_COST_FUNCTION_BASE_HPP

#include <algorithm>
#include <atomic>
#include <Eigen/Core>

#include "operon/core/types.hpp"

namespace Operon {

// Standard WLS-via-LM trick: scaling both the residual and its Jacobian row by
// sqrt(w_i) makes the unweighted LM/GN normal equations solve the weighted
// problem (sum(w_i * r_i^2)) instead. Shared by LMCostFunction and
// JitLMCostFunction so the two backends can't drift apart on how weighting works.
//
// `weights` here is already range-local (callers pass Problem::Weights(range),
// not the whole dataset column - see GaussianLoss for the contrast), so this
// all_of only scans the rows this cost function will actually use.
inline void ValidateLMWeights(Operon::Span<Operon::Scalar const> weights, std::size_t numResiduals)
{
    EXPECT(weights.empty() || weights.size() == numResiduals);
    EXPECT(std::all_of(weights.begin(), weights.end(), [](auto w) { return w >= Operon::Scalar{0}; }));
}

inline void ApplyLMResidualWeights(Operon::Span<Operon::Scalar const> weights, Operon::Scalar* residuals, std::size_t numResiduals)
{
    if (weights.empty()) { return; }
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, static_cast<Eigen::Index>(numResiduals));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights.data(), static_cast<Eigen::Index>(numResiduals));
    x *= w.sqrt();
}

inline void ApplyLMJacobianWeights(Operon::Span<Operon::Scalar const> weights, Operon::Scalar* jacobian, std::size_t numResiduals, std::size_t numParameters)
{
    if (weights.empty()) { return; }
    Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, -1>> j(jacobian, static_cast<Eigen::Index>(numResiduals), static_cast<Eigen::Index>(numParameters));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w(weights.data(), static_cast<Eigen::Index>(numResiduals));
    j.array().colwise() *= w.sqrt();
}

// CRTP base providing the Eigen::LevenbergMarquardt / ceres::TinySolver adapter
// boilerplate shared by LMCostFunction and JitLMCostFunction: both solvers only
// need Derived::Evaluate(parameters, residuals, jacobian), everything else here
// (the Eigen::Matrix-based overloads, values()/inputs(), call counters) is identical
// across backends.
template<typename Derived, int StorageOrder = Eigen::ColMajor>
struct LMCostFunctionBase {
    static auto constexpr Storage{ StorageOrder };
    using Scalar = Operon::Scalar;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,  // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    explicit LMCostFunctionBase(std::size_t numResiduals, std::size_t numParameters)
        : numResiduals_{numResiduals}, numParameters_{numParameters}
    { }

    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return self().Evaluate(parameters, residuals, jacobian);
    }

    // there is no real documentation but looking at Eigen unit tests, these functions should return zero
    // see: https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NonLinearOptimization.cpp
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1>& residual) const -> int
    {
        self().Evaluate(input.data(), residual.data(), nullptr);
        return 0;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1>& jacobian) const -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        self().Evaluate(input.data(), nullptr, jacobian.data());
        return 0;
    }

    [[nodiscard]] auto NumResiduals() const -> int { return static_cast<int>(numResiduals_); }
    [[nodiscard]] auto NumParameters() const -> int { return static_cast<int>(numParameters_); }
    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

    [[nodiscard]] auto ResidualCalls() const -> std::size_t { return residualCallCount_.load(); }
    [[nodiscard]] auto JacobianCalls() const -> std::size_t { return jacobianCallCount_.load(); }

protected:
    auto self() const -> Derived const& { return static_cast<Derived const&>(*this); }

    std::size_t numResiduals_;
    std::size_t numParameters_;

    mutable std::atomic_size_t jacobianCallCount_{0};
    mutable std::atomic_size_t residualCallCount_{0};
};

} // namespace Operon

#endif
