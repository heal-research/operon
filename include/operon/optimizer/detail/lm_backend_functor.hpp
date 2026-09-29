// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LM_BACKEND_FUNCTOR_HPP
#define OPERON_LM_BACKEND_FUNCTOR_HPP

#include <Eigen/Core>
#include <atomic>
#include <cstddef>

#include "operon/core/types.hpp"

namespace Operon::detail {

// Private CRTP base providing the Eigen::LevenbergMarquardt / ceres::TinySolver
// adapter boilerplate: both solvers only need Derived::Evaluate(parameters,
// residuals, jacobian), everything else here (the Eigen::Matrix-based
// overloads, values()/inputs(), call counters) is identical across backends.
template <typename Derived, int StorageOrder = Eigen::ColMajor>
struct LMBackendFunctor {
    static auto constexpr Storage { StorageOrder };
    using Scalar = Operon::Scalar;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic, // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver = Eigen::ColPivHouseholderQR<JacobianType>;

    explicit LMBackendFunctor(std::size_t numResiduals, std::size_t numParameters)
        : numResiduals_ { numResiduals }
        , numParameters_ { numParameters }
    {
    }

    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return self().Evaluate(parameters, residuals, jacobian);
    }

    // there is no real documentation but looking at Eigen unit tests, these functions should return zero
    // see: https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NonLinearOptimization.cpp
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1>& residual) const -> int
    {
        return self().Evaluate(input.data(), residual.data(), nullptr) ? 0 : -1;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1>& jacobian) const -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        return self().Evaluate(input.data(), nullptr, jacobian.data()) ? 0 : -1;
    }

    [[nodiscard]] auto NumResiduals() const -> int { return static_cast<int>(numResiduals_); }
    [[nodiscard]] auto NumParameters() const -> int { return static_cast<int>(numParameters_); }
    [[nodiscard]] auto values() const -> int { return NumResiduals(); } // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

    [[nodiscard]] auto ResidualCalls() const -> std::size_t { return residualCallCount_.load(); }
    [[nodiscard]] auto JacobianCalls() const -> std::size_t { return jacobianCallCount_.load(); }

protected:
    auto self() const -> Derived const& { return static_cast<Derived const&>(*this); }

    std::size_t numResiduals_;
    std::size_t numParameters_;

    mutable std::atomic_size_t jacobianCallCount_ { 0 };
    mutable std::atomic_size_t residualCallCount_ { 0 };
};

} // namespace Operon::detail

#endif
