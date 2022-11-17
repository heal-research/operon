// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_OPTIMIZER_TINY_HPP
#define OPERON_OPTIMIZER_TINY_HPP

#include <Eigen/Core>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the CostFunctor is the actual functor for computing the residuals
// - the Dual type represents a dual number, the user can specify the type for the Scalar part (float, double) and the Stride (Ceres-specific)
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

namespace detail {
    template<typename CostFunctor, typename Dual, typename Scalar, int JacobianLayout = Eigen::ColMajor>
    inline auto Autodiff(CostFunctor const& function, Scalar const* parameters, Scalar* residuals, Scalar* jacobian) -> bool
    {
        static_assert(std::is_convertible_v<typename Dual::Scalar, Scalar>, "The chosen Jet and Scalar types are not compatible.");
        static_assert(std::is_convertible_v<Scalar, typename Dual::Scalar>, "The chosen Jet and Scalar types are not compatible.");

        EXPECT(parameters != nullptr);
        EXPECT(residuals != nullptr || jacobian != nullptr);

        if (jacobian == nullptr) {
            return function(parameters, residuals);
        }

        std::vector<Dual> inputs(function.NumParameters());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i].a = parameters[i];
            inputs[i].v.setZero();
        }
        std::vector<Dual> outputs(function.NumResiduals());

        static auto constexpr dim{Dual::DIMENSION};
        Eigen::Map<Eigen::Matrix<Scalar, -1, -1, JacobianLayout>> jmap(jacobian, outputs.size(), inputs.size());

        for (auto s = 0U; s < inputs.size(); s += dim) {
            auto r = std::min(static_cast<uint32_t>(inputs.size()), s + dim); // remaining parameters

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 1.0;
            }

            if (!function(inputs.data(), outputs.data())) {
                return false;
            }

            for (auto i = s; i < r; ++i) {
                inputs[i].v[i - s] = 0.0;
            }

            // fill in the jacobian trying to exploit its layout for efficiency
            if constexpr (JacobianLayout == Eigen::ColMajor) {
                for (auto i = s; i < r; ++i) {
                    std::transform(outputs.cbegin(), outputs.cend(), jmap.col(i).data(), [&](auto const& jet) { return jet.v[i - s]; });
                }
            } else {
                for (auto i = 0; i < outputs.size(); ++i) {
                    std::copy_n(outputs[i].v.data(), r - s, jmap.row(i).data() + s);
                }
            }
        }
        if (residuals != nullptr) {
            std::transform(std::cbegin(outputs), std::cend(outputs), residuals, [](auto const& jet) { return jet.a; });
        }
        return true;
    }
} // namespace detail

template <typename CostFunctor, typename DualType, typename ScalarType, int StorageOrder = Eigen::RowMajor>
struct TinyCostFunction {
    static constexpr int Stride = DualType::DIMENSION;
    static constexpr int Storage = StorageOrder;
    using Scalar = ScalarType;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,  // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    explicit TinyCostFunction(CostFunctor const& functor)
        : functor_(functor)
    {
    }

    auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return detail::Autodiff<CostFunctor, DualType, ScalarType, StorageOrder>(functor_, parameters, residuals, jacobian);
    }

    // ceres solver - jacobian must be in row-major format
    // ceres tiny solver - jacobian must be in col-major format
    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    [[nodiscard]] auto NumResiduals() const -> int { return functor_.NumResiduals(); }
    [[nodiscard]] auto NumParameters() const -> int { return functor_.NumParameters(); }

    // required by Eigen::LevenbergMarquardt
    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    // there is no real documentation but looking at Eigen unit tests, these functions should return zero
    // see: https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NonLinearOptimization.cpp
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1> &residual) -> int
    {
        Evaluate(input.data(), residual.data(), nullptr);
        return 0;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1> &jacobian) -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        Evaluate(input.data(), nullptr, jacobian.data());
        return 0;
    }

    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

private:
    CostFunctor functor_;
};
} // namespace Operon

#endif
