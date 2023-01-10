// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPTIMIZER_TINY_HPP
#define OPERON_OPTIMIZER_TINY_HPP

#include <Eigen/Core>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the AutodiffCalculator will compute and return the Jacobian matrix
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

template<typename DerivativeCalculator, int StorageOrder = Eigen::ColMajor>
struct CostFunction {
    static auto constexpr Storage{ StorageOrder };
    using Scalar = Operon::Scalar;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,  // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    explicit CostFunction(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Span<Operon::Scalar const> target, Operon::Range const range, DerivativeCalculator& calculator)
        : tree_{tree}
        , dataset_{dataset}
        , target_{target}
        , range_{range}
        , derivativeCalculator_{calculator}
        , numResiduals_{range.Size()}
        , numParameters_{ParameterCount(tree)}
    { }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, numParameters_ };

        if (jacobian != nullptr) {
            derivativeCalculator_.template operator()<StorageOrder>(tree_, dataset_, params, range_, jacobian);
        }

        if (residuals != nullptr) {
            Operon::Span<Operon::Scalar> result{ residuals, numResiduals_ };
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> res(residuals, numResiduals_);
            auto const& interpreter = derivativeCalculator_.GetInterpreter();
            interpreter.template operator()<Operon::Scalar>(tree_, dataset_, range_, result, params);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.subspan(range_.Start(), range_.Size()).data(), numResiduals_);
            res -= y;
        }
        return true;
    }

    // ceres solver - jacobian must be in row-major format
    // tiny solver - jacobian must be in col-major format
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
    Operon::Tree const& tree_;
    Operon::Dataset const& dataset_;
    Operon::Range const range_; // NOLINT
    DerivativeCalculator& derivativeCalculator_;
    Operon::Span<Operon::Scalar const> target_;
    std::size_t numResiduals_;
    std::size_t numParameters_;

    inline auto ParameterCount(auto const& tree) const -> std::size_t {
        auto const& nodes = tree.Nodes();
        return std::count_if(nodes.cbegin(), nodes.cend(), [](auto const& n) { return n.Optimize; });
    }
};
} // namespace Operon

#endif
