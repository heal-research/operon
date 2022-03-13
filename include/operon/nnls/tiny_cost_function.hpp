// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_NNLS_TINY_OPTIMIZER
#define OPERON_NNLS_TINY_OPTIMIZER

#include <Eigen/Core>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the CostFunctor is the actual functor for computing the residuals
// - the Dual type represents a dual number, the user can specify the type for the Scalar part (float, double) and the Stride (Ceres-specific)
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

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
        if (residuals != nullptr && jacobian == nullptr) {
            return functor_(parameters, residuals);
        }

        auto np = NumParameters();
        auto nr = NumResiduals();

        Operon::Vector<Dual> inputs(np);
        Operon::Vector<Dual> outputs(nr);

        for (int i = 0; i < np; ++i) {
            inputs[i].a = parameters[i];
            inputs[i].v.setZero();
        }
        Eigen::Map<Eigen::Matrix<Scalar, -1, -1, StorageOrder>> jmap(jacobian, nr, np);

        // Evaluate all of the strides. Each stride is a chunk of the derivative to evaluate,
        // typically some size proportional to the size of the SIMD registers of the CPU.
        int ns = static_cast<int>(std::ceil(static_cast<float>(np) / static_cast<float>(Stride)));

        for (int s = 0; s < ns * Stride; s += Stride) {
            // Set most of the jet components to zero, except for non-constant #Stride parameters.
            int r = std::min(np, s + Stride); // remaining parameters

            for (int i = s; i < r; ++i) {
                inputs[i].v[i - s] = 1.0;
            }

            if (!functor_(inputs.data(), outputs.data())) {
                return false;
            }

            for (int i = s; i < r; ++i) {
                inputs[i].v[i - s] = 0.0;
                std::transform(outputs.begin(), outputs.end(), jmap.col(i).data(), [&](auto const& jet) { return jet.v[i - s]; });
            }
        }

        if (residuals != nullptr) {
            std::transform(outputs.begin(), outputs.end(), residuals, [](auto const& jet) { return jet.a; });
        }
        return true;
    }

    // required by tiny solver
    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    [[nodiscard]] auto NumResiduals() const -> int { return functor_.NumResiduals(); }
    [[nodiscard]] auto NumParameters() const -> int { return functor_.NumParameters(); }

    // required by Eigen::LevenbergMarquardt
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1> &residual) -> int
    {
        return Evaluate(input.data(), residual.data(), nullptr);
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1> &jacobian) -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        return Evaluate(input.data(), nullptr, jacobian.data());
    }

    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

private:
    CostFunctor functor_;
};
} // namespace Operon

#endif
