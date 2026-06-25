// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_LIKELIHOOD_BASE_HPP
#define OPERON_LIKELIHOOD_BASE_HPP

#include <concepts>
#include <Eigen/Core>
#include <gsl/pointers>
#include <vstat/vstat.hpp>

#include "operon/core/concepts.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

namespace Concepts {
    // Types satisfying Likelihood that also provide ComputeFisherMatrix.
    // Used by MDL/FBF evaluators and the Levenberg-Marquardt optimizer.
    template<typename T>
    concept HasFisherMatrix = requires(
        Operon::Span<Operon::Scalar const> x,
        Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> z
    ) {
        { T::ComputeFisherMatrix(x, y, z) } -> std::convertible_to<Eigen::Matrix<Operon::Scalar, -1, -1>>;
    };
} // namespace Concepts

template <typename T = Operon::Scalar>
struct LikelihoodBase {
    using Scalar = T;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;
    using Vector = Eigen::Matrix<Scalar, -1, 1>;
    using Ref    = Eigen::Ref<Vector>;
    using Cref   = Eigen::Ref<Vector const> const&;

    using scalar_t = T; // for lbfgs solver NOLINT

    explicit LikelihoodBase(gsl::not_null<Operon::InterpreterBase<T> const*> interpreter)
        : interpreter_(interpreter)
    {
    }

    [[nodiscard]] auto GetInterpreter() const -> InterpreterBase<Operon::Scalar> const* { return interpreter_.get(); }

    // compute function and gradient when called by the optimizer
    [[nodiscard]] virtual auto operator()(Cref, Ref) const noexcept -> Scalar = 0;

    // compute the likelihood value when called standalone
    [[nodiscard]] virtual auto FunctionEvaluations() const -> std::size_t = 0;
    [[nodiscard]] virtual auto JacobianEvaluations() const -> std::size_t = 0;
    [[nodiscard]] virtual auto NumParameters() const -> std::size_t = 0;
    [[nodiscard]] virtual auto NumObservations() const -> std::size_t = 0;

private:
    gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> interpreter_;
};

namespace Concepts {
    // Tighter concept for gradient-based optimizer loss functions.
    // Requires Likelihood (static methods) plus derivation from LikelihoodBase
    // (operator(), FunctionEvaluations, JacobianEvaluations, etc.).
    // Prevents confusing template errors when a static-only *Likelihood struct
    // is mistakenly passed to LBFGSOptimizer or SGDOptimizer.
    template<typename T>
    concept OptimizerLoss = Likelihood<T> && HasFisherMatrix<T> && std::derived_from<T, LikelihoodBase<typename T::Scalar>>;
} // namespace Concepts

} // namespace Operon

#endif