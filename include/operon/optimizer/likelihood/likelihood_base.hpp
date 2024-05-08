// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_LIKELIHOOD_BASE_HPP
#define OPERON_LIKELIHOOD_BASE_HPP

#include <Eigen/Core>
#include <vstat/vstat.hpp>

#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

namespace Concepts {
    template<typename T>
    concept Likelihood = requires(
        Operon::Span<Operon::Scalar const> x,
        Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> z
    ) {
        { T::ComputeLikelihood(x, y, z) } -> std::same_as<Operon::Scalar>;
        { T::ComputeFisherMatrix(z, y, z) } -> std::convertible_to<Eigen::template Matrix<Operon::Scalar, -1, -1>>;
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

    explicit LikelihoodBase(Operon::InterpreterBase<T> const& interpreter)
        : interpreter_(interpreter)
    {
    }

    [[nodiscard]] auto GetInterpreter() const -> InterpreterBase<Operon::Scalar> const& { return interpreter_.get(); }

    // compute function and gradient when called by the optimizer
    [[nodiscard]] virtual auto operator()(Cref, Ref) const noexcept -> Scalar = 0;

    // compute the likelihood value when called standalone
    [[nodiscard]] virtual auto FunctionEvaluations() const -> std::size_t = 0;
    [[nodiscard]] virtual auto JacobianEvaluations() const -> std::size_t = 0;
    [[nodiscard]] virtual auto NumParameters() const -> std::size_t = 0;
    [[nodiscard]] virtual auto NumObservations() const -> std::size_t = 0;

private:
    std::reference_wrapper<Operon::InterpreterBase<Operon::Scalar> const> interpreter_;
};
} // namespace Operon

#endif