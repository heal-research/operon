// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_STATISTICAL_CONCEPTS_HPP
#define OPERON_STATISTICAL_CONCEPTS_HPP

#include <Eigen/Core>
#include <concepts>

#include "operon/core/types.hpp"

namespace Operon {

namespace Concepts {
    // Types satisfying Likelihood that also provide ComputeFisherMatrix.
    // Used by MDL/FBF evaluators and Fisher-information consumers. Statistical-
    // only: never included by a numerical cost/concept header.
    template <typename T>
    concept HasFisherMatrix = requires(
        Operon::Span<Operon::Scalar const> x,
        Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> z) {
        { T::ComputeFisherMatrix(x, y, z) } -> std::convertible_to<Eigen::Matrix<Operon::Scalar, -1, -1>>;
    };
} // namespace Concepts

// Optional capability for a caller that genuinely needs dynamic statistical
// diagnostics from a fitted model. Not a base of OptimizerBase or any
// numerical cost; a policy adapter (e.g. a Gaussian/Poisson diagnostics
// wrapper) implements it explicitly when a caller asks for one.
class StatisticalDiagnosticsCapability {
public:
    StatisticalDiagnosticsCapability() = default;
    StatisticalDiagnosticsCapability(StatisticalDiagnosticsCapability const&) = default;
    auto operator=(StatisticalDiagnosticsCapability const&) -> StatisticalDiagnosticsCapability& = default;
    StatisticalDiagnosticsCapability(StatisticalDiagnosticsCapability&&) = default;
    auto operator=(StatisticalDiagnosticsCapability&&) -> StatisticalDiagnosticsCapability& = default;
    virtual ~StatisticalDiagnosticsCapability() = default;

    [[nodiscard]] virtual auto ComputeLikelihood(
        Operon::Span<Operon::Scalar const> predicted,
        Operon::Span<Operon::Scalar const> observed,
        Operon::Span<Operon::Scalar const> sigmaOrExposure) const -> Operon::Scalar = 0;
    [[nodiscard]] virtual auto ComputeFisherMatrix(
        Operon::Span<Operon::Scalar const> predicted,
        Operon::Span<Operon::Scalar const> flatJacobian,
        Operon::Span<Operon::Scalar const> sigmaOrExposure) const
        -> Eigen::Matrix<Operon::Scalar, -1, -1> = 0;
};

} // namespace Operon

#endif
