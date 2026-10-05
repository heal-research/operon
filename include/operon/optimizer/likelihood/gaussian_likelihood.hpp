// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_GAUSSIAN_LIKELIHOOD_HPP
#define OPERON_GAUSSIAN_LIKELIHOOD_HPP

#include <Eigen/Core>
#include <cmath>
#include <limits>

#include "operon/core/concepts.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/memory_view.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/sum_of_squared_errors.hpp"
#include "operon/optimizer/fisher_information.hpp"
#include <vstat/vstat.hpp>

namespace Operon {

namespace detail {
    struct SquaredResidual {
        template <Operon::Concepts::Arithmetic T> auto operator()(T const x, T const y) const -> T
        {
            auto const e = x - y;
            return e * e;
        }
    };
} // namespace detail

// Pure static struct satisfying Concepts::Likelihood.
// Use this type anywhere only the statistical computation is needed
// (e.g. MinimumDescriptionLengthEvaluator, LikelihoodEvaluator).
template <typename T = Operon::Scalar> struct GaussianLikelihood {
    using Scalar = T;

    static constexpr bool UsesSigma = true; // sigma is required; empty span is invalid

    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> s) noexcept -> Scalar
    {
        EXPECT(!s.empty());
        static_assert(std::is_arithmetic_v<Scalar>);
        auto const n { std::ssize(x) };
        constexpr Scalar z { 0.5 };

        if (s.size() == 1) {
            auto s2 = s[0] * s[0];
            auto ssr
                = vstat::univariate::accumulate<Scalar>(x.begin(), x.end(), y.begin(), detail::SquaredResidual {}).sum;
            return z * (n * std::log(Operon::Math::Tau * s2) + ssr / s2);
        }

        if (s.size() == x.size()) {
            auto const t = std::sqrt(Operon::Math::Tau);
            auto sum { 0.0 };
            for (auto i = 0; i < n; ++i) {
                auto const si { s[i] };
                auto const ei { x[i] - y[i] };
                auto const pi { ei / si };
                sum += std::log(si * t) + z * pi * pi;
            }
            return sum;
        }

        return std::numeric_limits<Operon::Scalar>::quiet_NaN();
    }

    /**
     * Canonical Fisher-diagonal contract for callers that only need the
     * per-coefficient MDL term. This overload is available only for
     * `Operon::Scalar`, matching the canonical view contract. `jacobian` is
     * logically (row, coefficient) with arbitrary valid strides; `diagonal`
     * has exactly one element per coefficient. Sigma is scalar or per row and
     * must be finite positive.
     */
    static auto ComputeFisherDiagonal(Span<Scalar const> pred, ConstScalarMatrixView jacobian, Span<Scalar const> sigma,
        ScalarSpan diagonal) -> tl::expected<void, FisherError>
        requires std::same_as<Scalar, Operon::Scalar>
    {
        auto const rows = pred.size();
        auto const columns = jacobian.extent(1);
        if (jacobian.extent(0) != rows) {
            return tl::unexpected(
                FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = rows, .Actual = jacobian.extent(0) });
        }
        if (diagonal.size() != columns) {
            return tl::unexpected(
                FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = columns, .Actual = diagonal.size() });
        }
        if (sigma.empty() || (sigma.size() != 1 && sigma.size() != rows)) {
            return tl::unexpected(
                FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = rows, .Actual = sigma.size() });
        }
        for (auto const value : sigma) {
            if (!std::isfinite(static_cast<double>(value)) || value <= Scalar { 0 }) {
                return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidSigma });
            }
        }
        for (std::size_t column = 0; column < columns; ++column) {
            AccumulationScalar sum {};
            for (std::size_t row = 0; row < rows; ++row) {
                auto const value = static_cast<AccumulationScalar>(At(jacobian, row, column));
                auto const sigmaAt = static_cast<AccumulationScalar>(sigma.size() == 1 ? sigma.front() : sigma[row]);
                sum += (value * value) / (sigmaAt * sigmaAt);
            }
            auto const result = static_cast<Scalar>(sum);
            if (!std::isfinite(sum) || !std::isfinite(static_cast<double>(result))) {
                return tl::unexpected(FisherError { .Code = FisherErrorCode::NonFiniteResult });
            }
            diagonal[column] = result;
        }
        return {};
    }
};

} // namespace Operon

#endif
