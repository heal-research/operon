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
#include "operon/core/types.hpp"
#include "operon/error_metrics/sum_of_squared_errors.hpp"
#include <vstat/vstat.hpp>

namespace Operon {

namespace detail {
    struct SquaredResidual {
        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T
        {
            auto const e = x - y;
            return e * e;
        }

        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T
        {
            auto const e = w * (x - y);
            return e * e;
        }
    };
} // namespace detail

// Pure static struct satisfying Concepts::Likelihood.
// Use this type anywhere only the statistical computation is needed
// (e.g. MinimumDescriptionLengthEvaluator, LikelihoodEvaluator).
template <typename T = Operon::Scalar>
struct GaussianLikelihood {
    using Scalar = T;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;
    using Vector = Eigen::Matrix<Scalar, -1, 1>;

    static constexpr bool UsesSigma = true; // sigma is required; empty span is invalid

    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> s) noexcept -> Scalar
    {
        EXPECT(!s.empty());
        static_assert(std::is_arithmetic_v<Scalar>);
        auto const n { std::ssize(x) };
        constexpr Scalar z { 0.5 };

        if (s.size() == 1) {
            auto s2 = s[0] * s[0];
            auto ssr = vstat::univariate::accumulate<Scalar>(x.begin(), x.end(), y.begin(), detail::SquaredResidual {}).sum;
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

    static auto ComputeFisherMatrix(Span<Scalar const> pred, Span<Scalar const> jac, Span<Scalar const> sigma) -> Matrix
    {
        EXPECT(!sigma.empty());
        auto const rows = pred.size();
        auto const cols = jac.size() / pred.size();
        Eigen::Map<Matrix const> m(jac.data(), rows, cols);
        if (sigma.size() == 1) {
            auto const s2 = sigma[0] * sigma[0];
            Matrix f = m.transpose() * m;
            f.array() /= s2;
            return f;
        }
        EXPECT(sigma.size() == rows);
        Eigen::Map<Vector const> s { sigma.data(), std::ssize(pred) };
        // F = J^T diag(1/σᵢ²) J = (diag(1/σᵢ) J)^T (diag(1/σᵢ) J)
        Matrix scaledJ = s.array().inverse().matrix().asDiagonal() * m;
        return scaledJ.transpose() * scaledJ;
    }
};

} // namespace Operon

#endif
