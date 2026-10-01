// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_POISSON_LIKELIHOOD_HPP
#define OPERON_POISSON_LIKELIHOOD_HPP

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "operon/core/concepts.hpp"
#include "operon/core/memory_view.hpp"
#include "operon/core/types.hpp"
#include "operon/optimizer/fisher_information.hpp"
#include <vstat/univariate.hpp>
#include <vstat/vstat.hpp>

namespace Operon {

namespace detail {
    struct Poisson {
        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T
        {
            return x - y * std::log(x) + std::lgamma(y + 1);
        }

        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T
        {
            return (*this)(w * x, y);
        }
    };

    struct PoissonLog {
        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T
        {
            return std::exp(x) - x * y + std::lgamma(y + 1);
        }

        template <Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T
        {
            return (*this)(x * w, y);
        }
    };
} // namespace detail

// Pure static struct satisfying Concepts::Likelihood.
// Use this type anywhere only the statistical computation is needed
// (e.g. MinimumDescriptionLengthEvaluator).
template <typename T = Operon::Scalar, bool LogInput = true>
struct PoissonLikelihood {
    using Scalar = T;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;
    using Vector = Eigen::Matrix<Scalar, -1, 1>;

    static constexpr bool UsesSigma = false; // w is an optional weight, not sigma; empty = unweighted

    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> w) -> Scalar
    {
        using F = std::conditional_t<LogInput, detail::PoissonLog, detail::Poisson>;
        vstat::univariate_accumulator<Scalar> acc;

        if (w.empty()) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F {}(x[i], y[i]));
            }
        } else if (w.size() == 1) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F {}(x[i], y[i], w[0]));
            }
        } else if (w.size() == x.size()) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F {}(x[i], y[i], w[i]));
            }
        } else {
            throw std::runtime_error("incompatible weights");
        }

        return vstat::univariate_statistics(acc).sum;
    }

    /**
     * Canonical Fisher-diagonal contract for MDL callers. This overload is
     * available only for `Operon::Scalar`, matching the canonical view
     * contract; the legacy Eigen-returning facade below remains generic in
     * `T`. `jacobian` is logically (row, coefficient) with arbitrary valid
     * strides. The third argument is unused: Poisson Fisher information is
     * determined by the model prediction, not a sigma profile.
     */
    static auto ComputeFisherDiagonal(Span<Scalar const> prediction, ConstScalarMatrixView jacobian,
        Span<Scalar const> /*unused*/, ScalarSpan diagonal) -> tl::expected<void, FisherError>
        requires std::same_as<Scalar, Operon::Scalar>
    {
        auto const rows = prediction.size();
        auto const columns = jacobian.extent(1);
        if (jacobian.extent(0) != rows) {
            return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = rows, .Actual = jacobian.extent(0) });
        }
        if (diagonal.size() != columns) {
            return tl::unexpected(FisherError { .Code = FisherErrorCode::InvalidShape, .Expected = columns, .Actual = diagonal.size() });
        }
        for (std::size_t column = 0; column < columns; ++column) {
            AccumulationScalar sum {};
            for (std::size_t row = 0; row < rows; ++row) {
                auto const value = static_cast<AccumulationScalar>(At(jacobian, row, column));
                auto const predictionAt = static_cast<AccumulationScalar>(prediction[row]);
                auto const weight = LogInput ? std::exp(predictionAt) : AccumulationScalar {1} / predictionAt;
                sum += weight * value * value;
            }
            auto const result = static_cast<Scalar>(sum);
            if (!std::isfinite(sum) || !std::isfinite(static_cast<double>(result))) {
                return tl::unexpected(FisherError { .Code = FisherErrorCode::NonFiniteResult });
            }
            diagonal[column] = result;
        }
        return {};
    }

    static auto ComputeFisherMatrix(Span<Scalar const> pred, Span<Scalar const> jac, Span<Scalar const> /*not used*/) -> Matrix
    {
        auto const rows = pred.size();
        auto const cols = jac.size() / pred.size();
        Eigen::Map<Matrix const> m(jac.data(), rows, cols);
        Eigen::Map<Vector const> s { pred.data(), std::ssize(pred) };

        if constexpr (LogInput) {
            return (s.array().exp().matrix().asDiagonal() * m).transpose() * m;
        } else {
            return (s.array().inverse().matrix().asDiagonal() * m).transpose() * m;
        }
    }
};

} // namespace Operon

#endif
