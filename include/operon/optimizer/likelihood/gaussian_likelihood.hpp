// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GAUSSIAN_LIKELIHOOD_HPP
#define OPERON_GAUSSIAN_LIKELIHOOD_HPP

#include <array>
#include <functional>
#include <limits>

#include "operon/core/concepts.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "likelihood_base.hpp"

namespace Operon {

namespace detail {
    struct SquaredResidual {
        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T {
            auto const e = x - y;
            return e * e;
        }

        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T {
            auto const e = w * (x - y);
            return e * e;
        }
    };
} // namespace detail

template<typename T = Operon::Scalar>
struct GaussianLikelihood : public LikelihoodBase<T> {
    GaussianLikelihood(Operon::RandomGenerator& rng, InterpreterBase<T> const& interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, std::size_t const batchSize = 0)
        : LikelihoodBase<T>(interpreter)
        , rng_(rng)
        , target_{target.data(), std::ssize(target)}
        , range_{range}
        , bs_{batchSize == 0 ? range.Size() : batchSize}
        , np_{static_cast<std::size_t>(interpreter.GetTree().CoefficientsCount())}
        , nr_{range_.Size()}
        , jac_{bs_, np_}
    { }

    using Scalar   = typename LikelihoodBase<T>::Scalar;

    using Vector   = typename LikelihoodBase<T>::Vector;
    using Ref      = typename LikelihoodBase<T>::Ref;
    using Cref     = typename LikelihoodBase<T>::Cref;
    using Matrix   = typename LikelihoodBase<T>::Matrix;

    // this loss can be used by the SGD or LBFGS optimizers
    auto operator()(Cref x, Ref grad) const noexcept -> Operon::Scalar final {
        ++feval_;
        auto const& interpreter = this->GetInterpreter();
        Operon::Span<Operon::Scalar const> c{x.data(), static_cast<std::size_t>(x.size())};
        auto range = SelectRandomRange();
        auto primal = interpreter.Evaluate(c, range);
        auto target = target_.segment(range.Start(), range.Size());
        Eigen::Map<Eigen::Array<Scalar, -1, 1> const> primalMap{primal.data(), std::ssize(primal)};
        auto e = primalMap - target;

        if (grad.size() != 0) {
            assert(grad.size() == x.size());
            ++jeval_;
            interpreter.JacRev(c, range, {jac_.data(), np_ * bs_});
            grad = (e.matrix().asDiagonal() * jac_.matrix()).colwise().sum();
        }

        return static_cast<Operon::Scalar>(e.square().sum()) * Operon::Scalar{0.5};
    }

    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> s) noexcept -> Scalar {
        EXPECT(!s.empty());
        static_assert(std::is_arithmetic_v<Scalar>);
        auto const n{ std::ssize(x) };
        constexpr Scalar z{0.5};

        if (s.size() == 1) {
            auto s2 = s[0] * s[0];
            auto ssr = vstat::univariate::accumulate<Scalar>(x.begin(), x.end(), y.begin(), detail::SquaredResidual{}).sum;
            return z * (n * std::log(Operon::Math::Tau * s2) + ssr / s2) ;
        }

        if (s.size() == x.size()) {
            auto const t = std::sqrt(Operon::Math::Tau);
            auto sum{0.0};
            for (auto i = 0; i < n; ++i) {
                auto const si{ s[i] };
                auto const ei{ x[i] - y[i] };
                auto const pi{ ei / si };
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
        typename LikelihoodBase<T>::Matrix f = m.transpose() * m;
        if (sigma.size() == 1) {
            auto const s2 = sigma[0] * sigma[0];
            f.array() /= s2;
        } else {
            EXPECT(sigma.size() == rows);
            Eigen::Map<Vector const> s{sigma.data(), std::ssize(pred)};
            f.array() /= s.array().square();
        }
        return f;
    }

    auto NumParameters() const -> std::size_t { return np_; }
    auto NumObservations() const -> std::size_t { return nr_; }
    auto FunctionEvaluations() const -> std::size_t { return feval_; }
    auto JacobianEvaluations() const -> std::size_t { return jeval_; }

private:
    auto SelectRandomRange() const -> Operon::Range {
        if (bs_ >= range_.Size()) { return range_; }
        auto s = std::uniform_int_distribution<std::size_t>{0UL, range_.Size()-bs_}(rng_.get());
        return Operon::Range{range_.Start() + s, range_.Start() + s + bs_};
    }

    std::reference_wrapper<Operon::RandomGenerator> rng_;
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> target_;
    Operon::Range const range_; // range of the training data NOLINT
    std::size_t bs_; // batch size
    std::size_t np_; // number of parameters to optimize
    std::size_t nr_; // number of data points (rows)
    mutable Eigen::Array<Scalar, -1, -1> jac_;
    mutable std::size_t feval_{};
    mutable std::size_t jeval_{};
};
} // namespace Operon

#endif
