// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_GAUSSIAN_LIKELIHOOD_HPP
#define OPERON_GAUSSIAN_LIKELIHOOD_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <limits>

#include "operon/core/concepts.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/sum_of_squared_errors.hpp"
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

// Pure static struct satisfying Concepts::Likelihood.
// Use this type anywhere only the statistical computation is needed
// (e.g. MinimumDescriptionLengthEvaluator, LevenbergMarquardtOptimizer).
template<typename T = Operon::Scalar>
struct GaussianLikelihood {
    using Scalar = T;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;
    using Vector = Eigen::Matrix<Scalar, -1,  1>;

    static constexpr bool UsesSigma = true; // sigma is required; empty span is invalid

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
        if (sigma.size() == 1) {
            auto const s2 = sigma[0] * sigma[0];
            Matrix f = m.transpose() * m;
            f.array() /= s2;
            return f;
        }
        EXPECT(sigma.size() == rows);
        Eigen::Map<Vector const> s{sigma.data(), std::ssize(pred)};
        // F = J^T diag(1/σᵢ²) J = (diag(1/σᵢ) J)^T (diag(1/σᵢ) J)
        Matrix scaledJ = s.array().inverse().matrix().asDiagonal() * m;
        return scaledJ.transpose() * scaledJ;
    }
};

// Callable loss object for gradient-based optimizers (L-BFGS, SGD).
// Inherits LikelihoodBase<T> for the virtual operator() interface;
// static methods delegate to GaussianLikelihood<T> so this type also
// satisfies Concepts::Likelihood and can be passed to
// OptimizerBase::ComputeLikelihood / ComputeFisherMatrix.
template<typename T = Operon::Scalar>
struct GaussianLoss : public LikelihoodBase<T> {
    static constexpr bool UsesSigma = true;

    // Diagnostic cost reported in OptimizerSummary (LBFGSOptimizer/SGDOptimizer):
    // must match what operator() actually optimizes, so weighted here too.
    template<typename Pred>
    static auto Cost(Pred const& pred, Operon::Span<Operon::Scalar const> target, Operon::Span<Operon::Scalar const> weights) noexcept -> Operon::Scalar {
        return static_cast<Operon::Scalar>(weights.empty()
            ? 0.5 * Operon::SumOfSquaredErrors(pred.begin(), pred.end(), target.begin())
            : 0.5 * Operon::SumOfSquaredErrors(pred.begin(), pred.end(), target.begin(), weights.begin()));
    }

    // `target` and `weights` must span the *whole* dataset column (absolute,
    // dataset-row-indexed - the same coordinate system used for `range` and
    // for any sub-range handed to the interpreter), not a slice pre-cut to
    // `range`. SelectBatch() below picks a sub-range of `range` for
    // minibatching and indexes target_/weights_ with that sub-range's
    // absolute Start() directly - no separate local offset to derive or keep
    // in sync, because target_/weights_ already use the same coordinates.
    GaussianLoss(gsl::not_null<Operon::RandomGenerator*> rng, gsl::not_null<InterpreterBase<T> const*> interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, std::size_t const batchSize = 0, Operon::Span<Operon::Scalar const> weights = {})
        : LikelihoodBase<T>(interpreter)
        , rng_(rng)
        , target_{target.data(), std::ssize(target)}
        , range_{range}
        , weights_{weights}
        , bs_{batchSize == 0 ? range.Size() : batchSize}
        , np_{static_cast<std::size_t>(interpreter->GetTree()->CoefficientsCount())}
        , nr_{range_.Size()}
        , jac_{bs_, np_}
    {
        EXPECT(range_.Start() + range_.Size() <= static_cast<std::size_t>(target_.size()));
        EXPECT(weights_.empty() || range_.Start() + range_.Size() <= weights_.size());
        // Only rows inside range_ are ever read (via SelectBatch); weights_ may
        // legitimately hold negative/placeholder values outside range_ (e.g. for
        // test/validation rows), so validate just the in-range slice. This can't
        // be hoisted to Dataset::SetWeights (validate-once-at-the-source) because
        // SetWeights has no notion of which rows any given Problem will train on.
        EXPECT(weights_.empty() || std::all_of(weights_.begin() + static_cast<std::ptrdiff_t>(range_.Start()), weights_.begin() + static_cast<std::ptrdiff_t>(range_.Start() + range_.Size()), [](auto w) { return w >= Operon::Scalar{0}; }));
    }

    using Scalar   = typename LikelihoodBase<T>::Scalar;

    using Vector   = typename LikelihoodBase<T>::Vector;
    using Ref      = typename LikelihoodBase<T>::Ref;
    using Cref     = typename LikelihoodBase<T>::Cref;
    using Matrix   = typename LikelihoodBase<T>::Matrix;

    // Callable by L-BFGS / SGD optimizers: returns loss and fills gradient.
    auto operator()(Cref x, Ref grad) const noexcept -> Operon::Scalar final {
        ++feval_;
        auto const& interpreter = this->GetInterpreter();
        Operon::Span<Operon::Scalar const> c{x.data(), static_cast<std::size_t>(x.size())};
        auto const batch = SelectBatch();
        auto primal = interpreter->Evaluate(c, batch);
        auto target = target_.segment(batch.Start(), batch.Size());
        Eigen::Map<Eigen::Array<Scalar, -1, 1> const> primalMap{primal.data(), std::ssize(primal)};
        auto e = primalMap - target;

        if (weights_.empty()) {
            if (grad.size() != 0) {
                assert(grad.size() == x.size());
                ++jeval_;
                interpreter->JacRev(c, batch, {jac_.data(), np_ * bs_});
                grad = (e.matrix().asDiagonal() * jac_.matrix()).colwise().sum();
            }
            return static_cast<Operon::Scalar>(e.square().sum()) * Operon::Scalar{0.5};
        }

        // Weighted loss: L = 0.5 * sum(w_i * e_i^2), dL/dtheta = sum(w_i * e_i * J_i).
        // Applied directly (rather than via the sqrt(w)-residual trick used for
        // LM) since there's no shared residual vector to keep consistent here.
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> w{weights_.data(), std::ssize(weights_)};
        auto wSeg = w.segment(batch.Start(), batch.Size());
        if (grad.size() != 0) {
            assert(grad.size() == x.size());
            ++jeval_;
            interpreter->JacRev(c, batch, {jac_.data(), np_ * bs_});
            auto we = (e * wSeg).eval();
            grad = (we.matrix().asDiagonal() * jac_.matrix()).colwise().sum();
        }

        return static_cast<Operon::Scalar>((wSeg * e.square()).sum()) * Operon::Scalar{0.5};
    }

    // Static delegation — GaussianLoss also satisfies Concepts::Likelihood.
    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> s) noexcept -> Scalar {
        return GaussianLikelihood<T>::ComputeLikelihood(x, y, s);
    }

    static auto ComputeFisherMatrix(Span<Scalar const> pred, Span<Scalar const> jac, Span<Scalar const> sigma) -> Matrix {
        return GaussianLikelihood<T>::ComputeFisherMatrix(pred, jac, sigma);
    }

    auto NumParameters() const -> std::size_t { return np_; }
    auto NumObservations() const -> std::size_t { return nr_; }
    auto FunctionEvaluations() const -> std::size_t { return feval_; }
    auto JacobianEvaluations() const -> std::size_t { return jeval_; }

private:
    // A random sub-range of range_, in the same absolute (dataset-row) coordinates
    // as range_ itself - safe to hand directly to the interpreter, and to index
    // target_/weights_ with, since both span the whole dataset column.
    auto SelectBatch() const -> Operon::Range {
        if (bs_ >= range_.Size()) { return range_; }
        auto s = std::uniform_int_distribution<std::size_t>{0UL, range_.Size()-bs_}(*rng_);
        return Operon::Range{range_.Start() + s, range_.Start() + s + bs_};
    }

    gsl::not_null<Operon::RandomGenerator*> rng_;
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> target_;
    Operon::Range const range_; // range of the training data NOLINT
    Operon::Span<Operon::Scalar const> weights_; // optional per-sample weights, empty = unweighted
    std::size_t bs_; // batch size
    std::size_t np_; // number of parameters to optimize
    std::size_t nr_; // number of data points (rows)
    mutable Eigen::Array<Scalar, -1, -1> jac_;
    mutable std::size_t feval_{};
    mutable std::size_t jeval_{};
};
} // namespace Operon

#endif
