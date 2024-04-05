// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_POISSON_LIKELIHOOD_HPP
#define OPERON_POISSON_LIKELIHOOD_HPP

#include "operon/core/concepts.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "likelihood_base.hpp"

#include <functional>
#include <type_traits>
#include <vstat/univariate.hpp>
#include <vstat/vstat.hpp>
namespace Operon {

namespace detail {
    struct Poisson {
        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T {
            return x - y * std::log(x) + std::lgamma(y+1);
        }

        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T {
            return (*this)(w * x, y);
        }
    };

    struct PoissonLog {
        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y) const -> T {
            return std::exp(x) - x * y + std::lgamma(y+1);
        }

        template<Operon::Concepts::Arithmetic T>
        auto operator()(T const x, T const y, T const w) const -> T {
            return (*this)(x * w, y);
        }
    };
} // namespace detail

template<typename T = Operon::Scalar, bool LogInput = true>
struct PoissonLikelihood : public LikelihoodBase<T> {

    PoissonLikelihood(Operon::RandomGenerator& rng, InterpreterBase<T> const& interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, std::size_t const batchSize = 0)
        : LikelihoodBase<T>(interpreter)
        , rng_{rng}
        , target_(target)
        , range_(range)
        , batchSize_(batchSize == 0 ? range.Size() : batchSize)
        , numParameters_{static_cast<std::size_t>(interpreter.GetTree().CoefficientsCount())}
        , numResiduals_{range_.Size()}
        , jac_{batchSize_, numParameters_}
    { }

    using Scalar   = typename LikelihoodBase<T>::Scalar;
    using scalar_t = Scalar; // needed by lbfgs library NOLINT

    using Vector   = typename LikelihoodBase<T>::Vector;
    using Ref      = typename LikelihoodBase<T>::Ref;
    using Cref     = typename LikelihoodBase<T>::Cref;
    using Matrix   = typename LikelihoodBase<T>::Matrix;

    // this loss can be used by the SGD or LBFGS optimizers
    auto operator()(Cref x, Ref g) const noexcept -> Operon::Scalar final {
        ++feval_;
        auto const& interpreter = this->GetInterpreter();
        Operon::Span<Operon::Scalar const> c{x.data(), static_cast<std::size_t>(x.size())};
        auto r = SelectRandomRange();
        auto p = interpreter.Evaluate(c, r);
        auto t = target_.subspan(r.Start(), r.Size());
        auto pmap = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(p.data(), std::ssize(p));
        auto tmap = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(t.data(), std::ssize(t));

        // compute jacobian
        if constexpr (LogInput) {
            if (g.size() != 0) {
                interpreter.JacRev(c, r, { jac_.data(), numParameters_ * batchSize_ });
                g = ((pmap.exp() - tmap).matrix().asDiagonal() * jac_.matrix()).colwise().sum();
            }
            return (pmap.exp() - tmap * pmap).sum();
        } else {
            auto tmap = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(t.data(), std::ssize(t));
            if (g.size() != 0) {
                interpreter.JacRev(c, r, { jac_.data(), numParameters_ * batchSize_ });
                g = ((1 - tmap * pmap.inverse()).matrix().asDiagonal() * jac_.matrix()).colwise().sum();
            }
            return (pmap - tmap * pmap.log()).sum();
        }
    }

    static auto ComputeLikelihood(Span<Scalar const> x, Span<Scalar const> y, Span<Scalar const> w) -> Scalar {
        using F = std::conditional_t<LogInput, detail::PoissonLog, detail::Poisson>;
        vstat::univariate_accumulator<Operon::Scalar> acc;

        if (w.empty()) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F{}(x[i], y[i]));
            }
        } else if (w.size() == 1) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F{}(x[i], y[i], w[0]));
            }
        } else if (w.size() == x.size()) {
            for (auto i = 0UL; i < x.size(); ++i) {
                acc(F{}(x[i], y[i], w[i]));
            }
        } else {
            throw std::runtime_error("incompatible weights");
        }

        return vstat::univariate_statistics(acc).sum;
    }

    static auto ComputeFisherMatrix(Span<Scalar const> pred, Span<Scalar const> jac, Span<Scalar const> /*not used*/) -> Matrix
    {
        auto const rows = pred.size();
        auto const cols = jac.size() / pred.size();
        Eigen::Map<Matrix const> m(jac.data(), rows, cols);
        Eigen::Map<Vector const> s{pred.data(), std::ssize(pred)};

        if constexpr (LogInput) {
            return (s.array().exp().matrix().asDiagonal() * m).transpose() * m;
        } else {
            return (s.array().inverse().matrix().asDiagonal() * m).transpose() * m;
        }
    }

    auto NumParameters() const -> std::size_t { return numParameters_; }
    auto NumObservations() const -> std::size_t { return numResiduals_; }
    auto FunctionEvaluations() const -> std::size_t { return feval_; }
    auto JacobianEvaluations() const -> std::size_t { return jeval_; }

private:
    auto SelectRandomRange() const -> Operon::Range {
        if (batchSize_ >= range_.Size()) { return range_; }
        auto s = std::uniform_int_distribution<std::size_t>{0UL, range_.Size()-batchSize_}(rng_.get());
        return Operon::Range{range_.Start() + s, range_.Start() + s + batchSize_};
    }

    std::reference_wrapper<Operon::RandomGenerator> rng_;
    Operon::Span<Operon::Scalar const> target_;
    Operon::Range const range_; // NOLINT
    std::size_t batchSize_; // batch size
    std::size_t numParameters_; // number of parameters to optimize
    std::size_t numResiduals_; // number of data points (rows)
    mutable Eigen::Array<Scalar, -1, -1> jac_;
    mutable std::size_t feval_{};
    mutable std::size_t jeval_{};
};
} // namespace Operon

#endif
