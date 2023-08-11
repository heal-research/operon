// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BFGS_COST_FUNCTION_HPP
#define OPERON_BFGS_COST_FUNCTION_HPP

#include <functional>
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

template<typename T = Operon::Scalar>
struct GDCostFunction {
    GDCostFunction(InterpreterBase<T> const& interpreter, Operon::Span<Operon::Scalar const> target, Operon::Range const range, std::size_t const batchSize = 0)
        : interpreter_{interpreter}
        , target_(target)
        , range_(range)
        , bs_(batchSize == 0 ? range.Size() : batchSize)
        , np_{static_cast<std::size_t>(interpreter.GetTree().CoefficientsCount())}
        , nr_{range_.Size()}
        , jac_{bs_, np_}
    { }

    using Scalar   = Operon::Scalar;
    using Vector   = Eigen::Matrix<Scalar, -1, 1>; // NOLINT
    using scalar_t = Operon::Scalar; // needed by lbfgs library NOLINT

    // this loss can be used by the SGD or LBFGS optimizers
    auto operator()(Eigen::Ref<Vector const> const& x, Eigen::Ref<Vector> g) const noexcept -> Operon::Scalar {
        ++callcount_;
        std::vector<Operon::Scalar> coeff{x.begin(), x.end()};
        Operon::Span<Operon::Scalar> c{coeff.data(), coeff.size()};
        auto r = SelectRandomRange();
        auto p = interpreter_.get().Evaluate(c, r);
        auto t = target_.subspan(r.Start(), r.Size());
        auto pmap = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(p.data(), std::ssize(p));
        auto tmap = Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const>(t.data(), std::ssize(t));
        auto e = pmap - tmap;

        interpreter_.get().JacRev(c, r, {jac_.data(), np_ * bs_});
        g = (e.matrix().asDiagonal() * jac_.matrix()).colwise().sum();
        return static_cast<Operon::Scalar>(e.square().sum()) * Operon::Scalar{0.5};
    }

    auto NumParameters() const { return np_; }
    auto NumObservations() const { return nr_; }
    auto CallCount() const { return callcount_; }

private:
    auto SelectRandomRange() const -> Operon::Range {
        if (bs_ >= range_.Size()) { return range_; }
        auto s = std::uniform_int_distribution<std::size_t>{0UL, range_.Size()-bs_}(rng_);
        return Operon::Range{range_.Start() + s, range_.Start() + s + bs_};
    }

    std::reference_wrapper<InterpreterBase<T> const> interpreter_;
    Operon::Span<Operon::Scalar const> target_;
    Operon::Range const range_; // NOLINT
    std::size_t bs_; // batch size
    std::size_t np_; // number of parameters to optimize
    std::size_t nr_; // number of data points (rows)
    mutable Eigen::Array<Scalar, -1, -1> jac_;
    mutable Operon::RandomGenerator rng_{0};
    mutable std::size_t callcount_{};
};
} // namespace Operon

#endif
