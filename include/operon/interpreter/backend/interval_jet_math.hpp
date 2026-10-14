// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_BACKEND_INTERVAL_JET_MATH_HPP
#define OPERON_BACKEND_INTERVAL_JET_MATH_HPP

#include <cmath>
#include <concepts>
#include <limits>

#include <pappus/pappus.hpp>

#include "operon/core/jet.hpp"

namespace Operon {

// JetMath (see core/jet.hpp) for scalar intervals: Jet<pappus::interval<T>, N> propagates an enclosure of the
// primal and of each of the N partial derivatives over a whole input box. Every function returns a (directed
// rounding) enclosure of the range of the corresponding real function over the argument interval; a domain
// error (log of a negative interval, ...) yields an empty interval, exactly as the IntervalEvaluator rules do.
//
// Dependency-blind interval arithmetic cannot tell that x * x is nonnegative; the sqr hook maps to
// interval::square so the derivative formulas of tan/tanh/atan/asin/acos/cbrt/aq stay tight.
//
// The non-smooth hooks (sgn, floor, ceil, fmin/fmax and their weights) are sound hulls: sgn is the hull of the
// pointwise signs and the min/max weights are [0, 1] wherever the operands overlap. They exist so the generic
// jet functions instantiate for intervals; IntervalJetEvaluator does not differentiate those operations.
template <std::floating_point T> struct JetMath<pappus::interval<T>> {
    using I = pappus::interval<T>;

    static auto abs(I const& x) -> I { return x.abs(); }
    static auto sqr(I const& x) -> I { return x.square(); }
    static auto sgn(I const& x) -> I
    {
        if (x.is_empty()) {
            return I::empty();
        }
        auto const lo = x.inf() > T { 0 } ? T { 1 } : (x.inf() < T { 0 } ? T { -1 } : T { 0 });
        auto const hi = x.sup() > T { 0 } ? T { 1 } : (x.sup() < T { 0 } ? T { -1 } : T { 0 });
        return I { lo, hi };
    }
    static auto sqrt(I const& x) -> I { return x.sqrt(); }
    static auto cbrt(I const& x) -> I { return x.cbrt(); }
    static auto exp(I const& x) -> I { return x.exp(); }
    static auto log(I const& x) -> I { return x.log(); }
    static auto log1p(I const& x) -> I { return x.log1p(); }
    static auto sin(I const& x) -> I { return x.sin(); }
    static auto cos(I const& x) -> I { return x.cos(); }
    static auto tan(I const& x) -> I { return x.tan(); }
    static auto asin(I const& x) -> I { return x.asin(); }
    static auto acos(I const& x) -> I { return x.acos(); }
    static auto atan(I const& x) -> I { return x.atan(); }
    static auto sinh(I const& x) -> I { return x.sinh(); }
    static auto cosh(I const& x) -> I { return x.cosh(); }
    static auto tanh(I const& x) -> I { return x.tanh(); }
    static auto floor(I const& x) -> I { return x.floor(); }
    static auto ceil(I const& x) -> I { return x.ceil(); }

    // x^y with the same rule selection as the built-in IntervalEvaluator Pow rule: a point exponent goes through
    // the integer-power operation when it is integral (exact for negative bases, tight for the even/odd cases),
    // through the real power otherwise; a genuine exponent interval uses the general enclosure.
    static auto pow(I const& x, I const& y) -> I
    {
        if (y.inf() == y.sup()) {
            return x.pow(y.inf());
        }
        return x.pow(y);
    }

    // x * log(y), with the exponent partial convention 0 at x == 0.
    static auto xlogy(I const& x, I const& y) -> I
    {
        if (x.is_zero()) {
            return I::zero();
        }
        return x * y.log();
    }

    // Exponent contribution to the tangent of x^y. Over an entirely negative base and a point integral exponent
    // the value is real but the exponent partial does not exist: zero for a zero tangent, empty (the interval NaN)
    // otherwise.
    static auto pow_exponent_tangent(I const& r, I const& x, I const& y, I const& gv) -> I
    {
        if (x.sup() < T { 0 } && y.inf() == y.sup() && y.inf() == std::floor(y.inf())) {
            return gv.is_zero() ? I::zero() : I::empty();
        }
        return xlogy(r, x) * gv;
    }

    static auto fmin(I const& x, I const& y) -> I
    {
        if (x.is_empty() || y.is_empty()) {
            return I::empty();
        }
        return I { std::fmin(x.inf(), y.inf()), std::fmin(x.sup(), y.sup()) };
    }
    static auto fmax(I const& x, I const& y) -> I
    {
        if (x.is_empty() || y.is_empty()) {
            return I::empty();
        }
        return I { std::fmax(x.inf(), y.inf()), std::fmax(x.sup(), y.sup()) };
    }

    static auto min_weight(I const& x, I const& y) -> I { return Weight(x.sup() < y.inf(), x.inf() > y.sup(), x, y); }
    static auto max_weight(I const& x, I const& y) -> I { return Weight(x.inf() > y.sup(), x.sup() < y.inf(), x, y); }

private:
    // [1,1] where x is selected over the whole box, [0,0] where y is, the hull [0,1] otherwise.
    static auto Weight(bool selected, bool rejected, I const& x, I const& y) -> I
    {
        if (x.is_empty() || y.is_empty()) {
            return I::empty();
        }
        if (selected) {
            return I { T { 1 } };
        }
        if (rejected) {
            return I { T { 0 } };
        }
        return I { T { 0 }, T { 1 } };
    }
};

} // namespace Operon

#endif
