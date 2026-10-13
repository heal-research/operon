// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_JET_HPP
#define OPERON_CORE_JET_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "operon/core/contracts.hpp"

// jet.hpp: a small forward-mode automatic-differentiation number type.
//
// A Jet<T, N> carries a primal value `a` and N tangent components `v`
// (the directional derivatives of the value with respect to N independent
// seed directions). Arithmetic and the elementary functions below propagate
// both by the chain rule.
//
// The function set is exactly Operon's built-in operator set (see BuiltinOp):
//   abs acos asin atan cbrt ceil cos cosh exp floor log log1p sin sinh sqrt
//   tan tanh, plus square logabs sqrtabs aq powabs pow fmin fmax.
// The tangent conventions at non-smooth points are those of the hand-written
// derivative backends (interpreter/backend/plain/derivatives.hpp):
//   - d abs(x)/dx = sgn(x), with sgn(0) = 0;
//   - ceil and floor have a zero tangent;
//   - d logabs(x)/dx = sgn(x)/|x| and d sqrtabs(x)/dx = sgn(x)/(2 sqrt|x|),
//     both NaN at x = 0 (0/0);
//   - d powabs(x, y)/dx carries the factor sgn(x), so it is 0 at x = 0
//     wherever |x|^(y-1) is finite; the exponent partials of pow/powabs
//     (|x|^y log|x|) are taken as 0 where |x|^y == 0, instead of 0 * -inf;
//   - fmin/fmax tangent weights are 1 for the selected operand and 0 for the
//     other, NaN for both operands on a tie (or when both primals are NaN),
//     and a NaN primal selects the other operand (std::fmin/std::fmax).
//
// All functions live in namespace Operon next to Jet so that unqualified calls
// in user lambdas (`using std::sin; sin(v)`) resolve to them through ADL.
//
// The elementary primal math is not hardcoded to <cmath>: every function is
// routed through the JetMath<T> customization point. This header provides
// JetMath for the built-in floating-point types; other scalar kinds (SIMD
// lanes, intervals, ...) specialize JetMath from the layer that owns their
// dependency. This header depends on the standard library and libassert only.

namespace Operon {

// Customization point for the scalar math underlying Jet<T, N>.
//
// A specialization for scalar kind T must provide the following static
// functions, each taking and returning T (T need not be a plain arithmetic
// type, but must support +, -, *, / and construction from an int literal,
// i.e. T(0), T(1), T(2), T(3)):
//
//   abs sgn sqrt cbrt exp log log1p sin cos tan asin acos atan sinh cosh
//   tanh floor ceil               -- unary, the usual meaning; sgn(x) is -1, 0
//                                    or +1 (0 at x == 0 and for NaN)
//   sqr(x)                        -- x * x. A separate hook because a scalar
//                                    kind with dependency-blind arithmetic
//                                    (intervals) can enclose x^2 much more
//                                    tightly than x * x; every derivative
//                                    formula below that squares a primal uses it.
//   xlogy(x, y)                   -- x * log(y), defined as 0 where x == 0
//                                    (the exponent partial of x^y at x == 0)
//   pow_exponent_tangent(r, x, y, gv)
//                                 -- the exponent's contribution to the tangent of r = x^y when y carries tangent
//                                    component gv: xlogy(r, x) * gv, except where x < 0 and y is integral (a real
//                                    value whose exponent partial does not exist): 0 if gv == 0, NaN otherwise.
//   fmin(x, y) fmax(x, y)         -- std::fmin/std::fmax semantics (a NaN
//                                    operand selects the other one)
//   min_weight(x, y)              -- d fmin(x, y)/dx: 1 if x is selected, 0 if
//   max_weight(x, y)                 y is selected, NaN if x == y or both are
//                                    NaN. max_weight is the same for fmax. The
//                                    weight for y is min_weight(y, x).
template <typename T> struct JetMath;

template <std::floating_point T> struct JetMath<T> {
    static auto sqr(T x) -> T { return x * x; }
    static auto abs(T x) -> T { return std::abs(x); }
    static auto sgn(T x) -> T { return static_cast<T>(static_cast<int>(T { 0 } < x) - static_cast<int>(x < T { 0 })); }
    static auto sqrt(T x) -> T { return std::sqrt(x); }
    static auto cbrt(T x) -> T { return std::cbrt(x); }
    static auto exp(T x) -> T { return std::exp(x); }
    static auto log(T x) -> T { return std::log(x); }
    static auto log1p(T x) -> T { return std::log1p(x); }
    static auto sin(T x) -> T { return std::sin(x); }
    static auto cos(T x) -> T { return std::cos(x); }
    static auto tan(T x) -> T { return std::tan(x); }
    static auto asin(T x) -> T { return std::asin(x); }
    static auto acos(T x) -> T { return std::acos(x); }
    static auto atan(T x) -> T { return std::atan(x); }
    static auto sinh(T x) -> T { return std::sinh(x); }
    static auto cosh(T x) -> T { return std::cosh(x); }
    static auto tanh(T x) -> T { return std::tanh(x); }
    static auto floor(T x) -> T { return std::floor(x); }
    static auto ceil(T x) -> T { return std::ceil(x); }
    static auto pow(T x, T y) -> T { return std::pow(x, y); }
    static auto xlogy(T x, T y) -> T { return x == T { 0 } ? T { 0 } : x * std::log(y); }
    static auto pow_exponent_tangent(T r, T x, T y, T gv) -> T
    {
        if (x < T { 0 } && y == std::floor(y)) {
            return gv == T { 0 } ? T { 0 } : std::numeric_limits<T>::quiet_NaN();
        }
        return xlogy(r, x) * gv;
    }
    static auto fmin(T x, T y) -> T { return std::fmin(x, y); }
    static auto fmax(T x, T y) -> T { return std::fmax(x, y); }
    static auto min_weight(T x, T y) -> T { return Weight(x, y, x < y); }
    static auto max_weight(T x, T y) -> T { return Weight(x, y, x > y); }

private:
    static auto Weight(T x, T y, bool selected) -> T
    {
        auto const nanX = std::isnan(x);
        auto const nanY = std::isnan(y);
        if ((nanX && nanY) || x == y) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        if (nanX) {
            return T { 0 };
        }
        if (nanY) {
            return T { 1 };
        }
        return selected ? T { 1 } : T { 0 };
    }
};

// Forward-mode dual number: primal value `a` and N tangent components `v`.
template <typename T, int N> struct Jet {
    static_assert(N > 0, "A Jet needs at least one tangent component.");

    static constexpr int DIMENSION = N;
    using Scalar = T;

    T a;
    std::array<T, N> v;

    // zero value, zero tangent
    constexpr Jet() : a(T(0)) { v.fill(T(0)); }

    // constant: the given value with a zero tangent
    constexpr explicit Jet(T value) : a(value) { v.fill(T(0)); }

    // independent variable: the given value with tangent component k seeded to 1
    Jet(T value, int k) : a(value)
    {
        EXPECT(k >= 0 && k < N);
        v.fill(T(0));
        v[static_cast<std::size_t>(k)] = T(1);
    }

    // explicit primal value and tangent
    constexpr Jet(T value, std::array<T, N> const& tangent) : a(value), v(tangent) {}

    // ---- unary +/- ----
    friend constexpr auto operator+(Jet const& f) -> Jet { return f; }

    friend constexpr auto operator-(Jet const& f) -> Jet
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = -f.v[k];
        }
        return { -f.a, t };
    }

    // ---- addition ----
    friend constexpr auto operator+(Jet const& f, Jet const& g) -> Jet
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = f.v[k] + g.v[k];
        }
        return { f.a + g.a, t };
    }

    friend constexpr auto operator+(Jet const& f, T const& s) -> Jet { return { f.a + s, f.v }; }

    friend constexpr auto operator+(T const& s, Jet const& g) -> Jet { return { s + g.a, g.v }; }

    // ---- subtraction ----
    friend constexpr auto operator-(Jet const& f, Jet const& g) -> Jet
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = f.v[k] - g.v[k];
        }
        return { f.a - g.a, t };
    }

    friend constexpr auto operator-(Jet const& f, T const& s) -> Jet { return { f.a - s, f.v }; }

    friend constexpr auto operator-(T const& s, Jet const& g) -> Jet
    {
        auto r = -g;
        r.a += s;
        return r;
    }

    // ---- multiplication ----
    friend constexpr auto operator*(Jet const& f, Jet const& g) -> Jet
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = f.a * g.v[k] + f.v[k] * g.a;
        }
        return { f.a * g.a, t };
    }

    friend constexpr auto operator*(Jet const& f, T const& s) -> Jet
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = f.v[k] * s;
        }
        return { f.a * s, t };
    }

    friend constexpr auto operator*(T const& s, Jet const& g) -> Jet { return g * s; }

    // ---- division ----
    friend constexpr auto operator/(Jet const& f, Jet const& g) -> Jet
    {
        auto const inv = T(1) / g.a;
        auto const q = f.a * inv;
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = (f.v[k] - q * g.v[k]) * inv;
        }
        return { q, t };
    }

    friend constexpr auto operator/(Jet const& f, T const& s) -> Jet
    {
        auto const inv = T(1) / s;
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = f.v[k] * inv;
        }
        return { f.a * inv, t };
    }

    friend constexpr auto operator/(T const& s, Jet const& g) -> Jet
    {
        auto const inv = T(1) / g.a;
        auto const q = s * inv;
        auto const d = -q * inv;
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = d * g.v[k];
        }
        return { q, t };
    }

    // ---- compound assignment ----
    constexpr auto operator+=(Jet const& g) -> Jet& { return *this = *this + g; }
    constexpr auto operator+=(T const& s) -> Jet&
    {
        a += s;
        return *this;
    }
    constexpr auto operator-=(Jet const& g) -> Jet& { return *this = *this - g; }
    constexpr auto operator-=(T const& s) -> Jet&
    {
        a -= s;
        return *this;
    }
    constexpr auto operator*=(Jet const& g) -> Jet& { return *this = *this * g; }
    constexpr auto operator*=(T const& s) -> Jet& { return *this = *this * s; }
    constexpr auto operator/=(Jet const& g) -> Jet& { return *this = *this / g; }
    constexpr auto operator/=(T const& s) -> Jet& { return *this = *this / s; }

    // ---- comparisons: primal part only (tangents are ignored) ----
    friend constexpr auto operator==(Jet const& f, Jet const& g) { return f.a == g.a; }
    friend constexpr auto operator==(Jet const& f, T const& s) { return f.a == s; }
    friend constexpr auto operator==(T const& s, Jet const& g) { return s == g.a; }
    friend constexpr auto operator!=(Jet const& f, Jet const& g) { return f.a != g.a; }
    friend constexpr auto operator!=(Jet const& f, T const& s) { return f.a != s; }
    friend constexpr auto operator!=(T const& s, Jet const& g) { return s != g.a; }
    friend constexpr auto operator<(Jet const& f, Jet const& g) { return f.a < g.a; }
    friend constexpr auto operator<(Jet const& f, T const& s) { return f.a < s; }
    friend constexpr auto operator<(T const& s, Jet const& g) { return s < g.a; }
    friend constexpr auto operator<=(Jet const& f, Jet const& g) { return f.a <= g.a; }
    friend constexpr auto operator<=(Jet const& f, T const& s) { return f.a <= s; }
    friend constexpr auto operator<=(T const& s, Jet const& g) { return s <= g.a; }
    friend constexpr auto operator>(Jet const& f, Jet const& g) { return f.a > g.a; }
    friend constexpr auto operator>(Jet const& f, T const& s) { return f.a > s; }
    friend constexpr auto operator>(T const& s, Jet const& g) { return s > g.a; }
    friend constexpr auto operator>=(Jet const& f, Jet const& g) { return f.a >= g.a; }
    friend constexpr auto operator>=(Jet const& f, T const& s) { return f.a >= s; }
    friend constexpr auto operator>=(T const& s, Jet const& g) { return s >= g.a; }
};

namespace detail {
    // f(x) with derivative d = f'(x): the tangent is d times the argument's
    template <typename T, int N> auto Chain(T value, T d, Jet<T, N> const& f) -> Jet<T, N>
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = d * f.v[k];
        }
        return { value, t };
    }

    // f(x) with a zero derivative: the tangent is zero
    template <typename T, int N> auto Constant(T value) -> Jet<T, N> { return Jet<T, N>(value); }

    // g(x, y) with partials dx, dy: the tangent is dx*f.v + dy*g.v
    template <typename T, int N> auto Chain(T value, T dx, Jet<T, N> const& f, T dy, Jet<T, N> const& g) -> Jet<T, N>
    {
        std::array<T, N> t;
        for (std::size_t k = 0; k < t.size(); ++k) {
            t[k] = dx * f.v[k] + dy * g.v[k];
        }
        return { value, t };
    }
} // namespace detail

// ---- unary functions ----

template <typename T, int N> auto abs(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::abs(f.a), M::sgn(f.a), f);
}

template <typename T, int N> auto acos(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::acos(f.a), -T(1) / M::sqrt(T(1) - M::sqr(f.a)), f);
}

template <typename T, int N> auto asin(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::asin(f.a), T(1) / M::sqrt(T(1) - M::sqr(f.a)), f);
}

template <typename T, int N> auto atan(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::atan(f.a), T(1) / (T(1) + M::sqr(f.a)), f);
}

template <typename T, int N> auto cbrt(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const c = M::cbrt(f.a);
    return detail::Chain(c, T(1) / (T(3) * M::sqr(c)), f);
}

template <typename T, int N> auto ceil(Jet<T, N> const& f) -> Jet<T, N>
{
    return detail::Constant<T, N>(JetMath<T>::ceil(f.a));
}

template <typename T, int N> auto cos(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::cos(f.a), -M::sin(f.a), f);
}

template <typename T, int N> auto cosh(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::cosh(f.a), M::sinh(f.a), f);
}

template <typename T, int N> auto exp(Jet<T, N> const& f) -> Jet<T, N>
{
    auto const e = JetMath<T>::exp(f.a);
    return detail::Chain(e, e, f);
}

template <typename T, int N> auto floor(Jet<T, N> const& f) -> Jet<T, N>
{
    return detail::Constant<T, N>(JetMath<T>::floor(f.a));
}

template <typename T, int N> auto log(Jet<T, N> const& f) -> Jet<T, N>
{
    return detail::Chain(JetMath<T>::log(f.a), T(1) / f.a, f);
}

template <typename T, int N> auto log1p(Jet<T, N> const& f) -> Jet<T, N>
{
    return detail::Chain(JetMath<T>::log1p(f.a), T(1) / (T(1) + f.a), f);
}

template <typename T, int N> auto sin(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::sin(f.a), M::cos(f.a), f);
}

template <typename T, int N> auto sinh(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::sinh(f.a), M::cosh(f.a), f);
}

template <typename T, int N> auto sqrt(Jet<T, N> const& f) -> Jet<T, N>
{
    auto const s = JetMath<T>::sqrt(f.a);
    return detail::Chain(s, T(1) / (T(2) * s), f);
}

template <typename T, int N> auto tan(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const t = M::tan(f.a);
    return detail::Chain(t, T(1) + M::sqr(t), f);
}

template <typename T, int N> auto tanh(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const t = M::tanh(f.a);
    return detail::Chain(t, T(1) - M::sqr(t), f);
}

// x * x
template <typename T, int N> auto square(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::sqr(f.a), T(2) * f.a, f);
}

// log(|x|)
template <typename T, int N> auto logabs(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const ax = M::abs(f.a);
    return detail::Chain(M::log(ax), M::sgn(f.a) / ax, f);
}

// sqrt(|x|)
template <typename T, int N> auto sqrtabs(Jet<T, N> const& f) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const s = M::sqrt(M::abs(f.a));
    return detail::Chain(s, M::sgn(f.a) / (T(2) * s), f);
}

// ---- binary functions ----

// analytic quotient: x / sqrt(1 + y^2)
template <typename T, int N> auto aq(Jet<T, N> const& f, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const d = T(1) + M::sqr(g.a);
    auto const s = M::sqrt(d);
    auto const q = f.a / s;
    return detail::Chain(q, T(1) / s, f, -q * g.a / d, g);
}

template <typename T, int N> auto aq(Jet<T, N> const& f, std::type_identity_t<T> const& s) -> Jet<T, N>
{
    return aq(f, Jet<T, N>(s));
}

template <typename T, int N> auto aq(std::type_identity_t<T> const& s, Jet<T, N> const& g) -> Jet<T, N>
{
    return aq(Jet<T, N>(s), g);
}

// x^y with both operands carrying tangents. For a negative base and an integral exponent the value is real and
// d/dx is g * x^(g-1); the exponent partial does not exist there (any change in the exponent leaves the real
// line), so a nonzero exponent tangent yields NaN while a zero one contributes nothing (JetMath::pow_exponent_tangent).
template <typename T, int N> auto pow(Jet<T, N> const& f, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const r = M::pow(f.a, g.a);
    auto const dx = g.a * M::pow(f.a, g.a - T(1));
    std::array<T, N> t;
    for (std::size_t k = 0; k < t.size(); ++k) {
        t[k] = dx * f.v[k] + M::pow_exponent_tangent(r, f.a, g.a, g.v[k]);
    }
    return { r, t };
}

// x^c with a constant exponent
template <typename T, int N> auto pow(Jet<T, N> const& f, std::type_identity_t<T> const& c) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::pow(f.a, c), c * M::pow(f.a, c - T(1)), f);
}

// c^y with a constant base
template <typename T, int N> auto pow(std::type_identity_t<T> const& c, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const r = M::pow(c, g.a);
    std::array<T, N> t;
    for (std::size_t k = 0; k < t.size(); ++k) {
        t[k] = M::pow_exponent_tangent(r, c, g.a, g.v[k]);
    }
    return { r, t };
}

// |x|^y
template <typename T, int N> auto powabs(Jet<T, N> const& f, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const ax = M::abs(f.a);
    auto const r = M::pow(ax, g.a);
    return detail::Chain(r, g.a * M::pow(ax, g.a - T(1)) * M::sgn(f.a), f, M::xlogy(r, ax), g);
}

template <typename T, int N> auto powabs(Jet<T, N> const& f, std::type_identity_t<T> const& c) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const ax = M::abs(f.a);
    return detail::Chain(M::pow(ax, c), c * M::pow(ax, c - T(1)) * M::sgn(f.a), f);
}

template <typename T, int N> auto powabs(std::type_identity_t<T> const& c, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    auto const r = M::pow(M::abs(c), g.a);
    return detail::Chain(r, M::xlogy(r, M::abs(c)), g);
}

template <typename T, int N> auto fmin(Jet<T, N> const& f, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::fmin(f.a, g.a), M::min_weight(f.a, g.a), f, M::min_weight(g.a, f.a), g);
}

template <typename T, int N> auto fmin(Jet<T, N> const& f, std::type_identity_t<T> const& s) -> Jet<T, N>
{
    return fmin(f, Jet<T, N>(s));
}

template <typename T, int N> auto fmin(std::type_identity_t<T> const& s, Jet<T, N> const& g) -> Jet<T, N>
{
    return fmin(Jet<T, N>(s), g);
}

template <typename T, int N> auto fmax(Jet<T, N> const& f, Jet<T, N> const& g) -> Jet<T, N>
{
    using M = JetMath<T>;
    return detail::Chain(M::fmax(f.a, g.a), M::max_weight(f.a, g.a), f, M::max_weight(g.a, f.a), g);
}

template <typename T, int N> auto fmax(Jet<T, N> const& f, std::type_identity_t<T> const& s) -> Jet<T, N>
{
    return fmax(f, Jet<T, N>(s));
}

template <typename T, int N> auto fmax(std::type_identity_t<T> const& s, Jet<T, N> const& g) -> Jet<T, N>
{
    return fmax(Jet<T, N>(s), g);
}

} // namespace Operon

#endif
