// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_JET_HPP
#define OPERON_CORE_JET_HPP

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "operon/core/contracts.hpp"

// jet.hpp: a small forward-mode automatic-differentiation number type.
//
// A first-order Jet<T, N> carries a primal value `a` and N tangent components `v`
// (the directional derivatives of the value with respect to N independent seed
// directions). A SecondDiagonal Jet additionally carries `w`, where w[k] is
// the pure second derivative along seed direction k. Mixed partial derivatives
// are not stored or independently exposed.
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

// Forward-mode jet. SecondDiagonal stores only pure second derivatives per direction.
enum class JetOrder : std::uint8_t { First, SecondDiagonal };
template <typename T, int N, JetOrder Order = JetOrder::First> struct Jet;

template <typename T, int N> struct Jet<T, N, JetOrder::First> {
    static_assert(N > 0, "A Jet needs at least one tangent component.");

    static constexpr JetOrder ORDER = JetOrder::First;

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

// The order-two specialization propagates only diagonal (pure) second derivatives.
template <typename T, int K> struct Jet<T, K, JetOrder::SecondDiagonal> {
    static_assert(K > 0, "A Jet needs at least one direction.");
    static constexpr int DIMENSION = K;
    static constexpr JetOrder ORDER = JetOrder::SecondDiagonal;
    using Scalar = T;

    T a;
    std::array<T, K> v;
    std::array<T, K> w;

    constexpr Jet() : a(T(0))
    {
        v.fill(T(0));
        w.fill(T(0));
    }
    constexpr explicit Jet(T value) : a(value)
    {
        v.fill(T(0));
        w.fill(T(0));
    }
    Jet(T value, int k) : a(value)
    {
        EXPECT(k >= 0 && k < K);
        v.fill(T(0));
        w.fill(T(0));
        v[static_cast<std::size_t>(k)] = T(1);
    }
    constexpr Jet(T value, std::array<T, K> const& first, std::array<T, K> const& second)
        : a(value)
        , v(first)
        , w(second)
    {
    }

    friend constexpr auto operator+(Jet const& f) -> Jet { return f; }
    friend constexpr auto operator-(Jet const& f) -> Jet
    {
        auto t = f.v;
        auto s = f.w;
        for (int k = 0; k < K; ++k) {
            t[k] = -t[k];
            s[k] = -s[k];
        }
        return { -f.a, t, s };
    }
    friend constexpr auto operator+(Jet const& f, Jet const& g) -> Jet
    {
        auto t = f.v;
        auto s = f.w;
        for (int k = 0; k < K; ++k) {
            t[k] += g.v[k];
            s[k] += g.w[k];
        }
        return { f.a + g.a, t, s };
    }
    friend constexpr auto operator+(Jet const& f, T const& x) -> Jet { return { f.a + x, f.v, f.w }; }
    friend constexpr auto operator+(T const& x, Jet const& f) -> Jet { return f + x; }
    friend constexpr auto operator-(Jet const& f, Jet const& g) -> Jet
    {
        auto t = f.v;
        auto s = f.w;
        for (int k = 0; k < K; ++k) {
            t[k] -= g.v[k];
            s[k] -= g.w[k];
        }
        return { f.a - g.a, t, s };
    }
    friend constexpr auto operator-(Jet const& f, T const& x) -> Jet { return { f.a - x, f.v, f.w }; }
    friend constexpr auto operator-(T const& x, Jet const& f) -> Jet { return x + -f; }
    friend constexpr auto operator*(Jet const& f, Jet const& g) -> Jet
    {
        std::array<T, K> t, s;
        for (int k = 0; k < K; ++k) {
            auto c = f.v[k] * g.v[k];
            t[k] = f.a * g.v[k] + f.v[k] * g.a;
            s[k] = f.w[k] * g.a + (c + c) + f.a * g.w[k];
        }
        return { f.a * g.a, t, s };
    }
    friend constexpr auto operator*(Jet const& f, T const& x) -> Jet
    {
        auto t = f.v;
        auto s = f.w;
        for (int k = 0; k < K; ++k) {
            t[k] *= x;
            s[k] *= x;
        }
        return { f.a * x, t, s };
    }
    friend constexpr auto operator*(T const& x, Jet const& f) -> Jet { return f * x; }
    friend constexpr auto operator/(Jet const& f, Jet const& g) -> Jet
    {
        auto inv = T(1) / g.a;
        auto q = f.a * inv;
        std::array<T, K> t, s;
        for (int k = 0; k < K; ++k) {
            t[k] = (f.v[k] - q * g.v[k]) * inv;
            auto c = t[k] * g.v[k];
            s[k] = (f.w[k] - (c + c) - q * g.w[k]) * inv;
        }
        return { q, t, s };
    }
    friend constexpr auto operator/(Jet const& f, T const& x) -> Jet { return f * (T(1) / x); }
    friend constexpr auto operator/(T const& x, Jet const& g) -> Jet { return Jet(x) / g; }
    constexpr auto operator+=(Jet const& x) -> Jet& { return *this = *this + x; }
    constexpr auto operator+=(T const& x) -> Jet& { return *this = *this + x; }
    constexpr auto operator-=(Jet const& x) -> Jet& { return *this = *this - x; }
    constexpr auto operator-=(T const& x) -> Jet& { return *this = *this - x; }
    constexpr auto operator*=(Jet const& x) -> Jet& { return *this = *this * x; }
    constexpr auto operator*=(T const& x) -> Jet& { return *this = *this * x; }
    constexpr auto operator/=(Jet const& x) -> Jet& { return *this = *this / x; }
    constexpr auto operator/=(T const& x) -> Jet& { return *this = *this / x; }
    friend constexpr auto operator==(Jet const& f, Jet const& g) { return f.a == g.a; }
    friend constexpr auto operator==(Jet const& f, T const& x) { return f.a == x; }
    friend constexpr auto operator==(T const& x, Jet const& f) { return x == f.a; }
    friend constexpr auto operator!=(Jet const& f, Jet const& g) { return f.a != g.a; }
    friend constexpr auto operator!=(Jet const& f, T const& x) { return f.a != x; }
    friend constexpr auto operator!=(T const& x, Jet const& f) { return x != f.a; }
    friend constexpr auto operator<(Jet const& f, Jet const& g) { return f.a < g.a; }
    friend constexpr auto operator<(Jet const& f, T const& x) { return f.a < x; }
    friend constexpr auto operator<(T const& x, Jet const& f) { return x < f.a; }
    friend constexpr auto operator<=(Jet const& f, Jet const& g) { return f.a <= g.a; }
    friend constexpr auto operator<=(Jet const& f, T const& x) { return f.a <= x; }
    friend constexpr auto operator<=(T const& x, Jet const& f) { return x <= f.a; }
    friend constexpr auto operator>(Jet const& f, Jet const& g) { return f.a > g.a; }
    friend constexpr auto operator>(Jet const& f, T const& x) { return f.a > x; }
    friend constexpr auto operator>(T const& x, Jet const& f) { return x > f.a; }
    friend constexpr auto operator>=(Jet const& f, Jet const& g) { return f.a >= g.a; }
    friend constexpr auto operator>=(Jet const& f, T const& x) { return f.a >= x; }
    friend constexpr auto operator>=(T const& x, Jet const& f) { return x >= f.a; }
};

namespace detail {
    template <typename T, int K>
    auto Chain2(T a, T d1, T d2, Jet<T, K, JetOrder::SecondDiagonal> const& f) -> Jet<T, K, JetOrder::SecondDiagonal>
    {
        std::array<T, K> v, w;
        for (int k = 0; k < K; ++k) {
            v[k] = d1 * f.v[k];
            w[k] = d2 * JetMath<T>::sqr(f.v[k]) + d1 * f.w[k];
        }
        return { a, v, w };
    }
    template <typename T, int K>
    auto Linear2(T a, T d, Jet<T, K, JetOrder::SecondDiagonal> const& f) -> Jet<T, K, JetOrder::SecondDiagonal>
    {
        return Chain2(a, d, T(0), f);
    }
    template <typename T, int K>
    auto Chain2(T a, T dx, T dy, T dxx, T dxy, T dyy, Jet<T, K, JetOrder::SecondDiagonal> const& f,
        Jet<T, K, JetOrder::SecondDiagonal> const& g) -> Jet<T, K, JetOrder::SecondDiagonal>
    {
        std::array<T, K> v, w;
        for (int k = 0; k < K; ++k) {
            v[k] = dx * f.v[k] + dy * g.v[k];
            auto c = dxy * f.v[k] * g.v[k];
            w[k] = dx * f.w[k] + dy * g.w[k] + dxx * JetMath<T>::sqr(f.v[k]) + (c + c) + dyy * JetMath<T>::sqr(g.v[k]);
        }
        return { a, v, w };
    }
    template <typename T, int K>
    auto Linear2(T a, T dx, T dy, Jet<T, K, JetOrder::SecondDiagonal> const& f,
        Jet<T, K, JetOrder::SecondDiagonal> const& g) -> Jet<T, K, JetOrder::SecondDiagonal>
    {
        return Chain2(a, dx, dy, T(0), T(0), T(0), f, g);
    }
    template <typename T> struct PowPartials2 {
        T d1;
        T d2;
    };
    template <typename T> auto PowConstantExponent2(T const& x, T const& c) -> PowPartials2<T>
    {
        using M = JetMath<T>;
        auto const exact = [&](auto e) -> std::optional<PowPartials2<T>> {
            if (e == 0) {
                return PowPartials2<T> { T(0), T(0) };
            }
            if (e == 1) {
                return PowPartials2<T> { T(1), T(0) };
            }
            if (e == 2) {
                return PowPartials2<T> { T(2) * x, T(2) };
            }
            return std::nullopt;
        };
        if constexpr (std::floating_point<T>) {
            if (auto e = exact(c)) {
                return *e;
            }
        } else if constexpr (requires {
                                 c.inf();
                                 c.sup();
                             }) {
            if (c.inf() == c.sup()) {
                if (auto e = exact(c.inf())) {
                    return *e;
                }
            }
        }
        return { c * M::pow(x, c - T(1)), c * (c - T(1)) * M::pow(x, c - T(2)) };
    }
    template <typename T> auto IsZero2(T const& value) -> bool
    {
        if constexpr (std::floating_point<T>) {
            return value == T(0);
        } else if constexpr (requires { value.is_zero(); }) {
            return value.is_zero();
        } else {
            return false;
        }
    }
}

template <typename T, int K>
auto abs(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Linear2(JetMath<T>::abs(x.a), JetMath<T>::sgn(x.a), x);
}
template <typename T, int K>
auto acos(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto u = T(1) - JetMath<T>::sqr(x.a);
    auto s = JetMath<T>::sqrt(u);
    return detail::Chain2(JetMath<T>::acos(x.a), -T(1) / s, -x.a / (u * s), x);
}
template <typename T, int K>
auto asin(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto u = T(1) - JetMath<T>::sqr(x.a);
    auto s = JetMath<T>::sqrt(u);
    return detail::Chain2(JetMath<T>::asin(x.a), T(1) / s, x.a / (u * s), x);
}
template <typename T, int K>
auto atan(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto d = T(1) / (T(1) + JetMath<T>::sqr(x.a));
    return detail::Chain2(JetMath<T>::atan(x.a), d, -T(2) * x.a * JetMath<T>::sqr(d), x);
}
template <typename T, int K>
auto cbrt(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto c = JetMath<T>::cbrt(x.a);
    return detail::Chain2(
        c, T(1) / (T(3) * JetMath<T>::sqr(c)), -T(2) / (T(9) * JetMath<T>::sqr(JetMath<T>::sqr(c)) * c), x);
}
template <typename T, int K>
auto ceil(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return Jet<T, K, JetOrder::SecondDiagonal>(JetMath<T>::ceil(x.a));
}
template <typename T, int K>
auto cos(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Chain2(JetMath<T>::cos(x.a), -JetMath<T>::sin(x.a), -JetMath<T>::cos(x.a), x);
}
template <typename T, int K>
auto cosh(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Chain2(JetMath<T>::cosh(x.a), JetMath<T>::sinh(x.a), JetMath<T>::cosh(x.a), x);
}
template <typename T, int K>
auto exp(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto e = JetMath<T>::exp(x.a);
    return detail::Chain2(e, e, e, x);
}
template <typename T, int K>
auto floor(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return Jet<T, K, JetOrder::SecondDiagonal>(JetMath<T>::floor(x.a));
}
template <typename T, int K>
auto log(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto d = T(1) / x.a;
    return detail::Chain2(JetMath<T>::log(x.a), d, -JetMath<T>::sqr(d), x);
}
template <typename T, int K>
auto log1p(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto d = T(1) / (T(1) + x.a);
    return detail::Chain2(JetMath<T>::log1p(x.a), d, -JetMath<T>::sqr(d), x);
}
template <typename T, int K>
auto sin(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Chain2(JetMath<T>::sin(x.a), JetMath<T>::cos(x.a), -JetMath<T>::sin(x.a), x);
}
template <typename T, int K>
auto sinh(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Chain2(JetMath<T>::sinh(x.a), JetMath<T>::cosh(x.a), JetMath<T>::sinh(x.a), x);
}
template <typename T, int K>
auto sqrt(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto s = JetMath<T>::sqrt(x.a);
    return detail::Chain2(s, T(1) / (T(2) * s), -T(1) / (T(4) * s * JetMath<T>::sqr(s)), x);
}
template <typename T, int K>
auto tan(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto t = JetMath<T>::tan(x.a);
    auto d = T(1) + JetMath<T>::sqr(t);
    return detail::Chain2(t, d, T(2) * t * d, x);
}
template <typename T, int K>
auto tanh(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto t = JetMath<T>::tanh(x.a);
    auto d = T(1) - JetMath<T>::sqr(t);
    return detail::Chain2(t, d, -T(2) * t * d, x);
}
template <typename T, int K>
auto square(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Chain2(JetMath<T>::sqr(x.a), T(2) * x.a, T(2), x);
}
template <typename T, int K>
auto logabs(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto a = JetMath<T>::abs(x.a);
    return detail::Chain2(JetMath<T>::log(a), JetMath<T>::sgn(x.a) / a, -JetMath<T>::sqr(T(1) / a), x);
}
template <typename T, int K>
auto sqrtabs(Jet<T, K, JetOrder::SecondDiagonal> const& x) -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto s = JetMath<T>::sqrt(JetMath<T>::abs(x.a));
    return detail::Chain2(s, JetMath<T>::sgn(x.a) / (T(2) * s), -T(1) / (T(4) * s * JetMath<T>::sqr(s)), x);
}
template <typename T, int K>
auto aq(Jet<T, K, JetOrder::SecondDiagonal> const& f, Jet<T, K, JetOrder::SecondDiagonal> const& g)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto d = T(1) + JetMath<T>::sqr(g.a);
    auto s = JetMath<T>::sqrt(d);
    auto q = f.a / s;
    return detail::Chain2(q, T(1) / s, -q * g.a / d, T(0), -g.a / (d * s),
        f.a * (T(2) * JetMath<T>::sqr(g.a) - T(1)) / (JetMath<T>::sqr(d) * s), f, g);
}
template <typename T, int K>
auto aq(Jet<T, K, JetOrder::SecondDiagonal> const& x, std::type_identity_t<T> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return aq(x, Jet<T, K, JetOrder::SecondDiagonal>(y));
}
template <typename T, int K>
auto aq(std::type_identity_t<T> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return aq(Jet<T, K, JetOrder::SecondDiagonal>(x), y);
}
template <typename T, int K>
auto pow(Jet<T, K, JetOrder::SecondDiagonal> const& x, std::type_identity_t<T> const& c)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto d = detail::PowConstantExponent2(x.a, c);
    return detail::Chain2(JetMath<T>::pow(x.a, c), d.d1, d.d2, x);
}
template <typename T, int K>
auto pow(Jet<T, K, JetOrder::SecondDiagonal> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto r = JetMath<T>::pow(x.a, y.a);
    auto p = pow(x, y.a);
    std::array<T, K> v, w;
    for (int k = 0; k < K; ++k) {
        auto ry = T(0);
        auto cross = T(0);
        auto yy = T(0);
        auto yw = T(0);
        if (!detail::IsZero2(y.v[k])) {
            ry = JetMath<T>::pow_exponent_tangent(r, x.a, y.a, y.v[k]);
            cross = x.v[k]
                * (JetMath<T>::pow(x.a, y.a - T(1)) * y.v[k]
                    + y.a * JetMath<T>::pow_exponent_tangent(JetMath<T>::pow(x.a, y.a - T(1)), x.a, y.a, y.v[k]));
            yy = JetMath<T>::pow_exponent_tangent(ry, x.a, y.a, y.v[k]);
        }
        if (!detail::IsZero2(y.w[k])) {
            yw = JetMath<T>::pow_exponent_tangent(r, x.a, y.a, y.w[k]);
        }
        v[k] = p.v[k] + ry;
        w[k] = p.w[k] + cross + cross + yy + yw;
    }
    return { r, v, w };
}
template <typename T, int K>
auto pow(std::type_identity_t<T> const& c, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto r = JetMath<T>::pow(c, y.a);
    std::array<T, K> v, w;
    for (int k = 0; k < K; ++k) {
        auto first = JetMath<T>::pow_exponent_tangent(r, c, y.a, y.v[k]);
        v[k] = first;
        w[k] = JetMath<T>::pow_exponent_tangent(first, c, y.a, y.v[k])
            + JetMath<T>::pow_exponent_tangent(r, c, y.a, y.w[k]);
    }
    return { r, v, w };
}
template <typename T, int K>
auto powabs(Jet<T, K, JetOrder::SecondDiagonal> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto a = JetMath<T>::abs(x.a);
    auto r = JetMath<T>::pow(a, y.a);
    auto p = JetMath<T>::pow(a, y.a - T(1));
    auto dy = JetMath<T>::xlogy(r, a);
    return detail::Chain2(r, y.a * p * JetMath<T>::sgn(x.a), dy, y.a * (y.a - T(1)) * JetMath<T>::pow(a, y.a - T(2)),
        JetMath<T>::sgn(x.a) * (p + y.a * JetMath<T>::xlogy(p, a)), JetMath<T>::xlogy(dy, a), x, y);
}
template <typename T, int K>
auto powabs(Jet<T, K, JetOrder::SecondDiagonal> const& x, std::type_identity_t<T> const& c)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto a = JetMath<T>::abs(x.a);
    auto d = detail::PowConstantExponent2(a, c);
    return detail::Chain2(JetMath<T>::pow(a, c), d.d1 * JetMath<T>::sgn(x.a), d.d2, x);
}
template <typename T, int K>
auto powabs(std::type_identity_t<T> const& c, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    auto r = JetMath<T>::pow(JetMath<T>::abs(c), y.a);
    auto d = JetMath<T>::xlogy(r, JetMath<T>::abs(c));
    return detail::Chain2(r, d, JetMath<T>::xlogy(d, JetMath<T>::abs(c)), y);
}
template <typename T, int K>
auto fmin(Jet<T, K, JetOrder::SecondDiagonal> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Linear2(
        JetMath<T>::fmin(x.a, y.a), JetMath<T>::min_weight(x.a, y.a), JetMath<T>::min_weight(y.a, x.a), x, y);
}
template <typename T, int K>
auto fmax(Jet<T, K, JetOrder::SecondDiagonal> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return detail::Linear2(
        JetMath<T>::fmax(x.a, y.a), JetMath<T>::max_weight(x.a, y.a), JetMath<T>::max_weight(y.a, x.a), x, y);
}
template <typename T, int K>
auto fmin(Jet<T, K, JetOrder::SecondDiagonal> const& x, std::type_identity_t<T> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return fmin(x, Jet<T, K, JetOrder::SecondDiagonal>(y));
}
template <typename T, int K>
auto fmin(std::type_identity_t<T> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return fmin(Jet<T, K, JetOrder::SecondDiagonal>(x), y);
}
template <typename T, int K>
auto fmax(Jet<T, K, JetOrder::SecondDiagonal> const& x, std::type_identity_t<T> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return fmax(x, Jet<T, K, JetOrder::SecondDiagonal>(y));
}
template <typename T, int K>
auto fmax(std::type_identity_t<T> const& x, Jet<T, K, JetOrder::SecondDiagonal> const& y)
    -> Jet<T, K, JetOrder::SecondDiagonal>
{
    return fmax(Jet<T, K, JetOrder::SecondDiagonal>(x), y);
}

} // namespace Operon

#endif
