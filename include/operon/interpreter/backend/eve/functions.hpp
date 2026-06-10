// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EVE_FUNCTIONS_HPP
#define OPERON_BACKEND_EVE_FUNCTIONS_HPP

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include "operon/core/dispatch.hpp"

namespace Operon::Backend::detail {

// FastExp: Cephes-style float polynomial, ~1-2 ULP. Ported from Eigen pexp_float.
// Double falls back to eve::exp (full accuracy).
template<std::floating_point T>
auto FastExp(eve::wide<T> a) -> eve::wide<T> {
    using W = eve::wide<T>;
    if constexpr (std::is_same_v<T, float>) {
        auto x  = eve::clamp(a, T(-88.723f), T(88.723f));
        auto m  = eve::floor(eve::fma(x, W{T(1.44269504088896341f)}, W{T(0.5f)}));
        auto r  = eve::fma(m, W{T(-0.693359375f)},    x);
        r       = eve::fma(m, W{T(2.12194440e-4f)},   r);
        auto r2 = r * r;
        auto r3 = r2 * r;
        auto y  = eve::fma(W{T(1.9875691500E-4f)}, r, W{T(1.3981999507E-3f)});
        auto y1 = eve::fma(W{T(4.1665795894E-2f)}, r, W{T(1.6666665459E-1f)});
        auto y2 = r + W{T(1.0f)};
        y       = eve::fma(y,  r,  W{T(8.3334519073E-3f)});
        y1      = eve::fma(y1, r,  W{T(5.0000001201E-1f)});
        y       = eve::fma(y,  r3, y1);
        y       = eve::fma(y,  r2, y2);
        return eve::max(eve::ldexp(y, eve::convert(m, eve::as<std::int32_t>{})), a);
    } else {
        return eve::exp(a);
    }
}

// FastLog: Cephes-style float polynomial, ~1-2 ULP. Ported from Eigen plog_impl_float.
// Double falls back to eve::log (full accuracy).
template<std::floating_point T>
auto FastLog(eve::wide<T> a) -> eve::wide<T> {
    using W = eve::wide<T>;
    if constexpr (std::is_same_v<T, float>) {
        constexpr float min_norm = std::bit_cast<float>(0x00800000u);
        auto x = eve::max(a, W{min_norm});

        // Extract significand in [0.5, 1) and exponent
        auto [mant, e] = eve::frexp(x);
        x = mant;

        // Shift [0.5, 1) to [√½, √2): if x < SQRTHF → e-=1, x = 2x-1; else x = x-1
        auto mask = x < W{0.707106781186547524f};
        x = eve::fma(eve::if_else(mask, x, W{0.0f}), W{1.0f}, x - W{1.0f});
        e = e - eve::if_else(mask, W{1.0f}, W{0.0f});

        auto x2 = x * x;
        auto x3 = x2 * x;

        // 8-degree Cephes polynomial in three parallel chains
        auto y  = eve::fma(W{7.0376836292E-2f},   x, W{-1.1514610310E-1f});
        auto y1 = eve::fma(W{-1.2420140846E-1f},  x, W{+1.4249322787E-1f});
        auto y2 = eve::fma(W{+2.0000714765E-1f},  x, W{-2.4999993993E-1f});
        y       = eve::fma(y,  x, W{1.1676998740E-1f});
        y1      = eve::fma(y1, x, W{-1.6668057665E-1f});
        y2      = eve::fma(y2, x, W{+3.3333331174E-1f});
        y       = eve::fma(y,  x3, y1);
        y       = eve::fma(y,  x3, y2);
        y       = y * x3;

        y = eve::fma(W{-0.5f}, x2, y);
        x = eve::fma(e, W{0.693147180559945309f}, x + y);  // + e * ln2

        // Edge cases: negative/NaN → NaN, zero → -inf, +inf → +inf
        auto neg_nan = eve::nan(eve::as<W>{});
        x = eve::if_else(a < W{0.0f}, neg_nan, x);
        x = eve::if_else(eve::is_nan(a), neg_nan, x);
        x = eve::if_else(a == W{0.0f}, W{-std::numeric_limits<float>::infinity()}, x);
        x = eve::if_else(a == eve::inf(eve::as<W>{}), a, x);
        return x;
    } else {
        return eve::log(a);
    }
}

// FastSin/FastCos: Eigen-style psincos_float polynomial, ~1-2 ULP.
// Uses 3-part FMA range reduction (Eigen's __FMA__ path):
//   sin: valid for |x| < 117435.992, cos: valid for |x| < 71476.0625.
// Falls back to eve::sin/cos beyond those limits. Double uses eve::sin/cos directly.
template<bool IsSin, std::floating_point T>
auto FastSinCos(eve::wide<T> a) -> eve::wide<T> {
    using W  = eve::wide<T>;
    using WI = eve::wide<std::int32_t>;
    if constexpr (std::is_same_v<T, float>) {
        // Per Eigen psincos_float __FMA__ path
        constexpr float LargeThreshold = IsSin ? 117435.992f : 71476.0625f;
        auto large = eve::abs(a) >= W{LargeThreshold};
        auto x     = eve::abs(a);

        // Scale by 2/π and round to nearest integer octant
        auto y       = x * W{0.636619746685028076171875f};  // x * 2/π
        auto y_round = y + W{12582912.0f};                  // rounding magic (2^23 + 2^22)
        auto y_int   = eve::bit_cast(y_round, eve::as<WI>{});
        y            = y_round - W{12582912.0f};

        // FMA 3-part range-reduce x to [-π/4, π/4]
        x = eve::fma(y, W{-1.57079601287841796875f},                          x);
        x = eve::fma(y, W{-3.1391647326017846353352069854736328125E-7f},      x);
        x = eve::fma(y, W{-5.3903025299577647655446810404100688174E-15f},     x);

        // Compute sign bit: sin uses input sign XOR bit-1(octant); cos uses bit-1(octant+1)
        WI sign_i;
        if constexpr (IsSin) {
            sign_i = eve::bit_cast(a, eve::as<WI>{}) ^ (y_int << 30);
        } else {
            sign_i = (y_int + WI{1}) << 30;
        }
        auto sign_bit = eve::bit_cast(sign_i, eve::as<W>{}) & W{-0.0f};

        // Polynomial selection: even octant → use sin kernel (y2) for sin, cos kernel (y1) for cos
        auto poly_even = (y_int & WI{1}) == WI{0};

        auto x2 = x * x;

        // Cos kernel (even powers)
        auto y1 = W{2.4372266125283204E-5f};
        y1 = eve::fma(y1, x2, W{-1.38865201734006405E-3f});
        y1 = eve::fma(y1, x2, W{0.041666619479656219482421875f});
        y1 = eve::fma(y1, x2, W{-0.5f});
        y1 = eve::fma(y1, x2, W{1.0f});

        // Sin kernel (odd)
        auto y2 = W{-1.9592341140837029E-4f};
        y2 = eve::fma(y2, x2, W{8.33268736556168517E-3f});
        y2 = eve::fma(y2, x2, W{-1.66666620398229826E-1f});
        y2 = y2 * x2;
        y2 = eve::fma(y2, x, x);

        W result;
        if constexpr (IsSin) {
            result = eve::if_else(poly_even, y2, y1);
        } else {
            result = eve::if_else(poly_even, y1, y2);
        }
        result = eve::bit_xor(result, sign_bit);

        if (eve::any(large)) {
            if constexpr (IsSin) { return eve::if_else(large, eve::sin(a), result); }
            else                  { return eve::if_else(large, eve::cos(a), result); }
        }
        return result;
    } else {
        if constexpr (IsSin) return eve::sin(a);
        else return eve::cos(a);
    }
}

template<std::floating_point T>
auto FastTanh(eve::wide<T> a) -> eve::wide<T> {
    using W = eve::wide<T>;
    if constexpr (std::is_same_v<T, float>) {
        // 13/6-degree rational minimax, ~2 ULP on [-8,8].
        // Eigen's generic_fast_tanh_float (Pedro Gonnet, 2014).
        auto x    = eve::clamp(a, T(-7.99881172180175781f), T(7.99881172180175781f));
        auto tiny = eve::abs(a) < T(0.0004f);
        auto x2   = x * x;
        auto p = eve::fma(x2, W{T(-2.76076847742355e-16f)}, W{T( 2.00018790482477e-13f)});
        p = eve::fma(x2, p, W{T(-8.60467152213735e-11f)});
        p = eve::fma(x2, p, W{T( 5.12229709037114e-08f)});
        p = eve::fma(x2, p, W{T( 1.48572235717979e-05f)});
        p = eve::fma(x2, p, W{T( 6.37261928875436e-04f)});
        p = eve::fma(x2, p, W{T( 4.89352455891786e-03f)});
        p = x * p;
        auto q = eve::fma(x2, W{T(1.19825839466702e-06f)}, W{T(1.18534705686654e-04f)});
        q = eve::fma(x2, q, W{T(2.26843463243900e-03f)});
        q = eve::fma(x2, q, W{T(4.89352518554385e-03f)});
        return eve::if_else(tiny, x, p / q);
    } else {
        // 19/18-degree rational minimax, ~2 ULP on [-18.7,18.7].
        // Ported from Eigen's ptanh_double (rminimax-optimised coefficients).
        auto x    = eve::clamp(a, T(-17.6610191624600077), T(17.6610191624600077));
        auto tiny = eve::abs(a) < T(0.0004);
        auto x2   = x * x;
        auto p = eve::fma(x2, W{T(2.6158007860482230e-23)}, W{T(7.6534862268749319e-19)});
        p = eve::fma(x2, p, W{T(3.1309488231386680e-15)});
        p = eve::fma(x2, p, W{T(4.2303918148209176e-12)});
        p = eve::fma(x2, p, W{T(2.4618379131293676e-09)});
        p = eve::fma(x2, p, W{T(6.8644367682497074e-07)});
        p = eve::fma(x2, p, W{T(9.3839087674268880e-05)});
        p = eve::fma(x2, p, W{T(5.9809711724441161e-03)});
        p = eve::fma(x2, p, W{T(1.5184719640284322e-01)});
        p = x * p;
        auto q = eve::fma(x2, W{T(6.463747022670968018e-21)}, W{T(5.782506856739003571e-17)});
        q = eve::fma(x2, q, W{T(1.293019623712687916e-13)});
        q = eve::fma(x2, q, W{T(1.123643448069621992e-10)});
        q = eve::fma(x2, q, W{T(4.492975677839633985e-08)});
        q = eve::fma(x2, q, W{T(8.785185266237658698e-06)});
        q = eve::fma(x2, q, W{T(8.295161192716231542e-04)});
        q = eve::fma(x2, q, W{T(3.437448108450402717e-02)});
        q = eve::fma(x2, q, W{T(4.851805297361760360e-01)});
        q = eve::fma(x2, q, W{T(1.0)});
        return eve::if_else(tiny, x, p / q);
    }
}

// FastPow: exp(y * log|x|) using FastExp+FastLog, ~2 ULP.
// Sign rule: negate when x<0 and y is a finite odd integer.
// Matches SLEEF xfastpowf_u3500 structure. Double falls back to eve::pow.
template<std::floating_point T>
auto FastPow(eve::wide<T> x, eve::wide<T> y) -> eve::wide<T> {
    using W = eve::wide<T>;
    if constexpr (std::is_same_v<T, float>) {
        auto z = FastExp(y * FastLog(eve::abs(x)));
        z = eve::if_else(eve::is_ltz(x) && eve::is_odd(y), -z, z);
        z = eve::if_else(y == W{0}, W{1}, z);
        return z;
    } else {
        return eve::pow(x, y);
    }
}

// FastPowabs: exp(y * log|x|) — same as FastPow but skips sign correction.
template<std::floating_point T>
auto FastPowabs(eve::wide<T> x, eve::wide<T> y) -> eve::wide<T> {
    using W = eve::wide<T>;
    if constexpr (std::is_same_v<T, float>) {
        auto z = FastExp(y * FastLog(eve::abs(x)));
        return eve::if_else(y == W{0}, W{1}, z);
    } else {
        return eve::pow(eve::abs(x), y);
    }
}

} // namespace Operon::Backend::detail

namespace Operon::Backend {
    // utility
    template<typename T, std::size_t S>
    auto Fill(T* res, T value) {
        std::ranges::fill_n(res, S, value);
    }

    template<typename T, std::size_t S>
    auto Fill(T* res, int n, T value) {
        std::ranges::fill_n(res, n, value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Add(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * (W{args+i} + ...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Mul(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * (W{args+i} * ...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sub(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sub(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Div(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::div(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Min(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::min(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Max(T* res, T weight, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::max(W{args+i}...), res+i);
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Aq(T* res, T weight, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [weight](auto t) {
            return weight * eve::get<0>(t) / eve::sqrt(T{1} + eve::sqr(eve::get<1>(t)));
        });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Pow(T* res, T weight, T const* a, T const* b) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastPow(W{a+i}, W{b+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastPowabs(W{a+i}, W{b+i}), res+i);
        }
    }

    // unary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cpy(T* res, T w, T const* arg) {
        //eve::algo::copy(std::span{arg, S}, res);
        std::transform(arg, arg+S, res, [w](auto x) { return w * x; });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Neg(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::minus(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Inv(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::rec(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Abs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::abs(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Ceil(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::ceil(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Floor(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::floor(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Square(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sqr(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Exp(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastExp(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastLog(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log1p(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::log1p(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Logabs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastLog(eve::abs(W{arg+i})), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sin(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastSinCos<true>(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cos(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastSinCos<false>(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tan(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::tan(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Asin(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::asin(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Acos(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::acos(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Atan(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::atan(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sinh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sinh(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cosh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::cosh(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tanh(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * detail::FastTanh(W{arg + i}), res + i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrt(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sqrt(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrtabs(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sqrt(eve::abs(W{arg + i})), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cbrt(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::cbrt(W{arg + i}), res+i);
        }
    }
} // namespace Operon::Backend

#endif
