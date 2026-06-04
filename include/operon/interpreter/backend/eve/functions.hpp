// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EVE_FUNCTIONS_HPP
#define OPERON_BACKEND_EVE_FUNCTIONS_HPP

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include "operon/core/dispatch.hpp"

namespace Operon::Backend::detail {

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
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [weight](auto t) {
            return weight * eve::pow(eve::get<0>(t), eve::get<1>(t));
        });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [weight](auto t) {
            return weight * eve::pow(eve::abs(eve::get<0>(t)), eve::get<1>(t));
        });
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
            eve::store(weight * eve::exp(W{arg+i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::log(W{arg+i}), res+i);
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
            eve::store(weight * eve::log(eve::abs(W{arg+i})), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sin(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::sin(W{arg + i}), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cos(T* res, T weight, T const* arg) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(weight * eve::cos(W{arg + i}), res+i);
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
