// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EVE_FUNCTIONS_HPP
#define OPERON_BACKEND_EVE_FUNCTIONS_HPP

#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/detail/kumi.hpp>
#include <eve/module/math/regular/pow.hpp>

#include "operon/interpreter/backend/backend.hpp"
#include "operon/core/node.hpp"

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
    auto Add(T* res, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(eve::add(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Mul(T* res, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(eve::mul(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sub(T* res, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(eve::sub(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Div(T* res, auto const*... args) {
        using W = eve::wide<T>;
        constexpr auto L = W::size();
        for (auto i = 0UL; i < S; i += L) {
            eve::store(eve::div(W{args+i}...), res+i);
        }
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Min(T* res, auto const*... args) {
        return eve::algo::transform_to(eve::views::zip(std::span{args, S}...), res, eve::min);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Max(T* res, auto const*... args) {
        return eve::algo::transform_to(eve::views::zip(std::span{args, S}...), res, eve::max);
    }

    // binary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Aq(T* res, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [](auto t) {
            return eve::get<0>(t) / eve::sqrt(T{1} + eve::sqr(eve::get<1>(t)));
        });
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Pow(T* res, T const* a, T const* b) {
        eve::algo::transform_to(eve::views::zip(std::span{a, S}, b), res, [](auto t) {
            return eve::pow(eve::get<0>(t), eve::get<1>(t));
        });
    }

    // unary functions
    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cpy(T* res, T const* arg) {
        eve::algo::copy(std::span{arg, S}, res);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Neg(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::minus);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Inv(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::rec);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Abs(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::abs);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Ceil(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::ceil);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Floor(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::floor);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Square(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::sqr);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Exp(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::exp);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::log);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Log1p(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::log1p);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Logabs(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::log_abs);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sin(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::sin);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cos(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::cos);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tan(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::tan);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Asin(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::asin);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Acos(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::acos);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Atan(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::atan);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sinh(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::sinh);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cosh(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::cosh);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Tanh(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::tanh);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrt(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::sqrt);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Sqrtabs(T* res, T const* arg) {
        auto sqrtabs = [](auto x){ return eve::sqrt(eve::abs(x)); };
        eve::algo::transform_to(std::span{arg, S}, res, sqrtabs);
    }

    template<typename T, std::size_t S>
    requires (S % eve::wide<T>::size() == 0)
    auto Cbrt(T* res, T const* arg) {
        eve::algo::transform_to(std::span{arg, S}, res, eve::cbrt);
    }
} // namespace Operon::Backend

#endif