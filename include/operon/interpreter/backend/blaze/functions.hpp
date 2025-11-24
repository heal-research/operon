// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_BLAZE_FUNCTIONS_HPP
#define OPERON_BACKEND_BLAZE_FUNCTIONS_HPP

#include <blaze/Blaze.h>
#include "operon/core/dispatch.hpp"

namespace Operon::Backend {
    namespace detail {
        template<typename T>
        using CVector = blaze::CustomVector<T, blaze::unaligned, blaze::unpadded, blaze::columnVector>;
    } // namespace detail

    template<typename T, std::size_t S>
    inline auto Map(T* res) -> detail::CVector<T> {
        return detail::CVector<T>(res, S);
    }

    template<typename T, std::size_t S>
    inline auto Map(T const* res) -> detail::CVector<std::remove_const_t<T>> {
        using U = std::remove_const_t<T>;
        return detail::CVector<U>(const_cast<U*>(res), S);
    }

    template<typename T, std::size_t S>
    auto Col(Backend::View<T, S> view, std::integral auto col) {
        return Map<T, S>(view.data_handle() + col * S);
    }

    template<typename T, std::size_t S>
    auto Col(Backend::View<T const, S> view, std::integral auto col) {
        return Map<T, S>(view.data_handle() + col * S);
    }

    // utility
    template<typename T, std::size_t S>
    auto Fill(T* res, T value) {
        std::fill_n(res, S, value);
    }

    template<typename T, std::size_t S>
    auto Fill(T* res, int n, T value) {
        std::fill_n(res, n, value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    auto Add(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T const, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T const, S>(args) * ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, T weight, auto* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * (Map<T const, S>(first) - (Map<T const, S>(rest) + ...));
    }

    template<typename T, std::size_t S>
    auto Div(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * Map<T const, S>(first) / (Map<T const, S>(rest) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        for (auto i = 0U; i < S; ++i) {
            res[i] = weight * std::min({first[i], args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, T weight, auto* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        for (auto i = 0U; i < S; ++i) {
            res[i] = weight * std::max({first[i], args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T weight, T const* a, T const* b) {
        blaze::DynamicVector<T> x = Map<T const, S>(b);
        Map<T, S>(res) = weight * Map<T const, S>(a) / blaze::sqrt(T{1} + x * x);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T weight, T const* a, T const* b) {
        Map<T, S>(res) = weight * blaze::pow(Map<T const, S>(a), Map<T const, S>(b));
    }

    template<typename T, std::size_t S>
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        Map<T, S>(res) = weight * blaze::pow(blaze::abs(Map<T const, S>(a)), Map<T const, S>(b));
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * -Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight / Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::abs(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::ceil(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::floor(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * x * x; });
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::exp(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::log(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::log1p(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::log(blaze::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::sin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::cos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::tan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::asin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::acos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::atan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::sinh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::cosh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::tanh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::sqrt(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::sqrt(blaze::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * blaze::cbrt(Map<T const, S>(arg));
    }
} // namespace Operon::Backend
#endif
