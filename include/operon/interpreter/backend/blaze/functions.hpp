// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_BLAZE_FUNCTIONS_HPP
#define OPERON_BACKEND_BLAZE_FUNCTIONS_HPP

#include <blaze/Blaze.h>
#include "operon/interpreter/backend/backend.hpp"

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
    auto Add(T* res, auto const*... args) {
        Map<T, S>(res) = (Map<T const, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, auto const*... args) {
        Map<T, S>(res) = (Map<T const, S>(args) * ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, auto* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = Map<T const, S>(first) - (Map<T const, S>(rest) + ...);
    }

    template<typename T, std::size_t S>
    auto Div(T* res, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = Map<T const, S>(first) / (Map<T const, S>(rest) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        for (auto i = 0U; i < S; ++i) {
            res[i] = std::min({first[i], args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, auto* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        for (auto i = 0U; i < S; ++i) {
            res[i] = std::max({first[i], args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T const* a, T const* b) {
        blaze::DynamicVector<T> x = Map<T const, S>(b);
        Map<T, S>(res) = Map<T const, S>(a) / blaze::sqrt(T{1} + x * x);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        Map<T, S>(res) = blaze::pow(Map<T const, S>(a), Map<T const, S>(b));
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T const* arg) {
        Map<T, S>(res) = Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T const* arg) {
        Map<T, S>(res) = -Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T const* arg) {
        Map<T, S>(res) = T{1} / Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T const* arg) {
        Map<T, S>(res) = blaze::abs(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T const* arg) {
        Map<T, S>(res) = blaze::ceil(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T const* arg) {
        Map<T, S>(res) = blaze::floor(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return x * x; });
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T const* arg) {
        Map<T, S>(res) = blaze::exp(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        Map<T, S>(res) = blaze::log(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        Map<T, S>(res) = blaze::log1p(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        Map<T, S>(res) = blaze::log(blaze::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        Map<T, S>(res) = blaze::sin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        Map<T, S>(res) = blaze::cos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        Map<T, S>(res) = blaze::tan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T const* arg) {
        Map<T, S>(res) = blaze::asin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T const* arg) {
        Map<T, S>(res) = blaze::acos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T const* arg) {
        Map<T, S>(res) = blaze::atan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T const* arg) {
        Map<T, S>(res) = blaze::sinh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        Map<T, S>(res) = blaze::cosh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        Map<T, S>(res) = blaze::tanh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        Map<T, S>(res) = blaze::sqrt(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        Map<T, S>(res) = blaze::sqrt(blaze::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T const* arg) {
        Map<T, S>(res) = blaze::cbrt(Map<T const, S>(arg));
    }
} // namespace Operon::Backend
#endif
