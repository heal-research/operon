// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_FASTOR_FUNCTIONS_HPP
#define OPERON_BACKEND_FASTOR_FUNCTIONS_HPP

// #define FASTOR_DONT_ALIGN // avoid segfaults due to wrong assumptions about alignment
#include <Fastor/Fastor.h>

#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"

namespace Operon::Backend {
    template<typename T, std::size_t S>
    auto Map(T* ptr) { return Fastor::TensorMap<T, S>(ptr); }

    template<typename T, std::size_t S>
    auto Map(T const* ptr) {
        return Fastor::TensorMap<T, S>(const_cast<T*>(ptr));
    }

    template<typename T, std::size_t S>
    auto Col(Backend::View<T, S> view, std::integral auto col) {
        using U = std::remove_const_t<T>;
        return Map<U, S>(const_cast<U*>(view.data_handle() + col * S));
    }

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
    auto Add(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T, S>(args) * ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * Map<T, S>(first) - (Map<T, S>(rest) + ...);
    }

    template<typename T, std::size_t S>
    auto Div(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * Map<T, S>(first) / (Map<T, S>(rest) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        Map<T, S>(res) = weight * (Fastor::min(Map<T, S>(first), Map<T, S>(args)), ...);
    }

    template<typename T, std::size_t S>
    auto Max(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        Map<T, S>(res) = weight * (Fastor::max(Map<T, S>(first), Map<T, S>(args)), ...);
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T weight, T* a, T* b) {
        auto m = Map<T, S>(b);
        Map<T, S>(res) = weight * Map<T, S>(a) / Fastor::sqrt(T{1} + m * m);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T weight, T* a, T* b) {
        Map<T, S>(res) = weight * Fastor::pow(Map<T, S>(a), Map<T, S>(b));
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Map<T, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * -Map<T, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight / Map<T, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::abs(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T weight, T* arg) {
        auto a = Map<T, S>(arg);
        Map<T, S>(res) = a * a;
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::ceil(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::floor(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::exp(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::log(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::log1p(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::log(Fastor::abs(Map<T, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::sin(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::cos(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::tan(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::asin(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::acos(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::atan(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::sinh(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::cosh(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::tanh(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::sqrt(Map<T, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::sqrt(Fastor::abs(Map<T, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T weight, T* arg) {
        Map<T, S>(res) = weight * Fastor::cbrt(Map<T, S>(arg));
    }
} // namespace Operon::Backend

#endif
