// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_ARMA_FUNCTIONS_HPP
#define OPERON_BACKEND_ARMA_FUNCTIONS_HPP

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "operon/core/dispatch.hpp"

namespace Operon::Backend {
    template<typename T, std::size_t S>
    auto Map(T* ptr) {
        return arma::Mat<Operon::Scalar>(ptr, S, 1, /*copy_aux_mem =*/false, /*strict=*/true);
    }

    template<typename T, std::size_t S>
    auto Map(T const* ptr) {
        using U = std::remove_const_t<T>;
        return arma::Mat<Operon::Scalar>(const_cast<U*>(ptr), S, 1, /*copy_aux_mem =*/false, /*strict=*/true);
    }

    template<typename T, std::size_t S>
    auto Col(Backend::View<T, S> view, std::integral auto col) {
        return Map<T, S>(view.data_handle() + col * S);
    }

    template<typename T, std::size_t S>
    auto Col(Backend::View<T const, S> view, std::integral auto col) {
        return Map<T const, S>(view.data_handle() + col * S);
    }

    // utility
    template<typename T, std::size_t S>
    auto Fill(T* res, T weight, T value) {
        Map<T, S>(res).fill(value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    auto Add(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T const, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, T weight, auto const*... args) {
        Map<T, S>(res) = weight * (Map<T const, S>(args) % ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, T weight, auto* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * Map<T const, S>(first) - (Map<T const, S>(rest) + ...);
    }

    template<typename T, std::size_t S>
    auto Div(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res) = weight * Map<T const, S>(first) / (Map<T const, S>(rest) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * std::min({first[i], args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, T weight, auto* first, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * std::max({first[i], args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T weight, T const* a, T const* b) {
        Map<T, S>(res) = weight * Map<T const, S>(a) / arma::sqrt((T{1} + arma::square(Map<T const, S>(b))));
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T weight, T const* a, T const* b) {
        Map<T, S>(res) = weight * arma::pow(Map<T const, S>(a), Map<T const, S>(b));
    }

    template<typename T, std::size_t S>
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        Map<T, S>(res) = weight * arma::pow(arma::abs(Map<T const, S>(a)), Map<T const, S>(b));
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
        Map<T, S>(res) = weight * arma::abs(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::ceil(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::floor(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::square(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::exp(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::log(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::log1p(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::log(arma::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::sin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::cos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::tan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::asin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::acos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::atan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::sinh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::cosh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::tanh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::sqrt(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::sqrt(arma::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T weight, T const* arg) {
        Map<T, S>(res) = weight * arma::cbrt(Map<T const, S>(arg));
    }
}  // namespace Operon::Backend
#endif
