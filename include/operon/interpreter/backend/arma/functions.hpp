// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_ARMA_FUNCTIONS_HPP
#define OPERON_BACKEND_ARMA_FUNCTIONS_HPP

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "operon/interpreter/backend/backend.hpp"

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
    auto Fill(T* res, T value) {
        Map<T, S>(res).fill(value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    auto Add(T* res, auto const*... args) {
        Map<T, S>(res) = (Map<T const, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, auto const*... args) {
        Map<T, S>(res) = (Map<T const, S>(args) % ...);
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
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::min({first[i], args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, auto* first, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::max({first[i], args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T const* a, T const* b) {
        Map<T, S>(res) = Map<T const, S>(a) / arma::sqrt((T{1} + arma::square(Map<T const, S>(b))));
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        Map<T, S>(res) = arma::pow(Map<T const, S>(a), Map<T const, S>(b));
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
        Map<T, S>(res) = arma::abs(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T const* arg) {
        Map<T, S>(res) = arma::ceil(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T const* arg) {
        Map<T, S>(res) = arma::floor(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T const* arg) {
        Map<T, S>(res) = arma::square(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T const* arg) {
        Map<T, S>(res) = arma::exp(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        Map<T, S>(res) = arma::log(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        Map<T, S>(res) = arma::log1p(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        Map<T, S>(res) = arma::log(arma::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        Map<T, S>(res) = arma::sin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        Map<T, S>(res) = arma::cos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        Map<T, S>(res) = arma::tan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T const* arg) {
        Map<T, S>(res) = arma::asin(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T const* arg) {
        Map<T, S>(res) = arma::acos(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T const* arg) {
        Map<T, S>(res) = arma::atan(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T const* arg) {
        Map<T, S>(res) = arma::sinh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        Map<T, S>(res) = arma::cosh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        Map<T, S>(res) = arma::tanh(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        Map<T, S>(res) = arma::sqrt(Map<T const, S>(arg));
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        Map<T, S>(res) = arma::sqrt(arma::abs(Map<T const, S>(arg)));
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T const* arg) {
        Map<T, S>(res) = arma::cbrt(Map<T const, S>(arg));
    }
}  // namespace Operon::Backend
#endif
