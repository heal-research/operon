// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EIGEN_FUNCTIONS_HPP
#define OPERON_BACKEND_EIGEN_FUNCTIONS_HPP

#include <Eigen/Core>
#include "operon/interpreter/backend/backend.hpp"

namespace Operon::Backend {
    template<typename T, std::size_t S>
    using Map = Eigen::Map<std::conditional_t<std::is_const_v<T>, Eigen::Array<std::remove_const_t<T>, S, 1> const, Eigen::Array<T, S, 1>>>;

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
        Map<T, S>(res, S, 1).setConstant(value);
    }

    template<typename T, std::size_t S>
    auto Fill(T* res, int n, T value) {
        Eigen::Map<Eigen::Array<T, -1, 1>>(res, n, 1).setConstant(value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    auto Add(T* res, auto const*... args) {
        Map<T, S>(res, S, 1) = (Map<T const, S>(args, S, 1) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, auto const*... args) {
        Map<T, S>(res, S, 1) = (Map<T const, S>(args, S, 1) * ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, auto* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res, S, 1) = Map<T const, S>(first, S, 1) - (Map<T const, S>(rest, S, 1) + ...);
    }

    template<typename T, std::size_t S>
    auto Div(T* res, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        Map<T, S>(res, S, 1) = Map<T const, S>(first, S, 1) / (Map<T const, S>(rest, S, 1) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        Map<T, S>(res, S, 1) = (Map<T const, S>(first, S, 1).min(Map<T const, S>(args, S, 1)), ...);
    }

    template<typename T, std::size_t S>
    auto Max(T* res, auto* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        Map<T, S>(res, S, 1) = (Map<T const, S>(first, S, 1).max(Map<T const, S>(args, S, 1)), ...);
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T const* a, T const* b) {
        Map<T, S>(res, S, 1) = Map<T const, S>(a, S, 1) / (T{1} + Map<T const, S>(b, S, 1).square()).sqrt();
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        Map<T, S>(res, S, 1) = Map<T const, S>(a, S, 1).pow(Map<T const, S>(b, S, 1));
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = -Map<T const, S>(arg, S, 1);
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).inverse();
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).abs();
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).ceil();
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).floor();
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).square();
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).exp();
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).log();
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).log1p();
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).abs().log();
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).sin();
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).cos();
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).tan();
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).asin();
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).acos();
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).atan();
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).sinh();
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).cosh();
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).tanh();
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).sqrt();
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).abs().sqrt();
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T const* arg) {
        Map<T, S>(res, S, 1) = Map<T const, S>(arg, S, 1).unaryExpr([](auto x) { return std::cbrt(x); });
    }
}  // namespace Operon::Backend
#endif
