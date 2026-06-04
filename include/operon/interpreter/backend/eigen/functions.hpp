// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_EIGEN_FUNCTIONS_HPP
#define OPERON_BACKEND_EIGEN_FUNCTIONS_HPP

#include <cmath>
#include <concepts>
#include <type_traits>
#include <Eigen/Core>
#include "operon/core/dispatch.hpp"

namespace Operon::Backend {
namespace detail {
    template<typename E, typename H>
    auto FoldMin(E e, H h) { return e.min(h); }
    template<typename E, typename H, typename... Tail>
    auto FoldMin(E e, H h, Tail... t) { return FoldMin(e.min(h), t...); }

    template<typename E, typename H>
    auto FoldMax(E e, H h) { return e.max(h); }
    template<typename E, typename H, typename... Tail>
    auto FoldMax(E e, H h, Tail... t) { return FoldMax(e.max(h), t...); }
} // namespace detail

    template<typename T, std::size_t S>
    using Map = Eigen::Map<std::conditional_t<std::is_const_v<T>,
        Eigen::Array<std::remove_const_t<T>, S, 1> const,
        Eigen::Array<T, S, 1>>>;

    // Avoid most-vexing-parse: use a factory instead of Map<T,S>(ptr) on LHS of =
    template<typename T, std::size_t S>
    auto MakeMap(T* ptr) { return Map<T, S>(ptr); }

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
        Map<T, S>(res).setConstant(value);
    }

    template<typename T, std::size_t S>
    auto Fill(T* res, int n, T value) {
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(res, n).setConstant(value);
    }

    // n-ary functions
    template<typename T, std::size_t S>
    auto Add(T* res, T weight, auto const*... args) {
        MakeMap<T, S>(res) = weight * (Map<T const, S>(args) + ...);
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, T weight, auto const*... args) {
        MakeMap<T, S>(res) = weight * (Map<T const, S>(args) * ...);
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        MakeMap<T, S>(res) = weight * (Map<T const, S>(first) - (Map<T const, S>(rest) + ...));
    }

    template<typename T, std::size_t S>
    auto Div(T* res, T weight, auto const* first, auto const*... rest) {
        static_assert(sizeof...(rest) > 0);
        MakeMap<T, S>(res) = weight * Map<T const, S>(first) / (Map<T const, S>(rest) * ...);
    }

    template<typename T, std::size_t S>
    auto Min(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        MakeMap<T, S>(res) = weight * detail::FoldMin(Map<T const, S>(first), Map<T const, S>(args)...);
    }

    template<typename T, std::size_t S>
    auto Max(T* res, T weight, auto const* first, auto const*... args) {
        static_assert(sizeof...(args) > 0);
        MakeMap<T, S>(res) = weight * detail::FoldMax(Map<T const, S>(first), Map<T const, S>(args)...);
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T weight, T const* a, T const* b) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(a) / (T{1} + Map<T const, S>(b).square()).sqrt();
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T weight, T const* a, T const* b) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(a).pow(Map<T const, S>(b));
    }

    template<typename T, std::size_t S>
    auto Powabs(T* res, T weight, T const* a, T const* b) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(a).abs().pow(Map<T const, S>(b));
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * -Map<T const, S>(arg);
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).inverse();
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).abs();
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).ceil();
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).floor();
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).square();
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).exp();
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).log();
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).log1p();
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).abs().log();
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).sin();
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).cos();
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).tan();
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).asin();
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).acos();
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).atan();
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).sinh();
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).cosh();
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).tanh();
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).sqrt();
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T weight, T const* arg) {
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).abs().sqrt();
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T weight, T const* arg) {
        // Eigen has no native cbrt; scalar fallback via unaryExpr. Alternative: pow(x, 1/3)
        // is faster but less accurate near zero and wrong for negative inputs.
        MakeMap<T, S>(res) = weight * Map<T const, S>(arg).unaryExpr([](auto x) { return std::cbrt(x); });
    }
}  // namespace Operon::Backend

#endif
