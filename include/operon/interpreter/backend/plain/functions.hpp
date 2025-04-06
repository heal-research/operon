// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_PLAIN_FUNCTIONS_HPP
#define OPERON_BACKEND_PLAIN_FUNCTIONS_HPP

#include "operon/core/dispatch.hpp"
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

    // unary functions
    template<typename T, std::size_t S>
    auto Add(T* res, T weight, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * (args[i] + ...);
        }
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, T weight, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * (args[i] * ...);
        }
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, T weight, auto const* first, auto const*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = weight * -first[i];
            } else {
                res[i] = weight * (first[i] - (rest[i] + ...));
            }
        }
    }

    template<typename T, std::size_t S>
    auto Div(T* res, T weight, auto const* first, auto const*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = weight / first[i];
            } else {
                res[i] = weight * first[i] / (rest[i] * ...);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Min(T* res, T weight, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * std::min({args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, T weight, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = weight * std::max({args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T weight, T const* a, T const* b) {
        std::transform(a, a+S, b, res, [&](auto x, auto y) { return weight * x / std::sqrt(T{1} + y*y); });
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T weight, T const* a, T const* b) {
        std::transform(a, a+S, b, res, [&](auto x, auto y) { return weight * std::pow(x, y); });
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T w, T const* arg) {
        //std::copy_n(arg, S, res);
        std::transform(arg, arg+S, res, [w](auto x) { return w * x; });
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * -x; });
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight / x; });
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::abs(x); });
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * x * x; });
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::ceil(x); });
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::floor(x); });
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::exp(x); });
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::log(x); });
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::log1p(x); });
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::log(std::abs(x)); });
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::sin(x); });
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::cos(x); });
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::tan(x); });
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::asin(x); });
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::acos(x); });
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::atan(x); });
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::sinh(x); });
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::cosh(x); });
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::tanh(x); });
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::sqrt(x); });
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto v){ return weight * std::sqrt(std::abs(v)); });
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T weight, T const* arg) {
        std::transform(arg, arg+S, res, [&](auto x) { return weight * std::cbrt(x); });
    }
} // namespace Operon::Backend

#endif
