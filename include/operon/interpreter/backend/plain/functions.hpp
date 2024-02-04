// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_PLAIN_FUNCTIONS_HPP
#define OPERON_BACKEND_PLAIN_FUNCTIONS_HPP

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

    // unary functions
    template<typename T, std::size_t S>
    auto Add(T* res, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = (args[i] + ...);
        }
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = (args[i] * ...);
        }
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, auto const* first, auto const*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = -first[i];
            } else {
                res[i] = first[i] - (rest[i] + ...);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Div(T* res, auto const* first, auto const*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = T{1} / first[i];
            } else {
                res[i] = first[i] / (rest[i] * ...);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Min(T* res, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::min({args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, auto const*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::max({args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T const* a, T const* b) {
        std::transform(a, a+S, b, res, [](auto x, auto y) { return x / std::sqrt(T{1} + y*y); });
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        std::transform(a, a+S, b, res, [](auto x, auto y) { return std::pow(x, y); });
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T const* arg) {
        std::copy_n(arg, S, res);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T const* arg) {
        std::transform(arg, arg+S, res, std::negate{});
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return T{1} / x; });
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::abs(x); });
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return x * x; });
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::ceil(x); });
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::floor(x); });
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::exp(x); });
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::log(x); });
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::log1p(x); });
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::log(std::abs(x)); });
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::sin(x); });
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::cos(x); });
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::tan(x); });
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::asin(x); });
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::acos(x); });
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::atan(x); });
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::sinh(x); });
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::cosh(x); });
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::tanh(x); });
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::sqrt(x); });
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto v){ return std::sqrt(std::abs(v)); });
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::cbrt(x); });
    }
} // namespace Operon::Backend

#endif