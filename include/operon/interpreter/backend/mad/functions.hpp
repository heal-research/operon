// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_MAD_FUNCTIONS_HPP
#define OPERON_BACKEND_MAD_FUNCTIONS_HPP

#include "operon/interpreter/backend/backend.hpp"
#include "operon/core/node.hpp"

// #include "impl/aq.hpp"
#include "impl/exp.hpp"
#include "impl/inv.hpp"
#include "impl/log.hpp"
// #include "impl/pow.hpp"
#include "impl/sqrt.hpp"
#include "impl/trig.hpp"
#include "impl/tanh.hpp"

namespace Operon::Backend {
namespace detail::mad {
#if defined(OPERON_MATH_MAD_ARITHMETIC_FAST)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 4;
    static auto constexpr Sin = 1;
    static auto constexpr Cos = 1;
    static auto constexpr Exp = 1;
    static auto constexpr Log = 1;
    static auto constexpr Sqrt = 1;
    static auto constexpr Tanh = 1;
};
#elif defined(OPERON_MATH_MAD_ARITHMETIC_FASTER)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 2;
    static auto constexpr Sin = 1;
    static auto constexpr Cos = 1;
    static auto constexpr Exp = 1;
    static auto constexpr Log = 1;
    static auto constexpr Sqrt = 1;
    static auto constexpr Tanh = 1;
};
#elif defined(OPERON_MATH_MAD_ARITHMETIC_FASTEST)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 0;
    static auto constexpr Sin = 1;
    static auto constexpr Cos = 1;
    static auto constexpr Exp = 1;
    static auto constexpr Log = 1;
    static auto constexpr Sqrt = 1;
    static auto constexpr Tanh = 1;
};
#elif defined(OPERON_MATH_MAD_TRANSCENDENTAL_FAST)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 4;
    static auto constexpr Sin = 1;
    static auto constexpr Cos = 1;
    static auto constexpr Exp = 5;
    static auto constexpr Log = 5;
    static auto constexpr Sqrt = 4;
    static auto constexpr Tanh = 3;
};
#elif defined(OPERON_MATH_MAD_TRANSCENDENTAL_FASTER)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 4;
    static auto constexpr Sin = 1;
    static auto constexpr Cos = 1;
    static auto constexpr Exp = 3;
    static auto constexpr Log = 3;
    static auto constexpr Sqrt = 2;
    static auto constexpr Tanh = 2;
};
#elif defined(OPERON_MATH_MAD_TRANSCENDENTAL_FASTEST)
struct Precision {
    static auto constexpr Add = 2;
    static auto constexpr Sub = 2;
    static auto constexpr Mul = 2;
    static auto constexpr Div = 1;
    static auto constexpr Sin = 0;
    static auto constexpr Cos = 0;
    static auto constexpr Exp = 1;
    static auto constexpr Log = 1;
    static auto constexpr Sqrt = 1;
    static auto constexpr Tanh = 0;
};
#else
#error "Unknown MAD primitive specification"
#endif

    // unary details
    inline auto constexpr Inv(Operon::Scalar x) -> Operon::Scalar {
        return mad::InvImpl<Precision::Div>(x);
    }

    inline auto constexpr Log(Operon::Scalar x) -> Operon::Scalar {
        return mad::LogImpl<Precision::Log>(x);
    }

    inline auto constexpr Log1p(Operon::Scalar x) -> Operon::Scalar {
        return mad::Log1pImpl<Precision::Log>(x);
    }

    inline auto constexpr Logabs(Operon::Scalar x) -> Operon::Scalar {
        return mad::LogabsImpl<Precision::Log>(x);
    }

    inline auto constexpr Exp(Operon::Scalar x) -> Operon::Scalar {
        return mad::ExpImpl<Precision::Exp>(x);
    }

    inline auto constexpr Sin(Operon::Scalar x) -> Operon::Scalar {
        return mad::SinImpl<Precision::Sin>(x);
    }

    inline auto constexpr Cos(Operon::Scalar x) -> Operon::Scalar {
        return mad::CosImpl<Precision::Cos>(x);
    }

    inline auto constexpr Tan(Operon::Scalar x) -> Operon::Scalar {
        return mad::DivImpl<Precision::Div>(mad::SinImpl<Precision::Sin>(x), mad::CosImpl<Precision::Cos>(x));
    }

    inline auto constexpr Sinh(Operon::Scalar x) -> Operon::Scalar {
        auto const e = Exp(x);
        return (e*e - 1.F) * Inv(e+e);
    }

    inline auto constexpr Cosh(Operon::Scalar x) -> Operon::Scalar {
        auto const e = Exp(x);
        return (e*e + 1.F) * Inv(e+e);
    }

    inline auto constexpr ISqrt(Operon::Scalar x) -> Operon::Scalar {
        return mad::ISqrtImpl<Precision::Sqrt>(x);
    }

    inline auto constexpr Sqrt(Operon::Scalar x) -> Operon::Scalar {
        return mad::SqrtImpl<Precision::Sqrt>(x);
    }

    inline auto constexpr Sqrtabs(Operon::Scalar x) -> Operon::Scalar {
        return mad::SqrtabsImpl<Precision::Sqrt>(x);
    }

    inline auto constexpr Div(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
        return mad::DivImpl<Precision::Div>(x, y);
    }

    inline auto constexpr Pow(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
        return std::pow(x, y);
    }

    inline auto constexpr Tanh(Operon::Scalar x) -> Operon::Scalar {
        return mad::TanhImpl<Precision::Tanh>(x);
    }

    inline auto constexpr Aq(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
        auto constexpr p{9999999980506447872.F};
        return std::abs(y) > p ? DivImpl<Precision::Div>(x, std::abs(y)) : x * ISqrtImpl<Precision::Sqrt>(1 + y*y);
    }
} // namespace detail::mad

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
                res[i] = detail::mad::Inv(first[i]);
            } else {
                res[i] = detail::mad::Div(first[i], (rest[i] * ...));
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
        std::transform(a, a+S, b, res, detail::mad::Aq);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        std::transform(a, a+S, b, res, detail::mad::Pow);
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T const* arg) {
        std::ranges::copy_n(arg, S, res);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T const* arg) {
        std::transform(arg, arg+S, res, std::negate{});
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Inv);
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::abs(x); });
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
        std::transform(arg, arg+S, res, detail::mad::Exp);
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Log);
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Log1p);
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Logabs);
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Sin);
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Cos);
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Tan);
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
        std::transform(arg, arg+S, res, detail::mad::Sinh);
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Sinh);
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Tanh);
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Sqrt);
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::mad::Sqrtabs);
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return x * x; });
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return std::cbrt(x); });
    }
} // namespace Operon::Backend


#endif
