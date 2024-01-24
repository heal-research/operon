// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_VDT_FUNCTIONS_HPP
#define OPERON_BACKEND_VDT_FUNCTIONS_HPP

#include <vdt/vdt.h>
#include <vdt/stdwrap.h>

#include "operon/interpreter/backend/backend.hpp"
#include "operon/core/node.hpp"

namespace Operon::Backend {

    namespace detail::vdt {
        // we need wrappers due to VDT calling conventions
        inline auto Acos(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_acosf(x); }
            else { return ::vdt::fast_acos(x); }
        }

        inline auto Asin(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_asinf(x); }
            else { return ::vdt::fast_asin(x); }
        }

        inline auto Atan(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_atanf(x); }
            else { return ::vdt::fast_atan(x); }
        }

        inline auto Cos(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_cosf(x); }
            else { return ::vdt::fast_cos(x); }
        }

        inline auto Sin(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_sinf(x); }
            else { return ::vdt::fast_sin(x); }
        }

        inline auto Tan(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_tanf(x); }
            else { return ::vdt::fast_tan(x); }
        }

        inline auto Tanh(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_tanhf(x); }
            else { return ::vdt::fast_tanh(x); }
        }

        inline auto Exp(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_expf(x); }
            else { return ::vdt::fast_exp(x); }
        }

        inline auto Log(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_logf(x); }
            else { return ::vdt::fast_log(x); }
        }

        inline auto Logabs(Operon::Scalar x) -> Operon::Scalar {
            return Log(std::abs(x));
        }

        inline auto Log1p(Operon::Scalar x) -> Operon::Scalar {
            return Log(1 + x);
        }

        inline auto Inv(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_invf(x); }
            else { return ::vdt::fast_inv(x); }
        }

        inline auto ISqrt(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            if constexpr (std::is_same_v<Operon::Scalar, float>) { return ::vdt::fast_isqrtf(x); }
            else { return ::vdt::fast_isqrt(x); }
        }

        inline auto Floor(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            return static_cast<Operon::Scalar>(::vdt::details::fpfloor(x));
        }

        inline auto Div(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            static_assert(std::is_arithmetic_v<decltype(x)>);
            return x * Inv(y);
        }
    } // namespace detail::vdt

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
    auto Add(T* res, auto*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = (args[i] + ...);
        }
    }

    template<typename T, std::size_t S>
    auto Mul(T* res, auto*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = (args[i] * ...);
        }
    }

    template<typename T, std::size_t S>
    auto Sub(T* res, auto* first, auto*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = -first[i];
            } else {
                res[i] = first[i] - (rest[i] + ...);
            }
        }
    }

    template<typename T, std::size_t S>
    auto Div(T* res, auto* first, auto*... rest) {
        for (auto i = 0UL; i < S; ++i) {
            if constexpr (sizeof...(rest) == 0) {
                res[i] = detail::vdt::Inv(first[i]);
            } else {
                // res[i] = first[i] * detail::vdt::Inv((rest[i] * ...));
                // res[i] = detail::vdt::Div(first[i], (rest[i] * ...));
                res[i] = first[i] / (rest[i] * ...); // this seems to be much faster than the above
            }
        }
    }

    template<typename T, std::size_t S>
    auto Min(T* res, auto*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::min({args[i]...});
        }
    }

    template<typename T, std::size_t S>
    auto Max(T* res, auto*... args) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::max({args[i]...});
        }
    }

    // binary functions
    template<typename T, std::size_t S>
    auto Aq(T* res, T* a, T* b) {
        for (auto i = 0UL; i < S; ++i) {
            auto v = b[i];
            res[i] = a[i] * detail::vdt::ISqrt((T{1} + v * v));
        }
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T* a, T* b) {
        for (auto i = 0UL; i < S; ++i) {
            res[i] = std::pow(a[i], b[i]);
        }
    }

    // unary functions
    template<typename T, std::size_t S>
    auto Cpy(T* res, T* arg) {
        std::ranges::copy_n(arg, S, res);
    }

    template<typename T, std::size_t S>
    auto Neg(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, std::negate{});
    }

    template<typename T, std::size_t S>
    auto Inv(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Inv);
    }

    template<typename T, std::size_t S>
    auto Abs(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::abs(x); });
    }

    template<typename T, std::size_t S>
    auto Square(T* res, T* arg) {
        std::transform(arg, arg+S, res, [](auto x) { return x * x; });
    }

    template<typename T, std::size_t S>
    auto Ceil(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::ceil(x); });
    }

    template<typename T, std::size_t S>
    auto Floor(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Floor);
    }

    template<typename T, std::size_t S>
    auto Exp(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Exp);
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Log);
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Log1p);
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Logabs);
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Sin);
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Cos);
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Tan);
    }

    template<typename T, std::size_t S>
    auto Asin(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Asin);
    }

    template<typename T, std::size_t S>
    auto Acos(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Acos);
    }

    template<typename T, std::size_t S>
    auto Atan(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Atan);
    }

    template<typename T, std::size_t S>
    auto Sinh(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::sinh(x); });
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::cosh(x); });
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, detail::vdt::Tanh);
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return x * detail::vdt::ISqrt(x); });
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::abs(x) * detail::vdt::ISqrt(std::abs(x)); });
    }

    template<typename T, std::size_t S>
    auto Cbrt(T* res, T* arg) {
        std::ranges::transform(std::span{arg, S}, res, [](auto x) { return std::cbrt(x); });
    }
} // namespace Operon::Backend

#endif
