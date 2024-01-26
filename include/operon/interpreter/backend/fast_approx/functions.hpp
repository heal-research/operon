// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_fast_approx_FUNCTIONS_HPP
#define OPERON_BACKEND_fast_approx_FUNCTIONS_HPP

#include "operon/interpreter/backend/backend.hpp"
#include "operon/core/node.hpp"

#include "impl/aq.hpp"
#include "impl/exp.hpp"
#include "impl/inv.hpp"
#include "impl/log.hpp"
#include "impl/pow.hpp"
#include "impl/sqrt.hpp"
#include "impl/trig.hpp"

namespace Operon::Backend {
    namespace detail::fast_approx {
        static constexpr auto Precision = OPERON_MATH_FAST_APPROX_PRECISION;

        // unary details
        inline auto constexpr Inv(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::InvImpl<Precision>(x);
        }

        inline auto constexpr Log(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::LogImpl(x);
        }

        inline auto constexpr Log1p(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::Log1pImpl(x);
        }

        inline auto constexpr Logabs(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::LogabsImpl(x);
        }

        inline auto constexpr Exp(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::ExpImpl<Precision>(x);
        }

        inline auto constexpr Sin(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::SinImpl<Precision>(x);
        }

        inline auto constexpr Cos(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::CosImpl<Precision>(x);
        }

        inline auto constexpr Tan(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::TanImpl<Precision>(x);
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
            return fast_approx::ISqrtImpl<Precision>(x);
        }

        inline auto constexpr Sqrt(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::SqrtImpl<Precision>(x);
        }

        inline auto constexpr Sqrtabs(Operon::Scalar x) -> Operon::Scalar {
            return fast_approx::SqrtabsImpl<Precision>(x);
        }

        inline auto constexpr Div(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            return fast_approx::DivImpl<Precision>(x, y);
        }

        inline auto constexpr Pow(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            return fast_approx::PowImpl<Precision>(x, y);
        }

        // fast_approx: tanh uses the less precise division operation
        inline auto constexpr Tanh(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            if (std::isnan(x)) { return nan; }
            if (x < -85) { return -1.F; }
            if (x > +85) { return +1.F; }

            auto expZeroShift = [](auto x) {
                constexpr auto shift{23U};
                constexpr auto ff{127U};
                auto a = Div((1U << shift), std::numbers::ln2_v<Operon::Scalar>);
                auto b = (ff * (1U << shift));
                auto f = a * x + b;
                auto i = static_cast<int32_t>(f);
                return std::bit_cast<float>(i);
            };

            auto a = expZeroShift(x);
            auto b = expZeroShift(-x);
            return Div(a-b, a+b);
        }

        inline auto constexpr Aq(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            return fast_approx::AqImpl<Precision>(x, y);
        }
    } // namespace detail::fast_approx

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
                res[i] = detail::fast_approx::Inv(first[i]);
            } else {
                res[i] = detail::fast_approx::Div(first[i], (rest[i] * ...));
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
        std::transform(a, a+S, b, res, detail::fast_approx::Aq);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        std::transform(a, a+S, b, res, detail::fast_approx::Pow);
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
        std::transform(arg, arg+S, res, detail::fast_approx::Inv);
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
        std::transform(arg, arg+S, res, detail::fast_approx::Exp);
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Log);
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Log1p);
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Logabs);
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Sin);
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Cos);
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Tan);
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
        std::transform(arg, arg+S, res, detail::fast_approx::Sinh);
    }

    template<typename T, std::size_t S>
    auto Cosh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Sinh);
    }

    template<typename T, std::size_t S>
    auto Tanh(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Tanh);
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Sqrt);
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_approx::Sqrtabs);
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
