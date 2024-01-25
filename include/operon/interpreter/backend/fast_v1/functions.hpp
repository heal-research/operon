// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_BACKEND_FAST_V1_FUNCTIONS_HPP
#define OPERON_BACKEND_FAST_V1_FUNCTIONS_HPP

#include "operon/interpreter/backend/backend.hpp"
#include "operon/core/node.hpp"

namespace Operon::Backend {
    namespace detail::fast_v1 {
        inline auto constexpr Inv(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_same_v<Operon::Scalar, float>, "this function only works in single-precision mode.");

            constexpr auto inf{ std::numeric_limits<float>::infinity() };
            constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();

            if (x == -0) { return -inf; }
            if (x == +0) { return inf; }
            if (std::isinf(x)) { return 0; }
            if (std::isnan(x)) { return nan; }

            auto sx = (x < 0) ? -1.F : 1.F;
            x = sx * x;

            auto constexpr m{0x7EF127EA};
            auto xi = static_cast<int>(m - std::bit_cast<std::uint32_t>(x));
            auto xf = std::bit_cast<float>(xi);
            auto w  = x * xf;

            // Efficient Iterative Approximation Improvement in horner polynomial form.
            xf = xf * (2 - w);     // Single iteration, Err = -3.36e-3 * 2^(-flr(log2(x)))
            // xf = xf * ( 4 + w * (-6 + w * (4 - w)));  // Second iteration, Err = -1.13e-5 * 2^(-flr(log2(x)))
            // xf = xf * (8 + w * (-28 + w * (56 + w * (-70 + w *(56 + w * (-28 + w * (8 - w)))))));  // Third Iteration, Err = +-6.8e-8 *  2^(-flr(log2(x)))
            return xf * sx;
        }

        inline auto constexpr Inv2(Operon::Scalar x) -> Operon::Scalar {
            /* Reciprocal approximation.

            See the following link for more details:
            https://bit.ly/42qbEHG.

            Number of multiply-adds: 3 (or 2 if `x ** 2` can be approximated).
            */
            constexpr auto inf{ std::numeric_limits<float>::infinity() };
            constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
            constexpr uint32_t fast_reciprocal_constant{0xBE6EB3BE};

            if (x == -0) { return -inf; }
            if (x == +0) { return inf; }
            if (std::isinf(x)) { return 0; }
            if (std::isnan(x)) { return nan; }

            auto xt = x < 0 ? -x : x;

            auto xi = std::bit_cast<uint32_t>(xt);
            xi = (fast_reciprocal_constant - xi) >> 1U;

            auto xf = std::bit_cast<float>(xi);
            xf = xf * xf;
            xf = xf * (2 - xf * xt);   // one iteration
            if (x < 0) { xf = xf * -1; }
            return xf;
        }

        inline auto constexpr Div(Operon::Scalar x1, Operon::Scalar x2) -> Operon::Scalar {
            static_assert(std::is_same_v<Operon::Scalar, float>, "this function only works in single-precision mode.");
            constexpr auto inf{ std::numeric_limits<Operon::Scalar>::infinity() };
            constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
            if (x1 == 0) { return x2 == 0 ? nan : 0.F; }
            if (x2 == -0) { return -inf; }
            if (x2 == +0) { return +inf; }
            return x1 * Inv(x2);
        }

        inline auto constexpr ISqrt(Operon::Scalar x) -> Operon::Scalar {
            static_assert(std::is_same_v<Operon::Scalar, float>, "this function only works in single-precision mode.");
            constexpr auto nan{ std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            constexpr auto inf{ std::numeric_limits<Operon::Scalar>::infinity() };
            constexpr auto fast_sqrt_constant{static_cast<float>(0x5F3759DF)};

            if (x < 0) { return nan; }
            if (x == -0) { return -inf; }
            if (x == +0) { return +inf; }

            auto xt = x * 0.5F;
            auto xi = std::bit_cast<int32_t>(x);
            xi = fast_sqrt_constant - (xi >> 1U);
            auto xf = std::bit_cast<float>(xi);
            xf = xf * (float{1.5} - (xt * (xf * xf)));
            return xf;
        }

        inline auto constexpr ISqrt2(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto nan{ std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            constexpr auto inf{ std::numeric_limits<Operon::Scalar>::infinity() };
            if (x < 0) { return nan; }
            if (x == -0) { return -inf; }
            if (x == +0) { return +inf; }

            auto xi = std::bit_cast<int32_t>(x);
            xi = (0xBE6EB3BE - xi) >> 1U;
            auto xf = std::bit_cast<float>(xi);
            xf = xf * (1.5F - (0.5F * x * (xf * xf)));
            return xf;
        }

        inline auto constexpr Sqrt(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
            if (x < 0) { return nan; }
            if (x == 0) { return 0; }
            return x * ISqrt(x);
        }

        inline auto constexpr Sqrtabs(Operon::Scalar x) {
            return Sqrt(std::abs(x));
        }

        inline auto constexpr Cos2(Operon::Scalar x) -> Operon::Scalar {
            auto constexpr invpi = std::numbers::inv_pi_v<Operon::Scalar>;
            x = std::abs(x) * invpi + 1.5F;
            auto xx = 2 * static_cast<float>(static_cast<int>(x/2));
            x = x - xx - 1;

            auto f = x < 0 ? 4 * (x*x + x)
                : 4 * (-(x*x) + x);

            return f;
        }

        inline auto constexpr Sin2(Operon::Scalar x) -> Operon::Scalar {
            auto constexpr invpi = std::numbers::inv_pi_v<Operon::Scalar>;
            auto offset = x < 0 ? 2.0F : 1.0F;
            x = std::abs(x) * invpi + offset;

            auto xx = 2 * static_cast<float>(static_cast<int>(x/2));
            x = x - xx - 1;

            auto f = x < 0 ? 4 * (x*x + x)
                : 4 * (-(x*x) + x);

            return f;
        }

        inline auto constexpr Cos(float x) noexcept
        {
            constexpr float tp = std::numbers::inv_pi_v<float>/2;
            constexpr float a{.25F};
            constexpr float b{16.F};
            constexpr float c{.50F};
            constexpr float d{.225F};
            x *= tp;
            x -= a + std::floor(x + a);
            x *= b * (std::abs(x) - c);
            x += d * x * (std::abs(x) - 1.F); // another step for extra precision
            return x;
        }

        inline auto constexpr Sin(float x) noexcept
        {
            constexpr float tp = std::numbers::pi_v<float>/2;
            return Cos(x - tp);
        }

        inline auto constexpr Tan(Operon::Scalar x) {
            return Div(Sin(x), Cos(x));
        }

        inline auto constexpr Exp(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto inf{ std::numeric_limits<float>::infinity() };
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };

            if (std::isnan(x)) { return nan; }
            if (x < -85.F) { return 0; }
            if (x > 85.F) { return inf; }
            return std::bit_cast<float>(static_cast<int>(12102203 * x + 1064866816));
        }

        inline auto constexpr Exp2(Operon::Scalar x) -> Operon::Scalar {
            float t = x * 1.442695041f;
            float fi = std::floor(t);
            float f = t - fi;
            int i = static_cast<int32_t>(fi);
            auto xf = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
            auto xi = (std::bit_cast<int32_t>(xf) + (i << 23));                                          /* scale by 2^i */
            return std::bit_cast<float>(xi);
        }

        inline auto constexpr Log(Operon::Scalar x) {
            constexpr auto inf { std::numeric_limits<Operon::Scalar>::infinity() };
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            if (std::isnan(x)) { return nan; }
            if (x < 0) { return nan; }
            if (x == 0) { return -inf; }
            if (x == 1) { return 0.F; }
            return (std::bit_cast<int32_t>(x) - 1065353217) * 8.262958405176314e-8F;
        }

        inline auto constexpr Log1p(Operon::Scalar x) {
            return Log(1 + x);
        }

        inline auto constexpr Logabs(Operon::Scalar x) {
            return Log(std::abs(x));
        }

        inline auto constexpr Pow(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            constexpr auto inf { std::numeric_limits<Operon::Scalar>::infinity() };
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };

            if (std::isnan(x)) { return nan; }
            if (std::isnan(y)) { return nan; }
            if (x == 0) { return y < 0 ? inf : x; }
            if (x < 0) { return nan; }
            if (y == 0) { return 1.F; }
            if (y < -85.F) { return 0; }
            if (y > 85.F) { return inf; }

            auto xi = std::bit_cast<int32_t>(x);
            auto a = static_cast<int>(y * (xi - 1064866805) + static_cast<float>(1064866805));
            return std::bit_cast<float>(a);
        }

        inline auto Pow2(Operon::Scalar x, Operon::Scalar y) {
            auto log2 = [](Operon::Scalar x) {
                auto i = std::bit_cast<std::uint32_t>(x);
                auto f = std::bit_cast<float>((i & 0x007FFFFF) | 0x3f000000);
                auto y = i * 1.1920928955078125e-7f;
                return y - 124.22551499f - 1.498030302f * f - 1.72587999f / (0.3520887068f + f);
            };

            auto pow2 = [](Operon::Scalar p) {
                float offset = (p < 0) ? 1.0f : 0.0f;
                float clipp = (p < -126) ? -126.0f : p;
                float z = clipp - static_cast<int32_t>(clipp) + offset;
                auto i = std::uint32_t((1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z));
                return std::bit_cast<float>(i);
            };
            return pow2(y * log2(x));
        }

        inline auto constexpr Sinh(Operon::Scalar x) -> Operon::Scalar {
            auto const e = Exp(x);
            return Div(e*e - Operon::Scalar{1}, e+e);
        }

        inline auto constexpr Cosh(Operon::Scalar x) -> Operon::Scalar {
            auto const e = Exp(x);
            return Div(e*e + Operon::Scalar{1}, e+e);
        }

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

        inline auto constexpr TanhAlt(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            if (std::isnan(x)) { return nan; }

            auto constexpr r1{8.F / 3.F};
            auto constexpr r2{1.F / 9.F};
            if (x <= -3) { return -1.F; }
            if (x >= 3) { return 1.F; }

            auto xr = Inv(x*x + 3);
            return x * (r1 * xr + r2);
        }

        inline auto constexpr Aq(Operon::Scalar x1, Operon::Scalar x2) {
            auto constexpr p{9999999980506447872.F};
            return std::abs(x2) > p ? Div(x1, std::abs(x2)) : x1 * ISqrt(1 + x2*x2);
        }
    } // namespace detail::fast_v1

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
                res[i] = detail::fast_v1::Inv(first[i]);
            } else {
                res[i] = detail::fast_v1::Div(first[i], (rest[i] * ...));
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
        std::transform(a, a+S, b, res, detail::fast_v1::Aq);
    }

    template<typename T, std::size_t S>
    auto Pow(T* res, T const* a, T const* b) {
        std::transform(a, a+S, b, res, detail::fast_v1::Pow);
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
        std::transform(arg, arg+S, res, detail::fast_v1::Inv);
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
        std::transform(arg, arg+S, res, detail::fast_v1::Exp);
    }

    template<typename T, std::size_t S>
    auto Log(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Log);
    }

    template<typename T, std::size_t S>
    auto Log1p(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Log1p);
    }

    template<typename T, std::size_t S>
    auto Logabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Logabs);
    }

    template<typename T, std::size_t S>
    auto Sin(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Sin);
    }

    template<typename T, std::size_t S>
    auto Cos(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Cos);
    }

    template<typename T, std::size_t S>
    auto Tan(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Tan);
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
        std::transform(arg, arg+S, res, detail::fast_v1::Tanh);
    }

    template<typename T, std::size_t S>
    auto Sqrt(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Sqrt);
    }

    template<typename T, std::size_t S>
    auto Sqrtabs(T* res, T const* arg) {
        std::transform(arg, arg+S, res, detail::fast_v1::Sqrtabs);
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
