// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#pragma once

#include <cmath>
#include <concepts>
#include <limits>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

namespace Operon::Backend::Mad {
    template<std::size_t P = 0>
    inline auto isqrt_impl(float x) -> float {
        constexpr auto nan{ std::numeric_limits<float>::quiet_NaN() };
        constexpr auto inf{ std::numeric_limits<float>::infinity() };
        constexpr auto fast_sqrt_constant{static_cast<float>(0x5F3759DF)};

        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
        if (x == -0) { return -inf; }
        if (x == +0) { return +inf; }

        auto xt = x * 0.5F;
        auto xi = std::bit_cast<int32_t>(x);
        xi = fast_sqrt_constant - (xi >> 1U);
        auto xf = std::bit_cast<float>(xi);
        for (auto i = 0UL; i < P; ++i) {
            xf = xf * (1.5F - (xt * (xf * xf)));
        }
        return xf;
    }

    template<std::size_t P = 0>
    inline auto sqrt_impl(float x) -> float {
        constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
        if (x == 0) { return 0; }
        if constexpr (P == 0) {
            auto xi = std::bit_cast<int32_t>(x);
            xi -= 0x3F800000;
            xi = ((((xi >> 31U) & 1) << 31U) | ((xi >> 1U) & 0x7FFFFFFF));
            xi += 0x3F800000;
            xi &= 0xFFFFFFFF;
            return std::bit_cast<float>(xi);
        } else {
            return x * isqrt_impl<P>(x);
        }
    }

    template<std::size_t P = 0>
    inline auto sqrtabs_impl(float x) {
        return sqrt_impl<P>(std::abs(x));
    }

    template<std::size_t P = 0, bool Check = true>
    auto isqrt_impl(eve::wide<float> x) -> eve::wide<float> {
        auto const nan = eve::nan(eve::as<eve::wide<float>>());
        auto const fast_sqrt_constant = eve::wide<int32_t>{0x5F3759DF};

        auto isqrt = [=](eve::wide<float> x) {
            constexpr auto a{0.5F};
            constexpr auto b{1.5F};
            auto xt = x * a;
            auto xi = eve::bit_cast(x, eve::as(eve::wide<int32_t>{}));
            xi = fast_sqrt_constant - (xi >> 1U);
            auto xf = eve::bit_cast(xi, eve::as(eve::wide<float>{}));

            for (auto i = 0UL; i < P; ++i) {
                xf = xf * (b - (xt * (xf * xf)));
            }
            return xf;
        };

        if constexpr (Check) {
            return eve::if_else(eve::is_nan(x) || x < 0, nan,
                eve::if_else(x == 0, 0, isqrt(x))
            );
        } else {
            return isqrt(x);
        }
    }

    template<std::size_t P = 0, bool Check = true>
    auto sqrt_impl(eve::wide<float> x) -> eve::wide<float> {
        constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

        auto sqrt = [](eve::wide<float> x) {
            if constexpr (P == 0) {
                auto a = 0x3F800000;
                auto b = 0x3F800000;
                auto c = 0xFFFFFFFF;
                auto xi = eve::bit_cast(x, eve::as<eve::wide<int32_t>>());
                xi -= a;
                xi = ((((xi >> 31U) & 1) << 31U) | ((xi >> 1U) & 0x7FFFFFFF));
                xi += b;
                xi &= c;
                return eve::bit_cast(xi, eve::as<eve::wide<float>>());
            } else {
                return x * isqrt_impl<P, Check>(x);
            }
        };

        if constexpr (Check) {
            return eve::if_else(eve::is_nan(x) || x < 0, nan,
                eve::if_else(eve::is_eqz(x), 0, sqrt(x))
            );
        } else {
            return sqrt(x);
        }
    }
}  // namespace Operon::Backend::Mad
