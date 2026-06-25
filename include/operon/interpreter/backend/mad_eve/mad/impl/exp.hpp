// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

#include <eve/wide.hpp>
#include <eve/module/core.hpp>

namespace Operon::Backend::Mad {
    // P=0: bit-cast trick (fastest, ~1% error)
    inline auto exp_v1(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};

        if (std::isnan(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < lower_limit) { return 0.F; }
        if (x > upper_limit) { return std::numeric_limits<float>::infinity(); }

        auto const f = (x * 12102203.161561485F) + 1065054451.F;
        auto const i = static_cast<int32_t>(f);
        return std::bit_cast<float>(i);
    }

    // P=1: polynomial refinement (http://stackoverflow.com/a/10792321)
    inline auto exp_v2(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};

        if (std::isnan(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < lower_limit) { return 0.F; }
        if (x > upper_limit) { return std::numeric_limits<float>::infinity(); }

        float const t  = x * std::numbers::log2e_v<float>;
        float const fi = std::floor(t);
        float const f  = t - fi;
        int   const i  = static_cast<int32_t>(fi);
        auto  const xf = ((0.3371894346F * f + 0.657636276F) * f) + 1.00172476F;
        auto  const xi = std::bit_cast<int32_t>(xf) + (i << 23);
        return std::bit_cast<float>(xi);
    }

    template<std::size_t P = 0>
    inline auto exp_impl(float x) -> float {
        if constexpr (P == 0) { return exp_v1(x); }
        else { return exp_v2(x); }
    }

    template<std::size_t P = 0, bool Check = true>
    auto exp_impl(eve::wide<float> x) -> eve::wide<float> {
        auto const lower_limit{eve::wide{-85.F}};
        auto const upper_limit{eve::wide{+85.F}};
        auto const inf{eve::inf(eve::as<eve::wide<float>>{})};

        auto exp = [](eve::wide<float> x) -> eve::wide<float> {
            if constexpr (P == 0) {
                auto const f = (x * 12102203.161561485F) + 1065054451.F;
                auto const i = eve::convert(f, eve::as<int32_t>{});
                return eve::bit_cast(i, eve::as<eve::wide<float>>{});
            } else {
                static_assert(P <= 1, "invalid precision spec for exp_impl");
                auto const t  = x * eve::wide<float>{std::numbers::log2e_v<float>};
                auto const fi = eve::floor(t);
                auto const f  = t - fi;
                auto const i  = eve::convert(fi, eve::as<int32_t>{});
                auto const xf = ((0.3371894346F * f + 0.657636276F) * f) + 1.00172476F;
                auto const xi = eve::bit_cast(xf, eve::as<eve::wide<int32_t>>{}) + (i << 23);
                return eve::bit_cast(xi, eve::as<eve::wide<float>>{});
            }
        };

        if constexpr (Check) {
            return eve::if_else(eve::is_nan(x),
                eve::nan(eve::as<eve::wide<float>>{}),
                eve::if_else(eve::is_eqz(x),
                    eve::wide<float>{1.F},
                    eve::if_else(x < lower_limit,
                        eve::wide<float>{0.F},
                        eve::if_else(x > upper_limit, inf, exp(x))
                    )
                )
            );
        } else {
            return exp(x);
        }
    }
} // namespace Operon::Backend::Mad
