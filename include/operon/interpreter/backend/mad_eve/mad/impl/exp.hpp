#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <numbers>

#include <eve/wide.hpp>
#include <eve/module/core.hpp>

namespace Operon::Backend::Mad {
    auto exp_v1(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};

        if (std::isnan(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < lower_limit) { return 0; }
        if (x > upper_limit) { return std::numeric_limits<float>::infinity(); }

        auto const f = (x * 12102203.161561485) + 1065054451;
        auto const i = static_cast<int32_t>(f);
        return std::bit_cast<float>(i);
    }

    // http://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo/10792321#10792321
    auto exp_v2(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};

        if (std::isnan(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < lower_limit) { return 0; }
        if (x > upper_limit) { return std::numeric_limits<float>::infinity(); }

        float t = x * std::numbers::log2e_v<float>;
        float fi = std::floor(t);
        float f = t - fi;
        int i = static_cast<int32_t>(fi);
        auto xf = ((0.3371894346F * f + 0.657636276F) * f) + 1.00172476F; /* compute 2^f */
        auto xi = (std::bit_cast<int32_t>(xf) + (i << 23));             /* scale by 2^i */
        return std::bit_cast<float>(xi);
    }

    auto exp_v3(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};

        if (std::isnan(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < lower_limit) { return 0; }
        if (x > upper_limit) { return std::numeric_limits<float>::infinity(); }

        constexpr float a = (1U << 23U) / std::numbers::ln2_v<float>;
        constexpr float b = (1U << 23U) * (127 - 0.043677448F);
        x = a * x + b;

        constexpr float c = (1U << 23U);
        constexpr float d = (1U << 23U) * 255;
        if (x < c || x > d) {
            x = (x < c) ? 0.0F : d;
        }

        return std::bit_cast<float>(static_cast<uint32_t>(x));
    }

    template<std::size_t P = 0>
    auto exp_impl(float x) -> float {
        if constexpr (P == 0) { return exp_v1(x); }
        else { return exp_v2(x); }
    }

    template<std::size_t P = 0, bool Check = true>
    auto exp_impl(eve::wide<float> x) {
        auto const lower_limit{eve::wide{-85.F}};
        auto const upper_limit{eve::wide{+85.F}};
        auto const inf{eve::inf(eve::as<eve::wide<float>>{})};

        auto exp = [](eve::wide<float> x) {
            if constexpr (P == 0) {
                auto const f = (x * 12102203.161561485) + 1065054451;
                auto const i = eve::convert(f, eve::as<int32_t>{});
                return eve::bit_cast(i, eve::as<eve::wide<float>>{});
            } else {
                static_assert(P <= 1, "invalid precision spec for exp_impl");
                auto t  = x * eve::wide<float>{std::numbers::log2e_v<float>};
                auto fi = eve::floor(t);
                auto f  = t - fi;
                auto i  = eve::convert(fi, eve::as<int32_t>{});
                auto xf = ((0.3371894346F * f + 0.657636276F) * f) + 1.00172476F; /* compute 2^f */
                auto xi = eve::bit_cast(xf, eve::as<eve::wide<int32_t>>{}) + (i << 23);
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
                        eve::if_else(x > upper_limit,
                            inf,
                            exp(x)
                        )
                    )
                )
            );
        } else {
            return exp(x);
        }
    }
} // namespace Operon::Backend::Mad
