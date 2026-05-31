#pragma once

#include <eve/module/core/regular/is_eqz.hpp>
#include "inv.hpp"
#include "exp.hpp"
#include "log.hpp"

namespace Operon::Backend::Mad {
    auto pow_v1(float x, float y) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};
        constexpr auto inf { std::numeric_limits<float>::infinity() };
        constexpr auto nan { std::numeric_limits<float>::quiet_NaN() };

        if (std::isnan(x)) { return nan; }
        if (std::isnan(y)) { return nan; }
        if (x == 0) { return y < 0 ? inf : x; }
        if (x < 0) { return nan; }
        if (y == 0) { return 1.F; }
        if (y < lower_limit) { return 0; }
        if (y > upper_limit) { return inf; }

        auto xi = std::bit_cast<int32_t>(x);
        auto a = static_cast<int>(y * (xi - 1064866805) + static_cast<float>(1064866805));
        return std::bit_cast<float>(a);
    }

    auto pow_v2(float x, float y) {
        auto log2 = [](float x) {
            auto i = std::bit_cast<std::uint32_t>(x);
            auto f = std::bit_cast<float>((i & 0x007FFFFF) | 0x3F000000);
            auto y = i * 1.1920928955078125e-7f;
            return y - 124.22551499f - 1.498030302f * f - 1.72587999F / (0.3520887068F + f);
        };

        auto pow2 = [](float p) {
            float offset = (p < 0) ? 1.0F : 0.0F;
            float clipp = (p < -126) ? -126.0F : p;
            float z = clipp - static_cast<int32_t>(clipp) + offset;
            auto i = static_cast<std::uint32_t>((1 << 23U) * (clipp + 121.2740575F + 27.7280233F / (4.84252568F - z) - 1.49012907F * z));
            return std::bit_cast<float>(i);
        };
        return pow2(y * log2(x));
    }

    template<std::size_t P = 0>
    auto pow_impl(float x, float y) -> float {
        if constexpr (P == 0) { return pow_v1(x, y); }
        // else { return PowV2(x, y); }
        else { return exp_impl<P>(y * log_impl<P>(x)); }
    }

    template<std::size_t P = 0>
    auto pow_impl(eve::wide<float> x, eve::wide<float> y) -> eve::wide<float> {
        auto const lower_limit = eve::wide<float>{-85.F};
        auto const upper_limit = eve::wide<float>{+85.F};
        auto const inf         = eve::inf(eve::as<eve::wide<float>>{});
        auto const nan         = eve::nan(eve::as<eve::wide<float>>{});

        auto pow_v1 = [](eve::wide<float> x, eve::wide<float> y) {
            auto xi = eve::bit_cast(x, eve::as<eve::wide<int32_t>>{});
            auto const c = eve::wide<int32_t>(1064866805);
            auto a = eve::bit_cast(y, eve::as<eve::wide<int32_t>>{}) * (xi - c) + c;
            return eve::bit_cast(a, eve::as<eve::wide<float>>{});
        };

        if constexpr (P == 0) {
            return eve::if_else(eve::is_nan(x) || eve::is_nan(y),
                nan,
                eve::if_else(eve::is_eqz(x),
                    eve::if_else(y < 0, inf, x),
                    eve::if_else(x < 0,
                        nan,
                        eve::if_else(eve::is_eqz(y),
                            1.F,
                            eve::if_else(y < lower_limit,
                                0.F,
                                eve::if_else(y > upper_limit,
                                    inf,
                                    pow_v1(x, y)
                                )
                            )
                        )
                    )
                )
            );
        } else {
            static_assert(P <= 1, "invalid precision spec for pow_impl");

            auto log2 = [](eve::wide<float> x) {
                auto i = eve::bit_cast(x, eve::as<eve::wide<uint32_t>>{});
                auto f = eve::bit_cast((i & 0x007FFFFF) | 0x3F000000, eve::as<eve::wide<float>>{});
                auto m = eve::bit_cast(i * 1.1920928955078125e-7F, eve::as<eve::wide<float>>{});
                return m - 124.22551499F - 1.498030302F * f - 1.72587999F / (0.3520887068F + f);
            };

            auto pow2 = [](eve::wide<float> x) {
                auto offset = eve::if_else(x < 0, eve::wide<float>{1.0F}, eve::wide<float>{0.0F});
                auto clipp  = eve::if_else(x < -126.F, -126.F, x);
                auto z = eve::floor(clipp) + offset;
                auto i = eve::bit_cast((1 << 23U) * (clipp + 121.2740575F + 27.7280233F / (4.84252568F - z) - 1.49012907F * z), eve::as<eve::wide<uint32_t>>{});
                return eve::bit_cast(i, eve::as<eve::wide<float>>{});
            };

            return pow2(y * log2(x));
        }
    }
} // namespace Operon::Backend::Mad
