#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

#include <eve/module/math/constant/log2_e.hpp>
#include <eve/wide.hpp>
#include <eve/module/core.hpp>

namespace Operon::Backend::Mad {
    template<std::size_t P = 0>
    inline auto log_impl(float x) -> float {
        constexpr auto inf { std::numeric_limits<float>::infinity() };
        constexpr auto nan { std::numeric_limits<float>::quiet_NaN() };
        constexpr auto ln2 { std::numbers::ln2_v<float> };

        if (std::isnan(x) || x < 0) { return nan; }
        if (x == 0) { return -inf; }
        if (x == 1) { return 0.F; }
        if (x == inf) { return inf; }
        if constexpr (P == 0) {
            return (std::bit_cast<int32_t>(x) - 1065353217) * 8.262958405176314e-8F;
        } else {
            auto bx = std::bit_cast<std::uint32_t>(x);
            auto ex = bx >> 23U;
            auto t = static_cast<std::int32_t>(ex) - static_cast<std::int32_t>(127);
            bx = 1065353216 | (bx & 8388607);
            x = std::bit_cast<float>(bx);
            if constexpr (P == 1) {
                return -1.49278F+((2.11263F+(-0.729104F+0.10969F*x)*x)*x)+(ln2 * t);
            } else {
                return -1.7417939F+((2.8212026F+(-1.4699568F+(0.44717955F-0.056570851F*x)*x)*x)*x)+(ln2 * t);
            }
        }
    }

    template<std::size_t P = 0>
    auto log1p_impl(float x) -> float  {
        return log_impl<P>(1 + x);
    }

    template<std::size_t P = 0>
    inline auto logabs_impl(float x) -> float  {
        return log_impl<P>(std::abs(x));
    }

    template<std::size_t P = 0, bool Check = true>
    auto log_impl(eve::wide<float> x) -> eve::wide<float> {
        auto const inf = eve::inf(eve::as<eve::wide<float>>{});
        auto const nan = eve::nan(eve::as<eve::wide<float>>{});

        auto log = [](eve::wide<float> x) {
            if constexpr (P == 0) {
                return (eve::bit_cast(x, eve::as<eve::wide<int32_t>>{}) - 1065353217) * 8.262958405176314e-8F;
            } else {
                // auto const ln2 = eve::wide<float>{std::numbers::ln2_v<float>};
                auto const ln2 = 0.6931471806;
                auto bx = eve::bit_cast(x, eve::as<eve::wide<int32_t>>{});
                auto e  = eve::convert((eve::convert(bx >> 23U, eve::as<int32_t>{}) - eve::wide<int32_t>(127)), eve::as<float>{});
                auto t = ln2 * e;
                auto m = (0x7F << 23) | (bx & 0x7FFFFF);
                auto f  = eve::bit_cast(m, eve::as<eve::wide<float>>{});
                if constexpr (P == 1) {
                    return (t + ((-0.239030721F * f + 1.40339138F) * f - 1.16093668F));
                } else {
                    static_assert(P <= 2, "error: invalid precision spec for log_impl");
                    return -1.7417939F+((2.821226F+(-1.4699568F+(0.44717955F-0.056570851F*f)*f)*f)*f)+t;
                }
            }
        };

        if constexpr (Check) {
            return eve::if_else(eve::is_nan(x) || x < 0,
                nan,
                eve::if_else(eve::is_eqz(x),
                    -inf,
                    eve::if_else(x == 1,
                        0.F,
                        eve::if_else(eve::is_infinite(x),
                            inf,
                            log(x)
                        )
                    )
                )
            );
        } else {
            return log(x);
        }
    }

    template<std::size_t P = 0, bool Check = true>
    auto log1p_impl(eve::wide<float> x) -> eve::wide<float>  {
        return log_impl<P, Check>(1 + x);
    }

    template<std::size_t P = 0, bool Check = true>
    auto logabs_impl(eve::wide<float> x) -> eve::wide<float>  {
        return log_impl<P, Check>(eve::abs(x));
    }
}  // namespace Operon::Backend::Mad
