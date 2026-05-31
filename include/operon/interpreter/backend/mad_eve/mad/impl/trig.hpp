#pragma once

#include "inv.hpp"
#include "exp.hpp"

namespace Operon::Backend::Mad {
    template<std::size_t P = 0>
    inline auto cos_impl(float x) -> float {
        if (!std::isfinite(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return 1.F; }

        if constexpr (P == 0) {
            auto inv_pi = std::numbers::inv_pi_v<float>;
            x = std::abs(x) * inv_pi + 1.5F;
            x = x - 2 * static_cast<float>(static_cast<int>(x/2)) - 1.F;
            return 4 * (x < 0 ? ((x*x) + x) : ((-x*x) + x));
        } else {
            constexpr float tp = std::numbers::inv_pi_v<float>/2;
            constexpr float a{.25F};
            constexpr float b{16.F};
            constexpr float c{.50F};
            constexpr float d{.225F};
            x *= tp;
            x -= a + std::floor(x + a);
            x *= b * (std::abs(x) - c);
            if constexpr (P >= 1) {
                x += d * x * (std::abs(x) - 1.F); // another step for extra precision
            }
            return x;
        }
    }

    template<std::size_t P = 0>
    inline auto sin_impl(float x) -> float {
        if (!std::isfinite(x)) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return x; }
        if constexpr (P == 0) {
            auto inv_pi = std::numbers::inv_pi_v<float>;
            auto const offset = x < 0 ? 2.F : 1.F;
            x = std::abs(x) * inv_pi + offset;
            x = x - 2 * static_cast<float>(static_cast<int>(x/2)) - 1.F;
            return 4 * (x < 0 ? ((x*x) + x) : ((-x*x) + x));
        } else {
            constexpr float tp = std::numbers::pi_v<float>/2.F;
            return cos_impl<P>(x-tp);
        }
    }

    template<std::size_t P = 0>
    inline auto tan_impl(float x) -> float {
        if (x == 0) { return x; }
        return div_impl<P>(sin_impl<P>(x), cos_impl<P>(x));
    }

    // eve implementations
    template<std::size_t P = 0, bool Check = true>
    auto cos_impl(eve::wide<float> x) -> eve::wide<float> {
        auto cos = [](eve::wide<float> x) {
            if constexpr (P == 0) {
                auto inv_pi = std::numbers::inv_pi_v<float>;
                x = x * inv_pi + 1.5F;
                x = x / 2;
                x = eve::if_else(x > 0,
                    2 * (x - eve::convert(eve::convert(x, eve::as<int32_t>{}), eve::as<float>{})) - 1.F,
                    2 * (x - eve::convert(eve::convert(x, eve::as<int32_t>{}), eve::as<float>{})) + 1.F
                );
                return 4 * (x * -eve::abs(x) + x);
            } else {
                constexpr float tp = std::numbers::inv_pi_v<float>/2;
                constexpr float a{.25F};
                constexpr float b{16.F};
                constexpr float c{.50F};
                constexpr float d{.225F};
                x *= tp;
                x -= a + eve::floor(x + a);
                x *= b * (eve::abs(x) - c);
                if constexpr (P >= 1) {
                    x += d * x * (eve::abs(x) - 1.F); // another step for extra precision
                }
                return x;
            }
        };

        if constexpr (Check) {
            auto const nan = eve::nan(eve::as<eve::wide<float>>{});

            return eve::if_else(eve::is_infinite(x),
                nan,
                eve::if_else(x == 0,
                    eve::one(eve::as<eve::wide<float>>{}),
                    cos(x)
                    )
            );
        } else {
            return cos(x);
        }
    }

    template<std::size_t P = 0, bool Check = true>
    auto sin_impl(eve::wide<float> x) -> eve::wide<float> {
        auto sin = [](eve::wide<float> x) -> eve::wide<float> {
            if constexpr (P == 0) {
                auto inv_pi = std::numbers::inv_pi_v<float>;
                x = x * inv_pi + 1.0F;
                x = x / 2;
                x = eve::if_else(x > 0,
                    2 * (x - eve::convert(eve::convert(x, eve::as<int32_t>{}), eve::as<float>{})) - 1.F,
                    2 * (x - eve::convert(eve::convert(x, eve::as<int32_t>{}), eve::as<float>{})) + 1.F
                );
                return 4 * (x * -eve::abs(x) + x);
            } else {
                constexpr float tp = std::numbers::pi_v<float>/2.F;
                return cos_impl<P>(x-tp);
            }
        };

        if constexpr (Check) {
            auto const nan = eve::nan(eve::as<eve::wide<float>>{});

            return eve::if_else(eve::is_infinite(x),
                nan,
                eve::if_else(eve::is_eqz(x),
                    x,
                    sin(x)
                )
            );
        } else {
            return sin(x);
        }
    }

    template<std::size_t P = 0, bool Check = true>
    auto tan_impl(eve::wide<float> x) -> eve::wide<float> {
        return eve::if_else(x == 0, x,
            div_impl<P, Check>(sin_impl<P, Check>(x), cos_impl<P, Check>(x))
        );
    }
}  // namespace Operon::Backend::Mad
