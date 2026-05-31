#pragma once

#include <numbers>
#include "inv.hpp"

namespace Operon::Backend::Mad {
    template<std::size_t P = 0>
    auto tanh_impl(float x) -> float {
        constexpr auto lower_limit{-85.F};
        constexpr auto upper_limit{+85.F};
        constexpr auto nan { std::numeric_limits<float>::quiet_NaN() };

        if (std::isnan(x)) { return nan; }
        if (x == 0) { return 0; }

        if constexpr (P == 0) {
            if (x <= -3) {
                return -1.F;
            }
            if (x >= +3) {
                return +1.F;
            }
            auto const r = inv_impl<P>((x * x) + 3.F);
            auto a = 8.F / 3.F;
            auto b = 1.F / 9.F;
            return x * (a * r + b);
        } else {
            if (x < lower_limit) { return -1.F; }
            if (x > upper_limit) { return +1.F; }

            auto exp_zero_shift = [](auto x) {
                constexpr auto shift{23U};
                constexpr auto ff{127U};
                auto a = div_impl<P>((1U << shift), std::numbers::ln2_v<float>);
                auto b = (ff * (1U << shift));
                auto f = (a * x) + b;
                auto i = static_cast<int32_t>(f);
                return std::bit_cast<float>(i);
            };

            auto a = exp_zero_shift(x);
            auto b = exp_zero_shift(-x);
            return div_impl<P>(a-b, a+b);
        }
    }

    template<std::size_t P = 0, bool Check = true>
    auto tanh_impl(eve::wide<float> x) {
        auto const nan{eve::nan(eve::as<eve::wide<float>>{})};

        auto tanh = [](eve::wide<float> x) {
            auto const lower_limit{eve::wide{-85.F}};
            auto const upper_limit{eve::wide{+85.F}};

            if constexpr (P == 0) {
                return eve::if_else(x <= -3 || x >= 3,
                    eve::signnz(x) * eve::one(eve::as<eve::wide<float>>{}),
                    [](eve::wide<float> x) {
                        auto const r = inv_impl<P>(x * x + 3.F);
                        auto const a = eve::wide<float>(8.F / 3.F);
                        auto const b = eve::wide<float>(1.F / 9.F);
                        return x * (a * r + b);
                    }(x)
                );
            } else {
                static_assert(P <= 1, "invalid precision spec for tanh_impl");

                auto exp_zero_shift = [](eve::wide<float> x) {
                    constexpr auto shift{23U};
                    constexpr auto ff{127U};
                    auto a = div_impl<P>(eve::wide<float>{1U << shift}, eve::wide<float>{std::numbers::ln2_v<float>});
                    auto b = (ff * (1U << shift));
                    auto f = (a * x) + b;
                    auto i = eve::convert(f, eve::as<int32_t>{});
                    return eve::bit_cast(i, eve::as<eve::wide<float>>{});
                };

                return eve::if_else(x < lower_limit || x > upper_limit,
                    eve::signnz(x) * eve::one(eve::as<eve::wide<float>>{}),
                    [&](eve::wide<float> x) {
                        auto a = exp_zero_shift(x);
                        auto b = exp_zero_shift(-x);
                        return div_impl<P>(a-b, a+b);
                    }(x)
                );
            }
        };

        if constexpr (Check) {
            return eve::if_else(eve::is_nan(x),
                nan,
                eve::if_else(x == 0,
                    eve::zero(eve::as<eve::wide<float>>{}),
                    tanh(x)
                )
            );
        } else {
            return tanh(x);
        }
    }
}  // namespace Operon::Backend::Mad
