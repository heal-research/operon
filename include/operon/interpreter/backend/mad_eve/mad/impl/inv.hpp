#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <eve/module/core.hpp>
#include <eve/wide.hpp>

namespace Operon::Backend::Mad {
/*
    See the following link for more details:
    https://bit.ly/42qbEHG.

    Number of multiply-adds (MADDs):
        - Accuracy level i (i >= 0): 2 * i MADDs
*/
template<std::size_t P = 0, bool Check = false>
auto inv_impl(float x) -> float {
    if constexpr (Check) {
        if (std::isnan(x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (x == 0.F) {
            return std::signbit(x)
                ? -std::numeric_limits<float>::infinity()
                : +std::numeric_limits<float>::infinity();
        }
        if (std::isinf(x) || std::abs(x) > 1.602756e38F) {
            return 0.F;
        }
    }

    auto const a = std::abs(x);
    auto const b = 0x7EF127EA - std::bit_cast<uint32_t>(a);
    auto f       = std::bit_cast<float>(b);

    if constexpr (P > 0) {
        for (auto i = 0; i < P; ++i) {
            f = f * (2 - f * a);
        }
    }
    return x < 0 ? -f : +f;
}

template<std::size_t P = 0>
auto div_impl(float x, float y) -> float {
    constexpr auto nan{ std::numeric_limits<float>::quiet_NaN() };
    constexpr auto inf{ std::numeric_limits<float>::infinity() };
    constexpr auto max{ 1.602756e38F };
    if (y == 0) {
        return x == 0 ? nan : (inf * std::copysign(x, y));
    }
    if ((std::isnan(x) || std::isnan(y)) || (std::isinf(x) && std::abs(y) > max)) {
        return nan;
    }
    return x * inv_impl<P>(y);
}

// inv(x)_x=0 = inf * sign(x)
// inv(nan) = nan
// inv(inf) = 0
template<std::size_t P = 0, bool Check = true>
auto inv_impl(eve::wide<float> x) -> eve::wide<float> {
    auto inv = [](eve::wide<float> x) {
        auto const a = eve::abs(x);
        auto const b = eve::wide<uint32_t>{0x7EF127EA} - eve::bit_cast(a, eve::as<eve::wide<uint32_t>>{});
        auto f       = eve::bit_cast(b, eve::as<eve::wide<float>>{});

        if constexpr (P > 0) {
            for (auto i = 0UL; i < P; ++i) {
                f = f * (2 - f * a);
            }
        }
        return eve::if_else(x < 0, -f, +f);
    };

    auto const inf = eve::inf(eve::as<eve::wide<float>>());
    auto const zero = eve::zero(eve::as<eve::wide<float>>());

    auto const max = eve::wide<float>{1.602756e38F};

    if constexpr (Check) {
        return eve::if_else(eve::is_eqz(x),
            eve::signnz(x) * inf,
            eve::if_else(eve::is_infinite(x) || eve::abs(x) > max,
                eve::signnz(x) * zero,
                inv(x)
            )
        );
    } else {
        return inv(x);
    }
}

template<std::size_t P = 0, bool Check = true>
auto div_impl(eve::wide<float> x, eve::wide<float> y) -> eve::wide<float> {
    auto const nan = eve::nan(eve::as<eve::wide<float>>());
    auto const inf = eve::inf(eve::as<eve::wide<float>>());
    auto const max = eve::wide<float>{1.602756E38F};

    auto div = [](eve::wide<float> x, eve::wide<float> y) {
        return x * inv_impl<P, Check>(y);
    };

    if constexpr (Check) {
        return eve::if_else(eve::is_eqz(y),
            eve::if_else(eve::is_eqz(x), nan, inf * eve::copysign(x, y)),
            eve::if_else(eve::is_nan(x) || eve::is_nan(y) || (eve::is_infinite(x) && eve::abs(y) > max),
                nan,
                div(x, y)
            )
        );
    } else {
        return div(x, y);
    }
}
}  // namespace Operon::Backend::Mad
