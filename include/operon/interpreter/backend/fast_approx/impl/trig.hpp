#ifndef OPERON_BACKEND_FAST_APPROX_TRIG_HPP
#define OPERON_BACKEND_FAST_APPROX_TRIG_HPP

#include "operon/core/types.hpp"
#include "inv.hpp"
#include "exp.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr CosImpl(Operon::Scalar x) -> Operon::Scalar {
        if (!std::isfinite(x)) { return std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (x == 0) { return 1.F; }

        if constexpr (P == 0) {
            auto constexpr invPi = std::numbers::inv_pi_v<float>;
            x = std::abs(x) * invPi + 1.5F;
            x = x - 2 * static_cast<float>(static_cast<int>(x/2)) - 1.F;
            return 4 * (x < 0 ? (x*x + x) : (-x*x + x));
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
    inline auto constexpr SinImpl(Operon::Scalar x) -> Operon::Scalar {
        if (!std::isfinite(x)) { return std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (x == 0) { return x; }
        if constexpr (P == 0) {
            auto constexpr invPi = std::numbers::inv_pi_v<float>;
            auto const offset = x < 0 ? 2.F : 1.F;
            x = std::abs(x) * invPi + offset;
            x = x - 2 * static_cast<float>(static_cast<int>(x/2)) - 1.F;
            return 4 * (x < 0 ? (x*x + x) : (-x*x + x));
        } else {
            constexpr float tp = std::numbers::pi_v<float>/2;
            return CosImpl<P>(x-tp);
        }
    }

    template<std::size_t P = 0>
    inline auto constexpr TanImpl(Operon::Scalar x) -> Operon::Scalar {
        if (x == 0) { return x; }
        return DivImpl<P>(SinImpl<P>(x), CosImpl<P>(x));
    }
}  // namespace Operon::Backend::detail::fast_approx

#endif
