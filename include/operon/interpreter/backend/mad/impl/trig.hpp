#ifndef OPERON_BACKEND_MAD_TRIG_HPP
#define OPERON_BACKEND_MAD_TRIG_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::mad {

    /*
        Almost all logic for sine/cosine overlaps.

        See the following links for more details:
        1. http://tinyurl.com/2u8nvb94
        2. http://tinyurl.com/tv7byxmk

        Number of multiply-adds (MADDs):
            - Accuracy level 0: 4 MADDs
            - Accuracy level 1: 9 MADDs

        Note that we can implement `res * (1 - abs(res))`
        with one MADD, rather than two, by implementing
        `res - res ** 2` with some additional logic for
        the sign of `res * abs(res)`.
    */
    template<int P = 0>
    inline auto constexpr CosImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr auto inv_pi = std::numbers::inv_pi_v<float>;

        if constexpr (P == 0) {
            auto r = 0.5F * (x * inv_pi + 1.5F);
            r = 2 * (r - std::trunc(r)) + (r > 0 ? -1.F : +1.F);
            return 4 * r * (1 - std::abs(r));
        } else {
            // handle input special cases
            if (std::isnan(x) || std::isinf(x) || std::abs(x) > 33567376.0F) {
                return std::numeric_limits<float>::quiet_NaN();
            }

            constexpr auto DP1F = 0.78515625F;
            constexpr auto DP2F = 2.4187564849853515625e-4F;
            constexpr auto DP3F = 3.77489497744594108e-8F;

            auto a = std::abs(x);
            auto q = static_cast<int32_t>(a * (4 * inv_pi));
            q =  (q + 1) & (~1U);
            auto y = static_cast<float>(q);
            a = ((a - y * DP1F) - y * DP2F) - y * DP3F;
            q = q - 2;
            auto const sc = q & 4U; // sign of cosine
            auto const sp = q & 2U; // sign of polynomial
            auto const b = a * a;

            // compute sine or cosine depending on the polynomial mask
            auto r = sp == 0
                ? ((((-1.9515295891e-4F * b + 8.3321608736e-3F) * b - 1.6666654611e-1F) * b * a) + a)
                : (((2.443315711809948e-5F * b - 1.388731625493765e-3F) * b + 4.166664568298827e-2F) * b * b - 0.5F * b + 1.0F);

            // flip sign if necessary
            r *= (sc == 0 ? -1.F : +1.F);
            return r;
        }
    }

    /*
        Almost all logic for sine/cosine overlaps.

        See the following link for more details:
        http://tinyurl.com/2u8nvb94.

        Number of multiply-adds (MADDs):
            - Accuracy level 0: 4 MADDs
            - Accuracy level 1: 9 MADDs

        Note that we can multiply and divide by two by simply
        shifting the bits of the floating-point exponent.
        In addition, we implement `res * (1 - abs(res))`
        with one MADD, rather than two, by implementing
        `res - res ** 2` with some additional logic for
        the sign of `res * abs(res)`.
    */
    template<int P = 0>
    inline auto constexpr SinImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr auto inv_pi = std::numbers::inv_pi_v<float>;

        if constexpr (P == 0) {
            auto r = (x * inv_pi + 1) * 0.5F;
            r = 2 * (r - std::trunc(r)) + (r > 0 ? -1.F : +1.F);
            r = 4 * r * (1 - std::abs(r));
            return r;
        } else {
            if (std::isnan(x) || std::isinf(x) || std::abs(x) > 31875756.0F) {
                return std::numeric_limits<float>::quiet_NaN();
            }

            constexpr auto DP1F = 0.78515625F;
            constexpr auto DP2F = 2.4187564849853515625e-4F;
            constexpr auto DP3F = 3.77489497744594108e-8F;

            auto a = std::abs(x);
            auto q = static_cast<int32_t>(a * 4 * inv_pi);
            q = (q + 1) & (~1);
            auto y = static_cast<float>(q);
            a = ((a - y * DP1F) - y * DP2F) - y * DP3F;
            // sign of sine
            auto ss = q & 4;
            q -= 2;
            auto sp = q & 2;
            auto z = a * a;

            auto r = sp == 0
                ? (((2.443315711809948e-5F * z - 1.388731625493765e-3F) * z + 4.166664568298827e-2) * z * z - 0.5F * z + 1.F)
                : ((((-1.9515295891e-4F * z + 8.3321608736e-3F) * z - 1.6666654611e-1F) * z * a) + a);

            if (ss != 0) {
                r *= -1;
            }

            if (x < 0) {
                r *= -1;
            }

            return r;
        }
    }
} // namespace Operon::Backend::detail::mad

#endif
