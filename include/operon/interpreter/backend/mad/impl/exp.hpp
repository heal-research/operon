#ifndef OPERON_BACKEND_MAD_EXP_HPP
#define OPERON_BACKEND_MAD_EXP_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::mad {

    /*
        See the following links for more details:
        1. https://bit.ly/3NmkzWu
        2. https://bit.ly/3ChoAoC
        3. http://tinyurl.com/3cvwajck
        4. https://tinyurl.com/2vbdcvuc

        Number of multiply-adds (MADDs):
            - Accuracy level 0: 1 MADDs
            - Accuracy level 1: 4 MADDs
            - Accuracy level 2: 7 MADDs
            - Accuracy level 3: 8 MADDs
            - Accuracy level 4: 9 MADDs
            - Accuracy level 5: 15 MADDs
    */
    template<int P = 0>
    inline auto constexpr ExpImpl(Operon::Scalar x) -> Operon::Scalar {
        if (x == 0) {
            return 1.F;
        }

        if (std::isnan(x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        if (x < -88) {
            return 0.F;
        }
        if (x > 88.72283905206835F) {
            return std::numeric_limits<float>::infinity();
        }

        if constexpr (P == 0) {
            auto r = x * 12102203.161561485F + 1065054451.F;
            auto i = static_cast<int32_t>(r);
            return std::bit_cast<float>(i);
        } else if constexpr (P >= 1 && P <= 4) {
            /*
                We compute `exp(x)` by way of the equivalent formulation
                `2 ** (x / log(2)) \approx 2 ** (x * 1.44269504)`.

                Effectively, we split `t = x * 1.44269504` into an
                integer `i` and fraction `f` such that `t = i + f`
                and `0 <= f <= 1`. From this, we can compute
                `2 ** (x * 1.44269504) = (2 ** f) * (2 ** i)` by first
                approximating `2 ** f` with a polynomial, and then by
                scaling with `2 ** i`, the latter of which can be
                performed simply by adding `i` to the exponent of the
                result of `2 ** f`.
                            To compute `i` (and then `f`) as described above, one
                could compute `floor(t)`, which is equal to `trunc(t)`
                when `t > 0` or when `t` is a negative *integer*, and
                equal to `trunc(t) - 1` otherwise. We avoid directly
                computing `floor` by instead using an extension of
                Schraudolph's algorithm: http://tinyurl.com/3cvwajck.
                This allows for more efficient logic in hardware.
                            `INV_LOG2_SHIFTED = (1 << 23) / log(2)`.
            */
            constexpr auto INV_LOG2_SHIFTED = 12102203.0F;
            constexpr auto EXP2_NEG_23 = 1.1920929e-7F;
            auto t = static_cast<int32_t>(INV_LOG2_SHIFTED * x);
            auto j = static_cast<int32_t>(t & 0xFF800000);
            auto f = EXP2_NEG_23 * static_cast<float>(t - j);

            auto xf = 0.F;
            if constexpr (P == 1) {
                xf = (0.3371894346F * f + 0.657636276F) * f + 1.00172476F;
            } else if constexpr (P == 2) {
                // degree-5 polynomial
                xf = (((((0.00189268149F * f + 0.00895538940F) * f + 0.0558525427F) * f + 0.240145453F) * f + 0.693153934F) * f + 0.999999917F);
            } else if constexpr (P == 3) {
                // degree-6 polynomial
                xf = ((((((0.000221577741F * f + 0.00122991652F) * f + 0.00969518396F) * f + 0.0554745401F) * f + 0.240231977F) * f + 0.693146803F) * f + 1.F);
            } else if constexpr (P == 4) {
                // degree-7 polynomial
                xf = (((((((0.0000217349529F * f + 0.000142668753F) * f + 0.00134347152F) * f + 0.00961318205F) * f + 0.0555054119F) * f + 0.240226344F) * f + 0.693147187F) * f + 1.F);
            }
            return std::bit_cast<float>(j + std::bit_cast<int32_t>(xf));
        } else if constexpr (P == 5) {
            constexpr auto LOG2_E = std::numbers::log2e_v<float>;
            auto a = x;
            auto z = std::floor(LOG2_E * a + 0.5F);
            a = a - 0.693359375F * z;
            a = a + 2.12194440e-4F * z;
            auto n = static_cast<int32_t>(z);
            n = (n + 127) << 23U;
            auto a2 = a * a;
            auto r = (((((1.9875691500e-4F * a + 1.3981999507e-3F) * a + 8.3334519073e-3F) * a + 4.1665795894e-2F) * a + 1.6666665459e-1F) * a + 5.0000001201e-1F);
            r = r * a2 + a;
            r = r + 1;
            r = r * std::bit_cast<float>(n);
            return r;
        }
    }
} // namespace Operon::Backend::detail::mad

#endif