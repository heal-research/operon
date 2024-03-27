#ifndef OPERON_BACKEND_MAD_LOG_HPP
#define OPERON_BACKEND_MAD_LOG_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::mad {

    /*
        See the following link for more details:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root.

        Number of multiply-adds (MADDs):
            - Accuracy level i (i >= 0): 3 * i MADDs
    */
    template<int P = 0>
    inline auto constexpr LogImpl(Operon::Scalar x) -> Operon::Scalar {
        if (x == 1) {
            return 0.F;
        }
        if (std::isnan(x) || x < 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (std::isinf(x) && !std::signbit(x)) {
            return std::numeric_limits<float>::infinity();
        }
        if (x == 0 && std::signbit(x)) {
            return -std::numeric_limits<float>::infinity();
        }

        if constexpr (P == 0) {
            auto i = std::bit_cast<int32_t>(x);
            return (i - 1065353217) * 8.262958405176314e-8F;
        } else if constexpr (P <= 4) {
            constexpr auto log2 = std::numbers::ln2_v<float>;
            auto i = std::bit_cast<uint32_t>(x);
            auto e = static_cast<float>(static_cast<int32_t>(i >> 23) - 127);
            auto r = log2 * e;
            auto m = std::bit_cast<float>((0x7F << 23) | (i & 0x7FFFFF));

            if constexpr (P == 1) {
                // degree-2 polynomial
                r += (-0.239030721F * m + 1.40339138F) * m - 1.16093668F;
            } else if constexpr (P == 2) {
                // degree-5 polynomial
                r += (((((0.0308913737F * m - 0.287210575F) * m + 1.12631109F) * m - 2.45526047F)  * m + 3.52527699F) * m - 1.94000044F);
            } else if constexpr (P == 3) {
                // degree-6 polynomial
                r += ((((((-0.0170793339F * m + 0.184865198F) * m - 0.859215646F) * m + 2.24670209F) * m - 3.67519809F) * m + 4.22523802F) * m - 2.10531206F);
            } else if constexpr (P == 4) {
                // degree-7 polynomial (error gets worse... perhaps more rounding error?)
                r += (((((((0.0102893135F * m - 0.125467594F) * m + 0.669001960F) * m - 2.04733924F) * m + 3.97620707F) * m - 5.16804080F)* m + 4.93256353F) * m - 2.24721410F);
            }
            return r;
        } else if constexpr (P == 5) {
            // Perform approximation given by the VDT library:
            // https://tinyurl.com/4aex3p2k.
            //
            // Separate the input `x` into a mantissa `m` and an exponent
            // `e` such that `0.5 <= abs(m) < 1.0`, and x = m * (2 ** e).
            // (An input of zero is a special condition for log.)
            // From this, we compute `log(x)` using the equality
            // `log(x) = log(m) + log_e(2) * e`, where we use
            // a polynomial approximation to compute `log(m)`.
            auto m = std::bit_cast<int32_t>(x);
            // Initial value of exponent.
            auto e = static_cast<int32_t>((m >> 23) - 127);
            // Desired representation of mantissa.
            // (We force the exponent to be equal to
            // a value of 0.5 and copy the original mantissa.)
            auto p = std::bit_cast<float>((m & 0x807FFFFF) | 0x3F000000);
            // We alter the exponent/mantissa value to meet the definition given above.
            constexpr auto sqrthf = 0.707106781186547524F;
            if (p > sqrthf) {
                e += 1;
                p -= 1;
            } else {
                p = 2 * p - 1;
            }
            auto f = static_cast<float>(e);
            // Now, we approximate `log(m)` with a polynomial.
            auto p2 = p * p;
            auto r = ((((((((7.0376836292e-2F * p + -1.1514610310e-1F) * p + 1.1676998740e-1F) * p + -1.2420140846e-1F) * p + 1.4249322787e-1F) * p + -1.6668057665e-1F) * p + 2.0000714765e-1F) * p + -2.4999993993e-1F) * p + 3.3333331174e-1F);
            r *= p2;
            r *= p;
            r += -2.12194440e-4F * f;
            r += -0.5F * p2;
            r += p;
            r += 0.693359375F * f;
            return r;
        }
    }

    template<int P = 0>
    inline auto constexpr Log1pImpl(Operon::Scalar x) -> Operon::Scalar {
        return LogImpl<P>(x+1.F);
    }

    template<int P = 0>
    inline auto constexpr LogabsImpl(Operon::Scalar x) -> Operon::Scalar {
        return LogImpl<P>(std::abs(x));
    }
} // namespace Operon::Backend::detail::mad

#endif