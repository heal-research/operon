#ifndef OPERON_BACKEND_MAD_SQRT_HPP
#define OPERON_BACKEND_MAD_SQRT_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::mad {

    /*
        See the following link for more details:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root.

        Number of multiply-adds (MADDs):
            - Accuracy level i (i >= 0): 3 * i MADDs
    */
    template<int P = 0>
    inline auto constexpr ISqrtImpl(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x) && x < 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        if (x == 0 && !std::signbit(x)) {
            return std::numeric_limits<float>::infinity();
        }

        if (std::isinf(x) && !std::signbit(x)) {
            return 0.F;
        }

        auto h = 0.5F * x;

        // approximation of `x ** (-0.5)`.
        auto r = 0x5F3759DF - (std::bit_cast<int32_t>(x) >> 1);

        // convert approximation back to floating-point.
        auto f = std::bit_cast<float>(r);

        for (auto i = 0; i < P; ++i) {
            f *= 1.5F - h * f * f;
        }
        return f;
    }

    template<int P = 0>
    inline auto constexpr SqrtImpl(Operon::Scalar x) -> Operon::Scalar {
        if (x < 0) { return std::numeric_limits<float>::quiet_NaN(); }
        if (x == 0) { return x; }
        if (std::isnan(x)) { return x; }
        if (std::isinf(x)) { return x; }
        auto y = ISqrtImpl<P>(x);
        return x * y;
    }

    template<int P = 0>
    inline auto constexpr SqrtabsImpl(Operon::Scalar x) -> Operon::Scalar {
        return SqrtImpl<P>(std::abs(x));
    }
} // namespace Operon::Backend::detail::mad

#endif
