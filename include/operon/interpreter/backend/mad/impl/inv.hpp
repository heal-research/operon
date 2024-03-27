#ifndef OPERON_BACKEND_MAD_INV_HPP
#define OPERON_BACKEND_MAD_INV_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::mad {
    /*
        See the following link for more details:
        https://bit.ly/42qbEHG.

        Number of multiply-adds (MADDs):
            - Accuracy level i (i >= 0): 2 * i MADDs
    */
    template<int P = 0>
    inline auto constexpr InvImpl(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (x == 0) {
            return std::signbit(x)
                ? -std::numeric_limits<float>::infinity()
                : +std::numeric_limits<float>::infinity();
        }
        if (std::isinf(x) || std::abs(x) > 1.602756e38F) {
            return 0.F;
        }

        auto a = std::abs(x);
        auto b = static_cast<int32_t>(0x7EF127EA - std::bit_cast<int32_t>(a));
        auto f = std::bit_cast<float>(b);

        for (auto i = 0; i < P; ++i) {
            f = f * (2 - f * a);
        }
        return x < 0 ? -f : +f;
    }

    template<int P = 0>
    inline auto constexpr DivImpl(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
        if (x == 0 && y == 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (std::isnan(x) || std::isnan(y)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (std::isinf(x) && std::abs(y) > 1.602756e38F) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (y == 0) {
            auto sx = std::signbit(x);
            auto sy = std::signbit(y);
            // if the signbits are the same return +inf otherwise return -inf
            if ((sx && sy) || (!sx && !sy)) { return +std::numeric_limits<float>::infinity(); }
            return -std::numeric_limits<float>::infinity();
        }
        return x * InvImpl<P>(y);
    }
} // namespace Operon::Backend::detail::mad

#endif