#ifndef OPERON_BACKEND_FAST_APPROX_LOG_HPP
#define OPERON_BACKEND_FAST_APPROX_LOG_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr LogImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr auto inf { std::numeric_limits<Operon::Scalar>::infinity() };
        constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
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
                return -1.49278+(2.11263+(-0.729104+0.10969*x)*x)*x+0.6931471806*t;
            } else {
                return -1.7417939+(2.8212026+(-1.4699568+(0.44717955-0.056570851*x)*x)*x)*x+0.6931471806*t;
            }
        }
    }

    template<std::size_t P = 0>
    inline auto constexpr Log1pImpl(Operon::Scalar x) -> Operon::Scalar  {
        return LogImpl<P>(1 + x);
    }

    template<std::size_t P = 0>
    inline auto constexpr LogabsImpl(Operon::Scalar x) -> Operon::Scalar  {
        return LogImpl<P>(std::abs(x));
    }
}  // namespace Operon::Backend::detail::fast_approx
#endif
