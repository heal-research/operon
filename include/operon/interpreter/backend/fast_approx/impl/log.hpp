#ifndef OPERON_BACKEND_FAST_APPROX_LOG_HPP
#define OPERON_BACKEND_FAST_APPROX_LOG_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::fast_approx {
    inline auto constexpr LogImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr auto inf { std::numeric_limits<Operon::Scalar>::infinity() };
        constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
        if (x == 0) { return -inf; }
        if (x == 1) { return 0.F; }
        return (std::bit_cast<int32_t>(x) - 1065353217) * 8.262958405176314e-8F;
    }

    inline auto constexpr Log1pImpl(Operon::Scalar x) -> Operon::Scalar  {
        return LogImpl(1 + x);
    }

    inline auto constexpr LogabsImpl(Operon::Scalar x) -> Operon::Scalar  {
        return LogImpl(std::abs(x));
    }
}  // namespace Operon::Backend::detail::fast_approx
#endif