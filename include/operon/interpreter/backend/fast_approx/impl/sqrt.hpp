#ifndef OPERON_BACKEND_FAST_APPROX_SQRT_HPP
#define OPERON_BACKEND_FAST_APPROX_SQRT_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr ISqrtImpl(Operon::Scalar x) -> Operon::Scalar {
        static_assert(std::is_same_v<Operon::Scalar, float>, "this function only works in single-precision mode.");
        constexpr auto nan{ std::numeric_limits<Operon::Scalar>::quiet_NaN() };
        constexpr auto inf{ std::numeric_limits<Operon::Scalar>::infinity() };
        constexpr auto fast_sqrt_constant{static_cast<float>(0x5F3759DF)};

        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
        if (x == -0) { return -inf; }
        if (x == +0) { return +inf; }

        auto xt = x * 0.5F;
        auto xi = std::bit_cast<int32_t>(x);
        xi = fast_sqrt_constant - (xi >> 1U);
        auto xf = std::bit_cast<float>(xi);
        for (auto i = 0UL; i < P; ++i) {
            xf = xf * (float{1.5} - (xt * (xf * xf)));
        }
        return xf;
    }

    template<std::size_t P = 0>
    inline auto constexpr SqrtImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        if (std::isnan(x)) { return nan; }
        if (x < 0) { return nan; }
        if (x == 0) { return 0; }
        if constexpr (P == 0) {
            auto xi = std::bit_cast<int32_t>(x);
            xi -= 0x3F800000;
            xi = ((((xi >> 31U) & 1) << 31U) | ((xi >> 1U) & 0x7FFFFFFF));
            xi += 0x3F800000;
            xi &= 0xFFFFFFFF;
            return std::bit_cast<float>(xi);
        } else {
            return x * ISqrtImpl<P>(x);
        }
    }

    template<std::size_t P = 0>
    inline auto constexpr SqrtabsImpl(Operon::Scalar x) {
        return SqrtImpl<P>(std::abs(x));
    }
}  // namespace Operon::Backend::detail::fast_approx

#endif
