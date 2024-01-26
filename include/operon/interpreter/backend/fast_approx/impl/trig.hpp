#ifndef OPERON_BACKEND_FAST_APPROX_TRIG_HPP
#define OPERON_BACKEND_FAST_APPROX_TRIG_HPP

#include "operon/core/types.hpp"
#include "inv.hpp"
#include "exp.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr CosImpl(Operon::Scalar x)
    {
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

    template<std::size_t P = 0>
    inline auto constexpr SinImpl(Operon::Scalar x) -> Operon::Scalar {
        constexpr float tp = std::numbers::pi_v<float>/2;
        return CosImpl<P>(x - tp);
    }

    template<std::size_t P = 0>
    inline auto constexpr TanImpl(Operon::Scalar x) -> Operon::Scalar {
        return DivImpl<P>(SinImpl<P>(x), CosImpl<P>(x));
    }
}  // namespace Operon::Backend::detail::fast_approx

#endif