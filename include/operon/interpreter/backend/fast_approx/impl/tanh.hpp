#ifndef OPERON_BACKEND_FAST_APPROX_TANH_HPP
#define OPERON_BACKEND_FAST_APPROX_TANH_HPP

#include "operon/core/types.hpp"
#include "inv.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr TanhImpl(Operon::Scalar x) -> Operon::Scalar {
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };
            if (std::isnan(x)) { return nan; }
            if (x == 0) { return 0; }

            if constexpr (P == 0) {
                if (x <= -3) {
                    return -1.F;
                }
                if (x >= +3) {
                    return +1.F;
                }
                auto const r = InvImpl<P>(x * x + 3.F);
                auto constexpr a = 8.F / 3.F;
                auto constexpr b = 1.F / 9.F;
                return x * (a * r + b);
            } else {
                if (x < -85) { return -1.F; }
                if (x > +85) { return +1.F; }

                auto expZeroShift = [](auto x) {
                    constexpr auto shift{23U};
                    constexpr auto ff{127U};
                    auto a = DivImpl<P>((1U << shift), std::numbers::ln2_v<Operon::Scalar>);
                    auto b = (ff * (1U << shift));
                    auto f = a * x + b;
                    auto i = static_cast<int32_t>(f);
                    return std::bit_cast<float>(i);
                };

                auto a = expZeroShift(x);
                auto b = expZeroShift(-x);
                return DivImpl<P>(a-b, a+b);
            }
        }
}  // namespace Operon::Backend::detail::fast_approx

#endif
