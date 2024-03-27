#ifndef OPERON_BACKEND_MAD_TANH_HPP
#define OPERON_BACKEND_MAD_TANH_HPP

#include "operon/core/types.hpp"
#include "inv.hpp"
#include "exp.hpp"

namespace Operon::Backend::detail::mad {

    /*
        See the following link for more details:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root.

        Number of multiply-adds (MADDs):
            - Accuracy level i (i >= 0): 3 * i MADDs
    */
    template<int P = 0>
    inline auto constexpr TanhImpl(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if constexpr (P <= 1) {
            constexpr auto c{3.F};
            if (x != std::clamp(x, -c, +c)) {
                // x is outside the approximation range
                return x > 0 ? +1.F : -1.F;
            }
            constexpr auto a{8.F/3.F};
            constexpr auto b{1.F/9.F};
            return x * (a * InvImpl<P>(x * x + 3) + b);
        } else {
            constexpr auto c{32.F};
            if (x != std::clamp(x, -c, +c)) {
                // x is outside the approximation range
                return x > 0 ? +1.F : -1.F;
            }
            constexpr auto E = P == 2 ? 0 : 3;
            return 1 - 2 * (InvImpl<3>(ExpImpl<E>(2 * x) + 1));
        }
    }
} // namespace Operon::Backend::detail::mad

#endif
