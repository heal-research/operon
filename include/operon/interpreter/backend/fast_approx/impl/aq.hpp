#ifndef OPERON_BACKEND_FAST_APPROX_AQ_HPP
#define OPERON_BACKEND_FAST_APPROX_AQ_HPP

#include "operon/core/types.hpp"
#include "inv.hpp"
#include "sqrt.hpp"

namespace Operon::Backend::detail::fast_approx {
    template<std::size_t P = 0>
    inline auto constexpr AqImpl(Operon::Scalar x1, Operon::Scalar x2) {
        auto constexpr p{9999999980506447872.F};
        return std::abs(x2) > p ? DivImpl<P>(x1, std::abs(x2)) : x1 * ISqrtImpl<P>(1 + x2*x2);
    }
} // namespace Operon::Backend::detail::fast_approx

#endif