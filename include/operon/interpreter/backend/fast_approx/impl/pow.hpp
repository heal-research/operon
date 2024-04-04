#ifndef OPERON_BACKEND_FAST_APPROX_POW_HPP
#define OPERON_BACKEND_FAST_APPROX_POW_HPP

#include "inv.hpp"
#include "exp.hpp"
#include "log.hpp"

namespace Operon::Backend::detail::fast_approx {
    inline auto constexpr PowV1(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            constexpr auto inf { std::numeric_limits<Operon::Scalar>::infinity() };
            constexpr auto nan { std::numeric_limits<Operon::Scalar>::quiet_NaN() };

            if (std::isnan(x)) { return nan; }
            if (std::isnan(y)) { return nan; }
            if (x == 0) { return y < 0 ? inf : x; }
            if (x < 0) { return nan; }
            if (y == 0) { return 1.F; }
            if (y < -85.F) { return 0; }
            if (y > 85.F) { return inf; }

            auto xi = std::bit_cast<int32_t>(x);
            auto a = static_cast<int>(y * (xi - 1064866805) + static_cast<float>(1064866805));
            return std::bit_cast<float>(a);
        }

        inline auto PowV2(Operon::Scalar x, Operon::Scalar y) {
            auto log2 = [](Operon::Scalar x) {
                auto i = std::bit_cast<std::uint32_t>(x);
                auto f = std::bit_cast<float>((i & 0x007FFFFF) | 0x3f000000);
                auto y = i * 1.1920928955078125e-7f;
                return y - 124.22551499f - 1.498030302f * f - 1.72587999f / (0.3520887068f + f);
            };

            auto pow2 = [](Operon::Scalar p) {
                float offset = (p < 0) ? 1.0f : 0.0f;
                float clipp = (p < -126) ? -126.0f : p;
                float z = clipp - static_cast<int32_t>(clipp) + offset;
                auto i = std::uint32_t((1 << 23U) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z));
                return std::bit_cast<float>(i);
            };
            return pow2(y * log2(x));
        }

        template<std::size_t P = 0>
        inline auto PowImpl(Operon::Scalar x, Operon::Scalar y) -> Operon::Scalar {
            if constexpr (P == 0) { return PowV1(x, y); }
            // else { return PowV2(x, y); }
            else { return ExpImpl<P>(y * LogImpl<P>(x)); }
        }
} // namespace Operon::Backend::detail::fast_approx

#endif
