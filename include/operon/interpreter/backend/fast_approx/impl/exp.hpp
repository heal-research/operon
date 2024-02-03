#ifndef OPERON_BACKEND_FAST_APPROX_EXP_HPP
#define OPERON_BACKEND_FAST_APPROX_EXP_HPP

#include "operon/core/types.hpp"

namespace Operon::Backend::detail::fast_approx {
    inline auto constexpr ExpV1(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x)) { return std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < -85.F) { return 0; }
        if (x > +85.F) { return std::numeric_limits<float>::infinity(); }

        auto const f = x * 12102203.161561485 + 1065054451;
        auto const i = static_cast<int32_t>(f);
        return std::bit_cast<float>(i);
    }

    // http://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo/10792321#10792321
    inline auto constexpr ExpV2(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x)) { return std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < -85.F) { return 0; }
        if (x > +85.F) { return std::numeric_limits<float>::infinity(); }

        float t = x * 1.442695041F;
        float fi = std::floor(t);
        float f = t - fi;
        int i = static_cast<int32_t>(fi);
        auto xf = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
        auto xi = (std::bit_cast<int32_t>(xf) + (i << 23));             /* scale by 2^i */
        return std::bit_cast<float>(xi);
    }

    inline auto constexpr ExpV3(Operon::Scalar x) -> Operon::Scalar {
        if (std::isnan(x)) { return std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (x == 0) { return 1.F; }
        if (x < -85.F) { return 0; }
        if (x > +85.F) { return std::numeric_limits<float>::infinity(); }

        constexpr float a = (1U << 23U) / 0.69314718f;
        constexpr float b = (1U << 23U) * (127 - 0.043677448f);
        x = a * x + b;

        constexpr float c = (1U << 23U);
        constexpr float d = (1U << 23U) * 255;
        if (x < c || x > d)
            x = (x < c) ? 0.0f : d;

        return std::bit_cast<float>(static_cast<uint32_t>(x));
    }

    template<std::size_t P = 0>
    inline auto constexpr ExpImpl(Operon::Scalar x) -> Operon::Scalar {
        if constexpr (P == 0) { return ExpV1(x); }
        else { return ExpV2(x); }
    }
}  // namespace Operon::Backend::detail::fast_approx

#endif
