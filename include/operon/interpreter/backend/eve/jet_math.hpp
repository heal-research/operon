// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_BACKEND_EVE_JET_MATH_HPP
#define OPERON_BACKEND_EVE_JET_MATH_HPP

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include "operon/core/jet.hpp"

namespace Operon {

// JetMath (see core/jet.hpp) for SIMD lanes: Jet<eve::wide<T>, N> evaluates
// the primal and the N tangents of a whole register of independent scalars at
// once. Every lane follows the same conventions as the scalar JetMath<T>
// (sgn(0) = 0, NaN tie weights for fmin/fmax, std::fmin NaN selection); the
// elementary functions are the full-accuracy eve ones, not the backend's Fast*
// approximations, so lanes agree with the scalar Jet to within a few ULP.
template <typename T, typename Cardinal> struct JetMath<eve::wide<T, Cardinal>> {
    using W = eve::wide<T, Cardinal>;

    static auto abs(W x) -> W { return eve::abs(x); }
    static auto sqr(W x) -> W { return x * x; }
    static auto sgn(W x) -> W
    {
        return eve::if_else(x > W { T(0) }, W { T(1) }, eve::if_else(x < W { T(0) }, W { T(-1) }, W { T(0) }));
    }
    static auto sqrt(W x) -> W { return eve::sqrt(x); }
    static auto cbrt(W x) -> W { return eve::cbrt(x); }
    static auto exp(W x) -> W { return eve::exp(x); }
    static auto log(W x) -> W { return eve::log(x); }
    static auto log1p(W x) -> W { return eve::log1p(x); }
    static auto sin(W x) -> W { return eve::sin(x); }
    static auto cos(W x) -> W { return eve::cos(x); }
    static auto tan(W x) -> W { return eve::tan(x); }
    static auto asin(W x) -> W { return eve::asin(x); }
    static auto acos(W x) -> W { return eve::acos(x); }
    static auto atan(W x) -> W { return eve::atan(x); }
    static auto sinh(W x) -> W { return eve::sinh(x); }
    static auto cosh(W x) -> W { return eve::cosh(x); }
    static auto tanh(W x) -> W { return eve::tanh(x); }
    static auto floor(W x) -> W { return eve::floor(x); }
    static auto ceil(W x) -> W { return eve::ceil(x); }
    static auto pow(W x, W y) -> W { return eve::pow(x, y); }
    static auto xlogy(W x, W y) -> W { return eve::if_else(x == W { T(0) }, W { T(0) }, x * eve::log(y)); }
    static auto pow_exponent_tangent(W r, W x, W y, W gv) -> W
    {
        auto const realNegativeBase = (x < W { T(0) }) && (y == eve::floor(y));
        return eve::if_else(
            realNegativeBase, eve::if_else(gv == W { T(0) }, W { T(0) }, eve::nan(eve::as<W> {})), xlogy(r, x) * gv);
    }

    // eve::min/max follow the x86 minps/maxps convention rather than the
    // std::fmin/std::fmax one, so select the non-NaN operand explicitly.
    static auto fmin(W x, W y) -> W
    {
        return eve::if_else(eve::is_nan(x), y, eve::if_else(eve::is_nan(y), x, eve::min(x, y)));
    }
    static auto fmax(W x, W y) -> W
    {
        return eve::if_else(eve::is_nan(x), y, eve::if_else(eve::is_nan(y), x, eve::max(x, y)));
    }

    static auto min_weight(W x, W y) -> W { return Weight(x, y, x < y); }
    static auto max_weight(W x, W y) -> W { return Weight(x, y, x > y); }

private:
    template <typename Mask> static auto Weight(W x, W y, Mask selected) -> W
    {
        auto const nanX = eve::is_nan(x);
        auto const nanY = eve::is_nan(y);
        auto w = eve::if_else(selected, W { T(1) }, W { T(0) });
        w = eve::if_else(nanY, W { T(1) }, w);
        w = eve::if_else(nanX, W { T(0) }, w);
        return eve::if_else((nanX && nanY) || (x == y), eve::nan(eve::as<W> {}), w);
    }
};

} // namespace Operon

#endif
