// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_CORE_DUAL_HPP
#define OPERON_CORE_DUAL_HPP

#include <ceres/jet.h>
#include "types.hpp"

namespace Operon {
#if defined(CERES_ALWAYS_DOUBLE)
using Dual = ceres::Jet<double, 4>;
#else
using Dual = ceres::Jet<Scalar, 4 * sizeof(double) / sizeof(Scalar)>;
#endif

namespace Numeric {
    template<typename T, std::enable_if_t<std::is_same_v<T, Dual>> = true>
    static constexpr inline auto Max() -> typename Dual::Scalar
    {
        return std::numeric_limits<typename Dual::Scalar>::max();
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, Dual>> = true>
    static constexpr inline auto Min() -> typename Dual::Scalar
    {
        return std::numeric_limits<typename Dual::Scalar>::min();
    }
} // namespace Numeric
} // namespace Operon

#endif
