// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_CORE_DUAL_HPP
#define OPERON_CORE_DUAL_HPP

#include "types.hpp"

#if defined(HAVE_CERES)
#include <ceres/jet.h>
#else
#include "operon/ceres/jet.h"
#endif

namespace Operon {

using Dual = ceres::Jet<Operon::Scalar, 4 * sizeof(double) / sizeof(Scalar)>;

namespace Numeric {
    template<typename T, std::enable_if_t<std::is_same_v<T, Dual>> = true>
    static constexpr inline auto Max() -> Operon::Scalar
    {
        return std::numeric_limits<Operon::Scalar>::max();
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, Dual>> = true>
    static constexpr inline auto Min() -> Operon::Scalar
    {
        if constexpr (std::is_floating_point_v<Operon::Scalar>) {
            return std::numeric_limits<T>::lowest();
        } else {
            return std::numeric_limits<T>::min();
        }
    }
} // namespace Numeric
} // namespace Operon

#endif
