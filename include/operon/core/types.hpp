// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <ceres/jet.h>
#include <cstddef>
#include <cstdint>
#include <nonstd/span.hpp>

#include "constants.hpp"
#include "operon/random/random.hpp"

namespace Operon {
using Hash = uint64_t; // at the moment, changing this will cause problems
using RandomGenerator = Random::RomuTrio;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Span = nonstd::span<T>;

#if defined(USE_SINGLE_PRECISION)
using Scalar = float;
#else
using Scalar = double;
#endif

#if defined(CERES_ALWAYS_DOUBLE)
using Dual = ceres::Jet<double, 4>;
#else
using Dual = ceres::Jet<Scalar, 4 * sizeof(double) / sizeof(Scalar)>;
#endif

namespace Numeric {
    template <typename T>
    static constexpr inline auto Max() -> T
    {
        return std::numeric_limits<T>::max();
    }
    template <typename T>
    static constexpr inline auto Min() -> T
    {
        if constexpr (std::is_floating_point_v<T>) {
            return std::numeric_limits<T>::lowest();
        } else if constexpr (std::is_same_v<T, Dual>) {
            return T { std::numeric_limits<typename Dual::Scalar>::lowest() };
        } else {
            return std::numeric_limits<T>::min();
        }
    }
} // namespace Numeric
} // namespace Operon

#endif
