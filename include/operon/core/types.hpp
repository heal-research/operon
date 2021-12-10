// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <cstdint>
#include <Eigen/StdVector>

#include "random/random.hpp"
#include "span.hpp"

#include "jet.h"

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

// Operon::Vector is just an aligned std::vector
// alignment can be controlled with the EIGEN_MAX_ALIGN_BYTES macro
// https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html#TopicPreprocessorDirectivesPerformance

namespace Numeric {
    template <typename T>
    static constexpr inline T Max()
    {
        return std::numeric_limits<T>::max();
    }
    template <typename T>
    static constexpr inline T Min()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return std::numeric_limits<T>::lowest();
        } else if constexpr (std::is_same_v<T, Dual>) {
            return T { std::numeric_limits<typename Dual::Scalar>::lowest() };
        } else {
            return std::numeric_limits<T>::min();
        }
    }
}
} // namespace operon

#endif
