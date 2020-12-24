/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include "random/random.hpp"

#if defined(HAVE_CERES)
#include <ceres/jet.h>
#else
#include "jet.h"
#endif

#include <cstdint>

namespace Operon {
using Hash = uint64_t;
using RandomGenerator = Random::RomuTrio;

#if defined(USE_SINGLE_PRECISION)
using Scalar = float;
#else
using Scalar = double;
#endif

#if defined(CERES_ALWAYS_DOUBLE)
using Dual = ceres::Jet<double, 4>;
#else
using Dual = ceres::Jet<Operon::Scalar, 4 * sizeof(double) / sizeof(Scalar)>;
#endif

// Operon::Vector is just an aligned std::vector
// alignment can be controlled with the EIGEN_MAX_ALIGN_BYTES macro
// https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html#TopicPreprocessorDirectivesPerformance
template <typename T>
using Vector = std::vector<T, Eigen::aligned_allocator<T>>;

namespace Numeric {
    template <typename T>
    static constexpr inline T Max()
    {
        return std::numeric_limits<T>::max();
    }
    template <typename T>
    static constexpr inline T Min()
    {
        if constexpr (std::is_floating_point_v<T>)
            return std::numeric_limits<T>::lowest();
        else if constexpr (std::is_same_v<T, Dual>)
            return T { std::numeric_limits<typename Dual::Scalar>::lowest() };
        else
            return std::numeric_limits<T>::min();
    }
}
} // namespace operon

#endif
