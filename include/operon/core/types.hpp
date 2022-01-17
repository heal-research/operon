// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_TYPES_HPP
#define OPERON_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <nonstd/span.hpp>

#include "constants.hpp"
#include "operon/random/random.hpp"

namespace Operon {
using Hash = uint64_t;
constexpr HashFunction HashFunc = HashFunction::XXHash;

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

namespace Numeric {
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    static constexpr inline auto Max() -> T
    {
        return std::numeric_limits<T>::max();
    }
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    static constexpr inline auto Min() -> T
    {
        if constexpr (std::is_floating_point_v<T>) {
            return std::numeric_limits<T>::lowest();
        } else {
            return std::numeric_limits<T>::min();
        }
    }
} // namespace Numeric
} // namespace Operon

#endif
