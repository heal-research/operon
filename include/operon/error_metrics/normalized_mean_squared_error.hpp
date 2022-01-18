// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP
#define OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP

#include <iterator>
#include <type_traits>
#include <vstat/vstat.hpp>
#include "operon/core/types.hpp"
#include "mean_squared_error.hpp"

namespace Operon {

template<typename InputIt1, typename InputIt2>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");

    constexpr double eps{1e-12};
    auto varY = vstat::univariate::accumulate<V1>(begin2, begin2 + std::distance(begin1, end1)).variance;
    if (std::abs(varY) < eps) {
        return varY;
    }
    return MeanSquaredError(begin1, end1, begin2) / varY;
}

template<typename T>
inline auto NormalizedMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    return NormalizedMeanSquaredError(x.begin(), x.end(), y.begin());
}
} // namespace Operon

#endif
