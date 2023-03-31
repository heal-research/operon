// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_R2_SCORE_HPP
#define OPERON_METRICS_R2_SCORE_HPP

#include <iterator>
#include <type_traits>
#include <vstat/vstat.hpp>
#include "operon/core/types.hpp"

namespace Operon {
template<typename InputIt1, typename InputIt2>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    using vstat::univariate::accumulate;
    auto const ssr = accumulate<V1>(begin1, end1, begin2, std::minus{}).ssr;
    auto const sst = accumulate<V2>(begin2, begin2 + std::distance(begin1, end1)).ssr;
    if (sst < std::numeric_limits<double>::epsilon()) { return std::numeric_limits<double>::lowest(); }
    return 1.0 - ssr / sst;
}

template<typename T>
inline auto R2Score(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return R2Score(x.begin(), x.end(), y.begin());
}
} // namespace Operon

#endif
