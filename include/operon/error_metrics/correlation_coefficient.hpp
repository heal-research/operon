// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_CORRELATION_COEFFICIENT_HPP
#define OPERON_METRICS_CORRELATION_COEFFICIENT_HPP

#include <iterator>
#include <type_traits>
#include <vstat/vstat.hpp>
#include "operon/core/types.hpp"

namespace Operon {

template<typename InputIt1, typename InputIt2>
inline auto CorrelationCoefficient(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    return vstat::bivariate::accumulate<V1>(begin1, end1, begin2).correlation;
}

template<typename T>
inline auto CorrelationCoefficient(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::bivariate::accumulate<T>(x.data(), y.data(), x.size()).correlation;
}
} // namespace Operon

#endif
