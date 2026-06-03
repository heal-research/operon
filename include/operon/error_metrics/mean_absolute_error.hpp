// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_MEAN_ABSOLUTE_ERROR_HPP
#define OPERON_METRICS_MEAN_ABSOLUTE_ERROR_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

template<std::random_access_iterator InputIt1, std::random_access_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::univariate::accumulate<V1>(begin1, end1, begin2, [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<std::random_access_iterator InputIt1, std::random_access_iterator InputIt2, std::random_access_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::univariate::accumulate<V1>(begin1, end1, begin2, begin3, [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteError(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::univariate::accumulate<T>(x.data(), x.data() + x.size(), y.data(), [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteError(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::univariate::accumulate<T>(x.data(), x.data() + x.size(), y.data(), w.data(), [](auto a, auto b) { return std::abs(a-b); }).mean;
}

} // namespace Operon

#endif
