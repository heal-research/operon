// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_SUM_OF_SQUARED_RESIDUALS_HPP
#define OPERON_METRICS_SUM_OF_SQUARED_RESIDUALS_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

template<std::random_access_iterator InputIt1, std::random_access_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto SumOfSquaredErrors(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto sqres = [](auto a, auto b){ auto e = a-b; return e*e; };
    return vstat::univariate::accumulate<V1>(begin1, end1, begin2, sqres).sum;
}

template<std::random_access_iterator InputIt1, std::random_access_iterator InputIt2, std::random_access_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto SumOfSquaredErrors(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto sqres = [](auto a, auto b){ auto e = a-b; return e*e; };
    return vstat::univariate::accumulate<V1>(begin1, end1, begin2, begin3, sqres).sum;
}

template<Concepts::Arithmetic T>
inline auto SumOfSquaredErrors(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return SumOfSquaredErrors(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto SumOfSquaredErrors(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return SumOfSquaredErrors(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
