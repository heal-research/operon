// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_CORRELATION_COEFFICIENT_HPP
#define OPERON_METRICS_CORRELATION_COEFFICIENT_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto CorrelationCoefficient(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::bivariate::accumulate<V1>(begin1, end1, begin2).correlation;
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto CorrelationCoefficient(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::bivariate::accumulate<V1>(begin1, end1, begin2, begin3).correlation;
}

template<Concepts::Arithmetic T>
inline auto CorrelationCoefficient(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::bivariate::accumulate<T>(x.data(), x.data() + x.size(), y.data()).correlation;
}

template<Concepts::Arithmetic T>
inline auto CorrelationCoefficient(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::bivariate::accumulate<T>(x.data(), x.data() + x.size(), y.data(), w.data()).correlation;
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelation(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double {
    auto r = CorrelationCoefficient(begin1, end1, begin2);
    return r * r;
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelation(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double {
    auto r = CorrelationCoefficient(begin1, end1, begin2, begin3);
    return r * r;
}

template<Concepts::Arithmetic T>
inline auto SquaredCorrelation(Operon::Span<T const> x, Operon::Span<T const> y) -> double {
    auto r = CorrelationCoefficient(x, y);
    return r * r;
}

template<Concepts::Arithmetic T>
inline auto SquaredCorrelation(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double {
    auto r = CorrelationCoefficient(x, y, w);
    return r * r;
}

} // namespace Operon

#endif
