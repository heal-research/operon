// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_METRICS_CORRELATION_COEFFICIENT_HPP
#define OPERON_METRICS_CORRELATION_COEFFICIENT_HPP

#include "operon/core/concepts.hpp"
#include "operon/core/contracts.hpp"
#include <iterator>
#include <utility>
#include <vstat/vstat.hpp>

namespace Operon {

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto CorrelationCoefficient(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::bivariate::accumulate<V1>(begin1, end1, begin2).correlation;
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto CorrelationCoefficient(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::bivariate::accumulate<V1>(begin1, end1, begin2, begin3).correlation;
}

template <Concepts::Arithmetic T>
inline auto CorrelationCoefficient(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::bivariate::accumulate<T>(x.data(), x.data() + x.size(), y.data()).correlation;
}

template <Concepts::Arithmetic T>
inline auto CorrelationCoefficient(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return vstat::bivariate::accumulate<T>(x.data(), x.data() + x.size(), y.data(), w.data()).correlation;
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelation(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    auto r = CorrelationCoefficient(begin1, end1, begin2);
    return r * r;
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelation(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    auto r = CorrelationCoefficient(begin1, end1, begin2, begin3);
    return r * r;
}

template <Concepts::Arithmetic T>
inline auto SquaredCorrelation(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    auto r = CorrelationCoefficient(x, y);
    return r * r;
}

template <Concepts::Arithmetic T>
inline auto SquaredCorrelation(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double
{
    auto r = CorrelationCoefficient(x, y, w);
    return r * r;
}

// Computes squared correlation over finite (prediction, target) pairs and
// returns the number of omitted pairs.
template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelationFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
    -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [stats, skipped] = vstat::bivariate::accumulate<V1, vstat::nan_policy::omit>(begin1, end1, begin2);
    return { stats.correlation * stats.correlation, skipped };
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto SquaredCorrelationFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept
    -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [stats, skipped] = vstat::bivariate::accumulate<V1, vstat::nan_policy::omit>(begin1, end1, begin2, begin3);
    return { stats.correlation * stats.correlation, skipped };
}

template <Concepts::Arithmetic T>
inline auto SquaredCorrelationFinite(Operon::Span<T const> x, Operon::Span<T const> y) noexcept
    -> std::pair<double, std::size_t>
{
    EXPECT(x.size() == y.size());
    return SquaredCorrelationFinite(x.data(), x.data() + x.size(), y.data());
}

template <Concepts::Arithmetic T>
inline auto SquaredCorrelationFinite(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept
    -> std::pair<double, std::size_t>
{
    EXPECT(x.size() == y.size());
    return SquaredCorrelationFinite(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
