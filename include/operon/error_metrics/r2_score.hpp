// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_R2_SCORE_HPP
#define OPERON_METRICS_R2_SCORE_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

// Convention: begin1 = predicted, begin2 = true values.
// vstat::metrics::r2_score uses a parallel SIMD Welford that introduces more
// floating-point roundoff than the sequential accumulate path; the difference
// exceeds our test tolerance against Elki's two-pass reference, so we keep
// the sequential path here.
template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using vstat::univariate::accumulate;
    auto sqres = [](auto a, auto b) { auto e = a - b; return e * e; };
    auto const ssr = accumulate<V1>(begin1, end1, begin2, sqres).sum;
    auto const sst = accumulate<V1>(begin2, begin2 + std::distance(begin1, end1)).ssr;
    if (sst < std::numeric_limits<double>::epsilon()) { return std::numeric_limits<double>::lowest(); }
    return 1.0 - ssr / sst;
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using vstat::univariate::accumulate;
    auto sqres = [](auto a, auto b) { auto e = a - b; return e * e; };
    auto const ssr = accumulate<V1>(begin1, end1, begin2, begin3, sqres).sum;
    auto end2 = begin2 + std::distance(begin1, end1);
    auto const m = accumulate<V1>(begin2, end2).mean;
    auto const sst = accumulate<V1>(begin2, end2, begin3, [&](auto v) { return sqres(v, m); }).sum;
    if (sst < std::numeric_limits<double>::epsilon()) { return std::numeric_limits<double>::lowest(); }
    return 1.0 - ssr / sst;
}

template<Concepts::Arithmetic T>
inline auto R2Score(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return R2Score(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto R2Score(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return R2Score(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
