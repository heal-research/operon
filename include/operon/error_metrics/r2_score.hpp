// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_METRICS_R2_SCORE_HPP
#define OPERON_METRICS_R2_SCORE_HPP

#include "operon/core/concepts.hpp"
#include "operon/core/contracts.hpp"
#include <iterator>
#include <limits>
#include <utility>
#include <vstat/vstat.hpp>

namespace Operon {

// Convention: begin1 = predicted, begin2 = true values.
// vstat::metrics::r2_score expects (y_true, y_pred), so arguments are swapped.
template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto const n = std::distance(begin1, end1);
    return vstat::metrics::r2_score<V1>(begin2, begin2 + n, begin1);
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto const n = std::distance(begin1, end1);
    return vstat::metrics::r2_score<V1>(begin2, begin2 + n, begin1, begin3);
}

template <Concepts::Arithmetic T>
inline auto R2Score(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return R2Score(x.data(), x.data() + x.size(), y.data());
}

template <Concepts::Arithmetic T>
inline auto R2Score(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return R2Score(x.data(), x.data() + x.size(), y.data(), w.data());
}

// Computes R² over finite (prediction, target) pairs and returns the number
// of omitted pairs. The target mean and TSS use that same finite subset.
template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto R2ScoreFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [stats, skipped] = vstat::bivariate::accumulate<V1, vstat::nan_policy::omit>(begin1, end1, begin2);
    auto const meanDifference = stats.mean_x - stats.mean_y;
    auto const residualSum
        = stats.ssr_x + stats.ssr_y - (2.0 * stats.sum_xy) + (stats.count * meanDifference * meanDifference);
    auto const value = stats.ssr_y < std::numeric_limits<double>::epsilon() ? std::numeric_limits<double>::lowest()
                                                                            : 1.0 - (residualSum / stats.ssr_y);
    return { value, skipped };
}

template <std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
    && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
        typename std::iterator_traits<InputIt2>::value_type>
inline auto R2ScoreFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept
    -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [stats, skipped] = vstat::bivariate::accumulate<V1, vstat::nan_policy::omit>(begin1, end1, begin2, begin3);
    auto const meanDifference = stats.mean_x - stats.mean_y;
    auto const residualSum
        = stats.ssr_x + stats.ssr_y - (2.0 * stats.sum_xy) + (stats.count * meanDifference * meanDifference);
    auto const value = stats.ssr_y < std::numeric_limits<double>::epsilon() ? std::numeric_limits<double>::lowest()
                                                                            : 1.0 - (residualSum / stats.ssr_y);
    return { value, skipped };
}

template <Concepts::Arithmetic T>
inline auto R2ScoreFinite(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> std::pair<double, std::size_t>
{
    EXPECT(y.size() == x.size());
    return R2ScoreFinite(x.data(), x.data() + x.size(), y.data());
}

template <Concepts::Arithmetic T>
inline auto R2ScoreFinite(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept
    -> std::pair<double, std::size_t>
{
    EXPECT(y.size() == x.size());
    return R2ScoreFinite(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
