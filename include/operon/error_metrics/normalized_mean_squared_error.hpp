// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP
#define OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"
#include "mean_squared_error.hpp"

namespace Operon {

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto varY = vstat::univariate::accumulate<V1>(begin2, begin2 + std::distance(begin1, end1)).variance;
    if (varY > 0) {
        return MeanSquaredError(begin1, end1, begin2) / varY;
    }
    return 0.0;
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto varY = vstat::univariate::accumulate<V1>(begin2, begin2 + std::distance(begin1, end1), begin3).variance;
    if (varY > 0) {
        return MeanSquaredError(begin1, end1, begin2, begin3) / varY;
    }
    return 0.0;
}

template<Concepts::Arithmetic T>
inline auto NormalizedMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    return NormalizedMeanSquaredError(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto NormalizedMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> double
{
    return NormalizedMeanSquaredError(x.data(), x.data() + x.size(), y.data(), w.data());
}

// Skips non-finite (x, y) pairs instead of letting them poison the whole
// result. The target's variance is masked by the same finiteness pairing
// (an estimated-value NaN/Inf excludes that row from both the numerator and
// the variance denominator) so the two stay consistent. Returns the NMSE
// over the finite subset, plus the count of skipped (non-finite) pairs.
template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [mse, skipped] = MeanSquaredErrorFinite(begin1, end1, begin2);
    auto varY = vstat::univariate::accumulate_finite<V1, vstat::stats::variance>(
        begin2, begin2 + std::distance(begin1, end1), begin1).first.variance;
    if (varY > 0) {
        return {mse / varY, skipped};
    }
    return {0.0, skipped};
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto [mse, skipped] = MeanSquaredErrorFinite(begin1, end1, begin2, begin3);
    auto varY = vstat::univariate::accumulate_finite<V1, vstat::stats::variance>(
        begin2, begin2 + std::distance(begin1, end1), begin1, begin3).first.variance;
    if (varY > 0) {
        return {mse / varY, skipped};
    }
    return {0.0, skipped};
}

template<Concepts::Arithmetic T>
inline auto NormalizedMeanSquaredErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> std::pair<double, std::size_t>
{
    return NormalizedMeanSquaredErrorFinite(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto NormalizedMeanSquaredErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> std::pair<double, std::size_t>
{
    return NormalizedMeanSquaredErrorFinite(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
