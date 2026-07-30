// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP
#define OPERON_METRICS_NORMALIZED_MEAN_SQUARED_ERROR_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::normalized_mean_squared_error<V1>(begin1, end1, begin2);
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::normalized_mean_squared_error<V1>(begin1, end1, begin2, begin3);
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
// the variance denominator) so the two stay consistent. Delegates to the
// single-pass vstat primitive that folds the residual-mean and
// target-variance accumulators into one chunked scan, so the per-chunk
// is_finite mask is computed once and BOTH accumulators run under it.
// Returns the NMSE over the finite subset, plus the count of skipped
// (non-finite) pairs.
template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
           && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                           typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::normalized_mean_squared_error<V1, vstat::nan_policy::omit>(begin1, end1, begin2);
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
           && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                           typename std::iterator_traits<InputIt2>::value_type>
inline auto NormalizedMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::normalized_mean_squared_error<V1, vstat::nan_policy::omit>(begin1, end1, begin2, begin3);
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
