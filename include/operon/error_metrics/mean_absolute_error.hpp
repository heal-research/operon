// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_METRICS_MEAN_ABSOLUTE_ERROR_HPP
#define OPERON_METRICS_MEAN_ABSOLUTE_ERROR_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::mean_absolute_error<V1>(begin1, end1, begin2);
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::mean_absolute_error<V1>(begin1, end1, begin2, begin3);
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteError(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return MeanAbsoluteError(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteError(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> double
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return MeanAbsoluteError(x.data(), x.data() + x.size(), y.data(), w.data());
}

// Skips non-finite (x, y) pairs instead of letting them poison the whole
// result. Returns the MAE over the finite subset, plus the count of skipped
// (non-finite) pairs.
template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::mean_absolute_error_finite<V1>(begin1, end1, begin2);
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto MeanAbsoluteErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> std::pair<double, std::size_t>
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    return vstat::metrics::mean_absolute_error_finite<V1>(begin1, end1, begin2, begin3);
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y) -> std::pair<double, std::size_t>
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return MeanAbsoluteErrorFinite(x.data(), x.data() + x.size(), y.data());
}

template<Concepts::Arithmetic T>
inline auto MeanAbsoluteErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) -> std::pair<double, std::size_t>
{
    EXPECT(x.size() == y.size());
    EXPECT(!x.empty());
    return MeanAbsoluteErrorFinite(x.data(), x.data() + x.size(), y.data(), w.data());
}

} // namespace Operon

#endif
