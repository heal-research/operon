// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_R2_SCORE_HPP
#define OPERON_METRICS_R2_SCORE_HPP

#include <iterator>
#include <vstat/vstat.hpp>
#include "operon/core/concepts.hpp"

namespace Operon {

// Convention: begin1 = predicted, begin2 = true values.
// vstat::metrics::r2_score expects (y_true, y_pred), so arguments are swapped.
template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto const n = std::distance(begin1, end1);
    return vstat::metrics::r2_score<V1>(begin2, begin2 + n, begin1);
}

template<std::contiguous_iterator InputIt1, std::contiguous_iterator InputIt2, std::contiguous_iterator InputIt3>
    requires Concepts::Arithmetic<typename std::iterator_traits<InputIt1>::value_type>
          && std::same_as<typename std::iterator_traits<InputIt1>::value_type,
                          typename std::iterator_traits<InputIt2>::value_type>
inline auto R2Score(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    auto const n = std::distance(begin1, end1);
    return vstat::metrics::r2_score<V1>(begin2, begin2 + n, begin1, begin3);
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
