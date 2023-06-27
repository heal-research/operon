// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_METRICS_ROOT_MEAN_SQUARED_ERROR_HPP
#define OPERON_METRICS_ROOT_MEAN_SQUARED_ERROR_HPP

#include <iterator>
#include <type_traits>
#include <vstat/vstat.hpp>
#include "operon/core/types.hpp"
#include "mean_squared_error.hpp"

namespace Operon {

template<typename InputIt1, typename InputIt2>
inline auto RootMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    return std::sqrt(MeanSquaredError(begin1, end1, begin2));
}

template<typename InputIt1, typename InputIt2, typename InputIt3>
inline auto RootMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> double
{
    return std::sqrt(MeanSquaredError(begin1, end1, begin2, begin3));
}

template<typename T>
inline auto RootMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename T>
inline auto RootMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> double
{
    return std::sqrt(MeanSquaredError(x, y, w));
}

} // namespace Operon

#endif
