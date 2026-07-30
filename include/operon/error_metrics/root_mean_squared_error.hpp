// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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

// Skips non-finite (x, y) pairs instead of letting them poison the whole
// result. Returns the RMSE over the finite subset, plus the count of skipped
// (non-finite) pairs. Delegates to `MeanSquaredErrorFinite` and applies the
// same `std::sqrt` the non-finite `RootMeanSquaredError` does, so a finite
// subset's RMSE stays consistent with its MSE.
template<typename InputIt1, typename InputIt2>
inline auto RootMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> std::pair<double, std::size_t>
{
    auto [mse, skipped] = MeanSquaredErrorFinite(begin1, end1, begin2);
    return {std::sqrt(mse), skipped};
}

template<typename InputIt1, typename InputIt2, typename InputIt3>
inline auto RootMeanSquaredErrorFinite(InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt3 begin3) noexcept -> std::pair<double, std::size_t>
{
    auto [mse, skipped] = MeanSquaredErrorFinite(begin1, end1, begin2, begin3);
    return {std::sqrt(mse), skipped};
}

template<typename T>
inline auto RootMeanSquaredErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> std::pair<double, std::size_t>
{
    auto [mse, skipped] = MeanSquaredErrorFinite(x, y);
    return {std::sqrt(mse), skipped};
}

template<typename T>
inline auto RootMeanSquaredErrorFinite(Operon::Span<T const> x, Operon::Span<T const> y, Operon::Span<T const> w) noexcept -> std::pair<double, std::size_t>
{
    auto [mse, skipped] = MeanSquaredErrorFinite(x, y, w);
    return {std::sqrt(mse), skipped};
}

} // namespace Operon

#endif
