// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/types.hpp"
#include "core/contracts.hpp"
#include "vstat.hpp"

#include <type_traits>

namespace Operon {
template<typename T>
inline double MeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), [](auto a, auto b) { auto e = a - b; return e * e; }).mean;
}

template<typename T>
inline double RootMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename T>
inline double NormalizedMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    auto var_y = univariate::accumulate<T>(y.begin(), y.end()).variance;
    if (std::abs(var_y) < 1e-12) {
        return var_y;
    }
    return MeanSquaredError(x, y) / var_y;
}

template<typename T>
inline double L2Norm(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> a(x.data(), x.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b(y.data(), y.size());
    return static_cast<double>((a - b).squaredNorm() / 2);
}

template<typename T>
inline double MeanAbsoluteError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<typename T>
inline double RSquared(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    auto r = bivariate::accumulate<T>(x.data(), y.data(), x.size()).correlation;
    return r * r;
}

// functors to plug into an evaluator
struct MSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanSquaredError(x, y);
    }
};

struct NMSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return NormalizedMeanSquaredError(x, y);
    }
};

struct RMSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return RootMeanSquaredError(x, y);
    }
};

struct MAE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanAbsoluteError(x, y);
    }
};

struct R2 {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return -RSquared(x, y);
    }
};

struct L2 {
    template<typename T>
    inline double operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept
    {
        return L2Norm(x, y);
    }
};

} // namespace
#endif
