// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/types.hpp"
#include "core/contracts.hpp"
#include "vstat.hpp"

#include <type_traits>

namespace Operon {

namespace detail {
    static auto se = [](auto a, auto b) { auto e = a - b; return e * e; };
}

template<typename T>
inline double MeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y) noexcept
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), detail::se).mean;
}

template<typename InputIt1, typename InputIt2>
inline double MeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    using v1 = typename std::iterator_traits<InputIt1>::value_type;
    using v2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<v1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<v2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<v1, v2>, "The types must be the same");
    return univariate::accumulate<v1>(begin1, end1, begin2, detail::se).mean;
}

template<typename T>
inline double RootMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y) noexcept
{
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename InputIt1, typename InputIt2>
inline double RootMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    return std::sqrt(MeanSquaredError(begin1, end1, begin2));
}

template<typename T>
inline double NormalizedMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y) noexcept
{
    auto var_y = univariate::accumulate<T>(y.begin(), y.end()).variance;
    if (std::abs(var_y) < 1e-12) {
        return var_y;
    }
    return MeanSquaredError(x, y) / var_y;
}

template<typename InputIt1, typename InputIt2>
inline double NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    using v1 = typename std::iterator_traits<InputIt1>::value_type;
    using v2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<v1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<v2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<v1, v2>, "The types must be the same");

    auto var_y = univariate::accumulate<v1>(begin2, begin2 + std::distance(begin1, end1)).variance;
    if (std::abs(var_y) < 1e-12) {
        return var_y;
    }
    return MeanSquaredError(begin1, end1, begin2) / var_y;
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

template<typename InputIt1, typename InputIt2>
inline double L2Norm(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    using v1 = typename std::iterator_traits<InputIt1>::value_type;
    using v2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<v1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<v2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<v1, v2>, "The types must be the same");

    auto sum = univariate::accumulate<v1>(begin1, end1, begin2, detail::se).sum;
    return sum / 2;
}

template<typename T>
inline double MeanAbsoluteError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<typename InputIt1, typename InputIt2>
inline double MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    using v1 = typename std::iterator_traits<InputIt1>::value_type;
    using v2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<v1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<v2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<v1, v2>, "The types must be the same");
    return univariate::accumulate<v1>(begin1, end1, begin2, [](auto a, auto b) { return std::abs(a-b); }).mean;
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

template<typename InputIt1, typename InputIt2>
inline double RSquared(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept
{
    using v1 = typename std::iterator_traits<InputIt1>::value_type;
    using v2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<v1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<v2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<v1, v2>, "The types must be the same");
    auto r = bivariate::accumulate<v1>(begin1, end1, begin2).correlation;
    return r * r;
}

// functors to plug into an evaluator
struct MSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return MeanSquaredError(begin1, end1, begin2);
    }
};

struct NMSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return NormalizedMeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return NormalizedMeanSquaredError(begin1, end1, begin2);
    }
};

struct RMSE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return RootMeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return RootMeanSquaredError(begin1, end1, begin2);
    }
};

struct MAE {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanAbsoluteError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return MeanAbsoluteError(begin1, end1, begin2);
    }
};

struct R2 {
    template<typename T>
    inline double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return -RSquared(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return -RSquared(begin1, end1, begin2);
    }
};

struct L2 {
    template<typename T>
    inline double operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept
    {
        return L2Norm(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline double operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept
    {
        return L2Norm(begin1, end1, begin2);
    }
};

} // namespace
#endif
