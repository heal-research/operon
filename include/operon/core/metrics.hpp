// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/types.hpp"
#include "core/contracts.hpp"
#include "core/stats.hpp"
#include "stat/pearson.hpp"
#include <type_traits>

namespace Operon {
template<typename T>
double NormalizedMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    PearsonsRCalculator calc;
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> a(x.data(), x.size());
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> b(y.data(), y.size());
    Eigen::Array<T, Eigen::Dynamic, 1> c = (a - b).square();
    calc.Add(Operon::Span<const T>(c.data(), c.size()), y);
    auto yvar = calc.NaiveVarianceY();
    auto errmean = calc.MeanX();
    return yvar > 0 ? errmean / yvar : yvar;
}

template<typename T>
double MeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> a(x.data(), x.size());
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> b(y.data(), y.size());
    Eigen::Array<T, Eigen::Dynamic, 1> c = (a - b).square();
    MeanVarianceCalculator mcalc;
    mcalc.Add(Operon::Span<T const>(c.data(), c.size()));
    return mcalc.Mean();
}

template<typename T>
double MeanAbsoluteError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> a(x.data(), x.size());
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> b(y.data(), y.size());
    Eigen::Array<T, Eigen::Dynamic, 1> c = (a - b).abs();
    MeanVarianceCalculator mcalc;
    mcalc.Add(Operon::Span<T const>(c.data(), c.size()));
    return mcalc.Mean();
}

template<typename T>
double RootMeanSquaredError(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename T>
double RSquared(Operon::Span<const T> x, Operon::Span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    PearsonsRCalculator calc;
    calc.Add(x, y);
    auto r = calc.Correlation();
    return r * r;
}

struct MSE {
    template<typename T>
    double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanSquaredError(x, y);
    }
};

struct NMSE {
    template<typename T>
    double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return NormalizedMeanSquaredError(x, y);
    }
};

struct RMSE {
    template<typename T>
    double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return RootMeanSquaredError(x, y);
    }
};

struct MAE {
    template<typename T>
    double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return MeanAbsoluteError(x, y);
    }
};

struct R2 {
    template<typename T>
    double operator()(Operon::Span<const T> x, Operon::Span<const T> y) const noexcept
    {
        return -RSquared(x, y);
    }
};

} // namespace
#endif
