// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef MEANVARIANCE_HPP
#define MEANVARIANCE_HPP

#include <cmath>

#include "core/common.hpp"

namespace Operon {
namespace detail {
    using A = Eigen::Array4d;
    using R = Eigen::Ref<A>;
    using M = Eigen::Map<A>;

}

class MeanVarianceCalculator {
public:
    MeanVarianceCalculator()
        : q { 0 }
        , s { 0 }
        , n { 0 }
    {
    }

    void Reset()
    {
        q = 0;
        s = 0;
        n = 0;
    }

    template <typename T>
    void Add(T value);

    template <typename T>
    void Add(T value, T weight);

    template <typename T>
    void Add(Operon::Span<const T> values);

    template <typename T>
    void AddTwoPass(Operon::Span<const T> values);

    template <typename T>
    void Add(Operon::Span<const T> values, Operon::Span<const T> weights);

    template <typename T>
    void Add(std::vector<T> const& values) { Add(Operon::Span<const T> { values.data(), values.size() }); }

    template <typename T>
    void Add(Operon::Vector<T> const& values) { Add(Operon::Span<const T> { values.data(), values.size() }); }

    template <typename T>
    void AddTwoPass(std::vector<T> const& values) { AddTwoPass(Operon::Span<const T> { values.data(), values.size() }); }

    template <typename T>
    void AddTwoPass(Operon::Vector<T> const& values) { AddTwoPass(Operon::Span<const T> { values.data(), values.size() }); }

    double NaiveVariance() const
    {
        EXPECT(n > 0);
        return q / n;
    }

    double SampleVariance() const
    {
        EXPECT(n > 1);
        return q / (n - 1);
    }

    double SumOfSquares() const { return q; }
    double NaiveStandardDeviation() const { return std::sqrt(NaiveVariance()); }
    double SampleStandardDeviation() const { return std::sqrt(SampleVariance()); }
    double Count() const { return n; }
    double Mean() const { return s / n; }

private:
    double q; // sum of squares
    double s; // sum
    double n; // number of elements
};
} // namespace Operon
#endif
