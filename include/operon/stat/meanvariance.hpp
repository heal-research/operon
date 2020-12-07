/* This code represents derived work from ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2019
 * ELKI Development Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MEANVARIANCE_HPP
#define MEANVARIANCE_HPP

#include <cmath>

#include "core/common.hpp"
#include "gsl/span"

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
    void Add(gsl::span<const T> values);

    template <typename T>
    void AddTwoPass(gsl::span<const T> values);

    template <typename T>
    void Add(gsl::span<const T> values, gsl::span<const T> weights);

    template <typename T>
    void Add(std::vector<T> const& values) { Add(gsl::span<const T> { values.data(), values.size() }); }

    template <typename T>
    void Add(Operon::Vector<T> const& values) { Add(gsl::span<const T> { values.data(), values.size() }); }

    template <typename T>
    void AddTwoPass(std::vector<T> const& values) { AddTwoPass(gsl::span<const T> { values.data(), values.size() }); }

    template <typename T>
    void AddTwoPass(Operon::Vector<T> const& values) { AddTwoPass(gsl::span<const T> { values.data(), values.size() }); }

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
