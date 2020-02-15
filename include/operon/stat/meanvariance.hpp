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

namespace Operon {

class MeanVarianceCalculator {

public:
    MeanVarianceCalculator()
        : m2{0}, sum{0}, n{0}
    {
    }

    void Reset()
    {
        m2 = 0;
        sum = 0;
        n = 0;
    }

    void Add(Operon::Scalar);
    void Add(Operon::Scalar val, Operon::Scalar weight);

    void Add(gsl::span<Operon::Scalar> vals);
    void Add(gsl::span<Operon::Scalar> vals, gsl::span<Operon::Scalar> weights);

    double NaiveVariance() const 
    {
        Expects(n > 0);
        return m2 / n; 
    }

    double SampleVariance() const
    {
        Expects(n > 1);
        return m2 / (n - 1);
    }

    double SumOfSquares() const { return m2; }
    double StandardDeviation() const { return std::sqrt(SampleVariance()); }
    double Count() const { return n; }
    double Mean() const { return sum / n; }

private:
    double m2;
    double sum;
    double n;
};
} // namespace Operon
#endif
