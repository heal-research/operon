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

#include "core/common.hpp"
#include <cmath>

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

    void Add(Operon::Scalar val)
    {
        if (n <= 0) {
            n = 1;
            sum = val;
            m2 = 0;
            return;
        }
        Operon::Scalar tmp = n * val - sum;
        Operon::Scalar oldn = n; // tmp copy
        n += 1.0;
        sum += val;
        m2 += tmp * tmp / (n * oldn);
    }

    void Add(Operon::Scalar val, Operon::Scalar weight)
    {
        if (weight == 0.) {
            return;
        }
        if (n <= 0) {
            n = weight;
            sum = val * weight;
            return;
        }
        val *= weight;
        Operon::Scalar tmp = n * val - sum * weight;
        Operon::Scalar oldn = n; // tmp copy
        n += weight;
        sum += val;
        m2 += tmp * tmp / (weight * n * oldn);
    }

    void Add(gsl::span<const Operon::Scalar> vals)
    {
        int l = vals.size();
        if (l < 2) {
            if (l == 1) {
                Add(vals[0]);
            }
            return;
        }
        // First pass:
        Operon::Scalar s1 = 0.;
        for (int i = 0; i < l; i++) {
            s1 += vals[i];
        }
        Operon::Scalar om1 = s1 / l;
        // Second pass:
        Operon::Scalar om2 = 0., err = 0.;
        for (int i = 0; i < l; i++) {
            Operon::Scalar v = vals[i] - om1;
            om2 += v * v;
            err += v;
        }
        s1 += err;
        om2 += err / l;
        if (n <= 0) {
            n = l;
            sum = s1;
            m2 = om2;
            return;
        }
        Operon::Scalar tmp = n * s1 - sum * l;
        Operon::Scalar oldn = n; // tmp copy
        n += l;
        sum += s1 + err;
        m2 += om2 + tmp * tmp / (l * n * oldn);
    }

    void Add(gsl::span<const Operon::Scalar> vals, gsl::span<const Operon::Scalar> weights)
    {
        Expects(vals.size() == weights.size());
        for (int i = 0, end = vals.size(); i < end; i++) {
            // TODO: use a two-pass update as in the other put
            Add(vals[i], weights[i]);
        }
    }

    // combine data from another MeanVarianceCalculator instance
    void Combine(MeanVarianceCalculator other)
    {
        Operon::Scalar on = other.n, osum = other.sum;
        Operon::Scalar tmp = n * osum - sum * on;
        Operon::Scalar oldn = n; // tmp copy
        n += on;
        sum += osum;
        m2 += other.m2 + tmp * tmp / (on * n * oldn);
    }

    Operon::Scalar NaiveVariance() const 
    {
        Expects(n > 0);
        return m2 / n; 
    }

    Operon::Scalar SampleVariance() const
    {
        Expects(n > 1);
        return m2 / (n - 1);
    }

    Operon::Scalar SumOfSquares() const { return m2; }
    Operon::Scalar StandardDeviation() const { return std::sqrt(SampleVariance()); }

    Operon::Scalar Count() const { return n; }

    Operon::Scalar Mean() const { return sum / n; }

private:
    Operon::Scalar m2;
    Operon::Scalar sum;
    Operon::Scalar n;
};
} // namespace Operon
#endif
