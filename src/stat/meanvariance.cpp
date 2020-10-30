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

#include "stat/meanvariance.hpp"

namespace Operon {
    template<typename T>
    void MeanVarianceCalculator::Add(T val) 
    {
        if (n <= 0) {
            n = 1;
            sum = val;
            m2 = 0;
            return;
        }
        double tmp = n * val - sum;
        double oldn = n; // tmp copy
        n += 1.0;
        sum += val;
        m2 += tmp * tmp / (n * oldn);
    }

    template<typename T>
    void MeanVarianceCalculator::Add(T val, T weight) 
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
        double tmp = n * val - sum * weight;
        double oldn = n; // tmp copy
        n += weight;
        sum += val;
        m2 += tmp * tmp / (weight * n * oldn);
    }

    template<typename T>
    void MeanVarianceCalculator::Add(gsl::span<const T> vals) 
    {
        auto l = vals.size();
        if (l < 2) {
            if (l == 1) {
                Add(vals[0]);
            }
            return;
        }
        // First pass:
        double s1 = 0.;
        for (size_t i = 0; i < l; i++) {
            s1 += vals[i];
        }
        double l1 = static_cast<double>(l);
        double om1 = s1 / l1;
        // Second pass:
        double om2 = 0., err = 0.;
        for (size_t i = 0; i < l; i++) {
            double v = vals[i] - om1;
            om2 += v * v;
            err += v;
        }
        s1 += err;
        om2 += err / l1;
        if (n <= 0) {
            n = l1;
            sum = s1;
            m2 = om2;
            return;
        }
        double tmp = n * s1 - sum * l1;
        double oldn = n; // tmp copy
        n += l1;
        sum += s1 + err;
        m2 += om2 + tmp * tmp / (l1 * n * oldn);
    }

    template<typename T>
    void MeanVarianceCalculator::Add(gsl::span<const T> vals, gsl::span<const T> weights) 
    {
        EXPECT(vals.size() == weights.size());
        for (size_t i = 0, end = vals.size(); i < end; i++) {
            // TODO: use a two-pass update as in the other put
            Add(vals[i], weights[i]);
        }
    }

    // necessary to prevent linker errors 
    // https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
    template void MeanVarianceCalculator::Add<float>(float);
    template void MeanVarianceCalculator::Add<float>(float, float);
    template void MeanVarianceCalculator::Add<float>(gsl::span<const float>);
    template void MeanVarianceCalculator::Add<float>(gsl::span<const float>, gsl::span<const float>);
    template void MeanVarianceCalculator::Add<double>(double);
    template void MeanVarianceCalculator::Add<double>(double, double);
    template void MeanVarianceCalculator::Add<double>(gsl::span<const double>);
    template void MeanVarianceCalculator::Add<double>(gsl::span<const double>, gsl::span<const double>);
}
