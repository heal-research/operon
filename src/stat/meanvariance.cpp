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

#include "stat/combine.hpp"
#include "stat/meanvariance.hpp"

#include <Eigen/Core>

namespace Operon {
    template<typename T>
    void MeanVarianceCalculator::Add(T x)
    {
        if (n <= 0) {
            n = 1;
            s = x;
            q = 0;
            return;
        }

        double d = n * x - s;
        n += 1;
        s += x;
        q += d * d / (n * (n-1));
    }

    template<typename T>
    void MeanVarianceCalculator::Add(T x, T w)
    {
        if (w == 0.) {
            return;
        }

        if (n <= 0) {
            n = 1;
            s = x * w;
            q = 0;
            return;
        }

        x *= w;
        double d = n * x - s * w;
        n += w;
        s += x;
        q += d * d / (w * n * (n-1));
    }

    template<typename T>
    void MeanVarianceCalculator::Add(gsl::span<const T> values)
    {
        constexpr int N = 4;
        // the general idea is to partition the data and perform this computation in parallel
        if (values.size() < 4 * N) {
            for (auto v : values) {
                Add(v);
            }
            return;
        }

        using A = Eigen::Array<double, N, 1>; // use doubles for the statistics (more precision)
        using M = Eigen::Map<const Eigen::Array<T, N, 1>>; // type for mapping data from memory

        size_t sz = values.size() - values.size() % 4; // closest multiple of 4

        A qq = A::Zero();
        A nn = A::Ones(); // counts

        A ss = M(values.data()).template cast<double>();

        for (size_t i = N; i < sz; i += N) {
            A xx = M(values.data() + i).template cast<double>();
            A dd = nn * xx - ss;
            nn += 1.0;
            ss += xx;
            qq += dd * dd / (nn * (nn - 1));
        }

        s = ss.sum();
        n = nn.sum();

        q = Combine(nn, ss, qq);

        // deal with remaining values
        if (sz < values.size()) {
            Add(values.subspan(sz, values.size() - sz));
        }
    }

    template<typename T>
    void MeanVarianceCalculator::AddTwoPass(gsl::span<const T> values)
    {
        auto l = values.size();
        if (l < 2) {
            if (l == 1) {
                Add(values[0]);
            }
            return;
        }
        // First pass:
        double s1 = 0.;
        for (size_t i = 0; i < l; i++) {
            s1 += values[i];
        }
        double l1 = static_cast<double>(l);
        double om1 = s1 / l1;
        // Second pass:
        double oq = 0., err = 0.;
        for (size_t i = 0; i < l; i++) {
            double v = values[i] - om1;
            oq += v * v;
            err += v;
        }
        s1 += err;
        oq += err / l1;
        if (n <= 0) {
            n = l1;
            s = s1;
            q = oq;
            return;
        }
        double tmp = n * s1 - s * l1;
        double oldn = n; // tmp copy
        n += l1;
        s += s1 + err;
        q += oq + tmp * tmp / (l1 * n * oldn);
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
    template void MeanVarianceCalculator::AddTwoPass<float>(gsl::span<const float>);
    template void MeanVarianceCalculator::Add<float>(gsl::span<const float>, gsl::span<const float>);
    template void MeanVarianceCalculator::Add<double>(double);
    template void MeanVarianceCalculator::Add<double>(double, double);
    template void MeanVarianceCalculator::Add<double>(gsl::span<const double>);
    template void MeanVarianceCalculator::AddTwoPass<double>(gsl::span<const double>);
    template void MeanVarianceCalculator::Add<double>(gsl::span<const double>, gsl::span<const double>);
}
