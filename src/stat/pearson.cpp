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

#include "stat/pearson.hpp"
#include "stat/combine.hpp"

namespace Operon {
    template<typename T>
    void PearsonsRCalculator::Add(gsl::span<const T> x, gsl::span<const T> y)
    {
        constexpr int N = 4;

        EXPECT(x.size() == y.size());
        // the general idea is to partition the data and perform this computation in parallel
        if (x.size() < 4 * N) {
            for (size_t i = 0; i < x.size(); ++i) {
                Add(x[i], y[i]);
            }
            return;
        }

        using A = Eigen::Array<double, N, 1>; // use doubles for the statistics (more precision)
        using M = Eigen::Map<const Eigen::Array<T, N, 1>>; // type for mapping data from memory

        size_t sz = x.size() - x.size() % N; // closest multiple of N

        A _sumX = M(x.data()).template cast<double>();
        A _sumY = M(y.data()).template cast<double>();

        A _sumXX = A::Zero();
        A _sumXY = A::Zero();
        A _sumYY = A::Zero();
        A _sumWe = A::Ones();

        for (size_t n = N; n < sz; n += N) {
            A xx = M(x.data() + n).template cast<double>();
            A yy = M(y.data() + n).template cast<double>();

            A dx = xx * _sumWe - _sumX;
            A dy = yy * _sumWe - _sumY;

            A _sumWeOld = _sumWe;
            _sumWe += 1;

            A f = (_sumWe * _sumWeOld).inverse();

            _sumXX += f * dx * dx;
            _sumYY += f * dy * dy;
            _sumXY += f * dx * dy;

            _sumX += xx;
            _sumY += yy;
        }

        sumWe = _sumWe.sum();
        sumX  = _sumX.sum();
        sumY  = _sumY.sum();

        auto [sxx, syy, sxy] = Combine(_sumWe, _sumX, _sumY, _sumXX, _sumYY, _sumXY);
        sumXX = sxx;
        sumYY = syy;
        sumXY = sxy;

        if (sz < x.size()) {
            Add(x.subspan(sz, x.size() - sz), y.subspan(sz, y.size() - sz));
        }

    }
    // necessary to prevent linker errors
    // https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
    template void   PearsonsRCalculator::Add<float>(float, float);
    template void   PearsonsRCalculator::Add<float>(float, float, float);
    template double PearsonsRCalculator::Coefficient<float>(gsl::span<const float>, gsl::span<const float>);
    template void   PearsonsRCalculator::Add<float>(gsl::span<const float>, gsl::span<const float>);
    template double PearsonsRCalculator::WeightedCoefficient<float>(gsl::span<const float>, gsl::span<const float>, gsl::span<const float>);
    template void   PearsonsRCalculator::Add<double>(double, double);
    template void   PearsonsRCalculator::Add<double>(double, double, double);
    template double PearsonsRCalculator::Coefficient<double>(gsl::span<const double>, gsl::span<const double>);
    template void   PearsonsRCalculator::Add<double>(gsl::span<const double>, gsl::span<const double>);
    template double PearsonsRCalculator::WeightedCoefficient<double>(gsl::span<const double>, gsl::span<const double>, gsl::span<const double>);
}


