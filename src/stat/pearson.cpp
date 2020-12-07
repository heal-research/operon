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
        EXPECT(x.size() == y.size());
        // the general idea is to partition the data and perform this computation in parallel
        if (x.size() < 16) {
            for (size_t i = 0; i < x.size(); ++i) {
                Add(x[i], y[i]);
            }
        }

        size_t sz = x.size() - x.size() % 4; // closest multiple of 4
        size_t ps = sz / 4; // partition size

        gsl::span<const T> xs[4] = {
            x.subspan(0 * ps, ps),
            x.subspan(1 * ps, ps),
            x.subspan(2 * ps, ps),
            x.subspan(3 * ps, ps)
        };

        gsl::span<const T> ys[4] = {
            y.subspan(0 * ps, ps),
            y.subspan(1 * ps, ps),
            y.subspan(2 * ps, ps),
            y.subspan(3 * ps, ps)
        };

        Eigen::Array4d _sumX {
            xs[0][0],
            xs[1][0],
            xs[2][0],
            xs[3][0]
        };

        Eigen::Array4d _sumY {
            ys[0][0],
            ys[1][0],
            ys[2][0],
            ys[3][0]
        };

        Eigen::Array4d _sumXX { 0, 0, 0, 0 };
        Eigen::Array4d _sumXY { 0, 0, 0, 0 };
        Eigen::Array4d _sumYY { 0, 0, 0, 0 };
        Eigen::Array4d _sumWe { 1, 1, 1 ,1 };

        for (size_t i = 1; i < ps; ++i) {
            Eigen::Array4d xx {
                xs[0][i],
                xs[1][i],
                xs[2][i],
                xs[3][i]
            };

            Eigen::Array4d yy {
                ys[0][i],
                ys[1][i],
                ys[2][i],
                ys[3][i]
            };

            Eigen::Array4d dx = xx * _sumWe - _sumX;
            Eigen::Array4d dy = yy * _sumWe - _sumY;

            _sumWe += 1;

            Eigen::Array4d f = (_sumWe * (_sumWe - 1)).inverse();

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


