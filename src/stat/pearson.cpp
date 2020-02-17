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

namespace Operon {
   
    void PearsonsRCalculator::Add(Operon::Scalar x, Operon::Scalar y) 
    {
        if (sumWe <= 0.) {
            sumX = x;
            sumY = y;
            sumWe = 1;
            return;
        }
        // Delta to previous mean
        double deltaX = x * sumWe - sumX;
        double deltaY = y * sumWe - sumY;
        double oldWe = sumWe;
        // Incremental update
        sumWe += 1;
        double f = 1. / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x;
        sumY += y;
    }

    void PearsonsRCalculator::Add(Operon::Scalar x, Operon::Scalar y, Operon::Scalar w) 
    {
        if (w == 0.) {
            return;
        }
        if (sumWe <= 0.) {
            sumX = x * w;
            sumY = y * w;
            sumWe = w;
            return;
        }
        // Delta to previous mean
        double deltaX = x * sumWe - sumX;
        double deltaY = y * sumWe - sumY;
        double oldWe = sumWe;
        // Incremental update
        sumWe += w;
        double f = w / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x * w;
        sumY += y * w;
    }

    double PearsonsRCalculator::Coefficient(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        double sumXX = 0., sumYY = 0., sumXY = 0.;
        double sumX = x[0], sumY = y[0];
        int i = 1;
        while(i < xdim) {
            double xv = x[i], yv = y[i];
            // Delta to previous mean
            double deltaX = xv * i - sumX;
            double deltaY = yv * i - sumY;
            // Increment count first
            double oldi = i;
            ++i;
            double f = 1. / (i * oldi);
            // Update
            sumXX += f * deltaX * deltaX;
            sumYY += f * deltaY * deltaY;
            // should equal deltaY * neltaX!
            sumXY += f * deltaX * deltaY;
            // Update sums
            sumX += xv;
            sumY += yv;
        }
        // One or both series were constant:
        if(!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    double PearsonsRCalculator::WeightedCoefficient(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y, gsl::span<const Operon::Scalar> weights)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        Expects(xdim == weights.size());
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        double sumXX = 0., sumYY = 0., sumXY = 0., sumWe = weights[0];
        double sumX = x[0] * sumWe, sumY = y[0] * sumWe;
        for(int i = 1; i < xdim; ++i) {
            double xv = x[i], yv = y[i], w = weights[i];
            // Delta to previous mean
            double deltaX = xv * sumWe - sumX;
            double deltaY = yv * sumWe - sumY;
            // Increment count first
            double oldWe = sumWe; // Convert to double!
            sumWe += w;
            double f = w / (sumWe * oldWe);
            // Update
            sumXX += f * deltaX * deltaX;
            sumYY += f * deltaY * deltaY;
            // should equal deltaY * neltaX!
            sumXY += f * deltaX * deltaY;
            // Update sums
            sumX += xv * w;
            sumY += yv * w;
        }
        // One or both series were constant:
        if(!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

}

