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

#ifndef PEARSON_HPP
#define PEARSON_HPP

#include "core/common.hpp"
#include <cmath>

namespace Operon {
class PearsonsRCalculator {
public:
    PearsonsRCalculator()
        : sumXX { 0 }
        , sumXY { 0 }
        , sumYY { 0 }
        , sumX { 0 }
        , sumY { 0 }
        , sumWe { 0. }
    {
    }

    void Reset()
    {
        sumXX = 0.;
        sumYY = 0.;
        sumXY = 0.;
        sumX = 0.;
        sumY = 0.;
        sumWe = 0.;
    }

    void Add(operon::scalar_t x, operon::scalar_t y)
    {
        if (sumWe <= 0.) {
            sumX = x;
            sumY = y;
            sumWe = 1;
            return;
        }
        // Delta to previous mean
        operon::scalar_t deltaX = x * sumWe - sumX;
        operon::scalar_t deltaY = y * sumWe - sumY;
        operon::scalar_t oldWe = sumWe;
        // Incremental update
        sumWe += 1;
        operon::scalar_t f = 1. / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x;
        sumY += y;
    }

    void Add(operon::scalar_t x, operon::scalar_t y, operon::scalar_t w)
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
        operon::scalar_t deltaX = x * sumWe - sumX;
        operon::scalar_t deltaY = y * sumWe - sumY;
        operon::scalar_t oldWe = sumWe;
        // Incremental update
        sumWe += w;
        operon::scalar_t f = w / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x * w;
        sumY += y * w;
    }

    operon::scalar_t Correlation() const
    {
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    operon::scalar_t Count() const
    {
        return sumWe;
    }

    operon::scalar_t MeanX() const
    {
        return sumX / sumWe;
    }

    operon::scalar_t MeanY() const
    {
        return sumY / sumWe;
    }

    operon::scalar_t NaiveCovariance()
    {
        return sumXY / sumWe;
    }

    operon::scalar_t SampleCovariance()
    {
        Expects(sumWe > 1.);
        return sumXY / (sumWe - 1.);
    }

    operon::scalar_t NaiveVarianceX()
    {
        return sumXX / sumWe;
    }

    operon::scalar_t SampleVarianceX()
    {
        Expects(sumWe > 1.);
        return sumXX / (sumWe - 1.);
    }

    operon::scalar_t NaiveStddevX()
    {
        return std::sqrt(NaiveVarianceX());
    }

    operon::scalar_t SampleStddevX()
    {
        return std::sqrt(SampleVarianceX());
    }

    operon::scalar_t NaiveVarianceY()
    {
        return sumYY / sumWe;
    }

    operon::scalar_t SampleVarianceY()
    {
        Expects(sumWe > 1.);
        return sumYY / (sumWe - 1.);
    }

    operon::scalar_t NaiveStddevY()
    {
        return std::sqrt(NaiveVarianceY());
    }

    operon::scalar_t SampleStddevY()
    {
        return std::sqrt(SampleVarianceY());
    }

    static operon::scalar_t Coefficient(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        operon::scalar_t sumXX = 0., sumYY = 0., sumXY = 0.;
        operon::scalar_t sumX = x[0], sumY = y[0];
        int i = 1;
        while(i < xdim) {
            operon::scalar_t xv = x[i], yv = y[i];
            // Delta to previous mean
            operon::scalar_t deltaX = xv * i - sumX;
            operon::scalar_t deltaY = yv * i - sumY;
            // Increment count first
            operon::scalar_t oldi = i;
            ++i;
            operon::scalar_t f = 1. / (i * oldi);
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

    static operon::scalar_t WeightedCoefficient(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y, gsl::span<const operon::scalar_t> weights)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        Expects(xdim == weights.size());
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        operon::scalar_t sumXX = 0., sumYY = 0., sumXY = 0., sumWe = weights[0];
        operon::scalar_t sumX = x[0] * sumWe, sumY = y[0] * sumWe;
        for(int i = 1; i < xdim; ++i) {
            operon::scalar_t xv = x[i], yv = y[i], w = weights[i];
            // Delta to previous mean
            operon::scalar_t deltaX = xv * sumWe - sumX;
            operon::scalar_t deltaY = yv * sumWe - sumY;
            // Increment count first
            operon::scalar_t oldWe = sumWe; // Convert to operon::scalar_t!
            sumWe += w;
            operon::scalar_t f = w / (sumWe * oldWe);
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

private:
    // Aggregation for squared residuals - we are not using sum-of-squares!
    operon::scalar_t sumXX;
    operon::scalar_t sumXY;
    operon::scalar_t sumYY;

    // Current mean for X and Y.
    operon::scalar_t sumX;
    operon::scalar_t sumY;

    // Weight sum
    operon::scalar_t sumWe;
};

} // namespace

#endif
