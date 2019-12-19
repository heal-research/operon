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

    void Add(Operon::Scalar x, Operon::Scalar y)
    {
        if (sumWe <= 0.) {
            sumX = x;
            sumY = y;
            sumWe = 1;
            return;
        }
        // Delta to previous mean
        Operon::Scalar deltaX = x * sumWe - sumX;
        Operon::Scalar deltaY = y * sumWe - sumY;
        Operon::Scalar oldWe = sumWe;
        // Incremental update
        sumWe += 1;
        Operon::Scalar f = 1. / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x;
        sumY += y;
    }

    void Add(Operon::Scalar x, Operon::Scalar y, Operon::Scalar w)
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
        Operon::Scalar deltaX = x * sumWe - sumX;
        Operon::Scalar deltaY = y * sumWe - sumY;
        Operon::Scalar oldWe = sumWe;
        // Incremental update
        sumWe += w;
        Operon::Scalar f = w / (sumWe * oldWe);
        // Update
        sumXX += f * deltaX * deltaX;
        sumYY += f * deltaY * deltaY;
        // should equal weight * deltaY * neltaX!
        sumXY += f * deltaX * deltaY;
        // Update means
        sumX += x * w;
        sumY += y * w;
    }

    Operon::Scalar Correlation() const
    {
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    Operon::Scalar Count() const
    {
        return sumWe;
    }

    Operon::Scalar MeanX() const
    {
        return sumX / sumWe;
    }

    Operon::Scalar MeanY() const
    {
        return sumY / sumWe;
    }

    Operon::Scalar NaiveCovariance()
    {
        return sumXY / sumWe;
    }

    Operon::Scalar SampleCovariance()
    {
        Expects(sumWe > 1.);
        return sumXY / (sumWe - 1.);
    }

    Operon::Scalar NaiveVarianceX()
    {
        return sumXX / sumWe;
    }

    Operon::Scalar SampleVarianceX()
    {
        Expects(sumWe > 1.);
        return sumXX / (sumWe - 1.);
    }

    Operon::Scalar NaiveStddevX()
    {
        return std::sqrt(NaiveVarianceX());
    }

    Operon::Scalar SampleStddevX()
    {
        return std::sqrt(SampleVarianceX());
    }

    Operon::Scalar NaiveVarianceY()
    {
        return sumYY / sumWe;
    }

    Operon::Scalar SampleVarianceY()
    {
        Expects(sumWe > 1.);
        return sumYY / (sumWe - 1.);
    }

    Operon::Scalar NaiveStddevY()
    {
        return std::sqrt(NaiveVarianceY());
    }

    Operon::Scalar SampleStddevY()
    {
        return std::sqrt(SampleVarianceY());
    }

    static Operon::Scalar Coefficient(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        Operon::Scalar sumXX = 0., sumYY = 0., sumXY = 0.;
        Operon::Scalar sumX = x[0], sumY = y[0];
        int i = 1;
        while(i < xdim) {
            Operon::Scalar xv = x[i], yv = y[i];
            // Delta to previous mean
            Operon::Scalar deltaX = xv * i - sumX;
            Operon::Scalar deltaY = yv * i - sumY;
            // Increment count first
            Operon::Scalar oldi = i;
            ++i;
            Operon::Scalar f = 1. / (i * oldi);
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

    static Operon::Scalar WeightedCoefficient(gsl::span<const Operon::Scalar> x, gsl::span<const Operon::Scalar> y, gsl::span<const Operon::Scalar> weights)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        Expects(xdim == ydim);
        Expects(xdim > 0);
        Expects(xdim == weights.size());
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        Operon::Scalar sumXX = 0., sumYY = 0., sumXY = 0., sumWe = weights[0];
        Operon::Scalar sumX = x[0] * sumWe, sumY = y[0] * sumWe;
        for(int i = 1; i < xdim; ++i) {
            Operon::Scalar xv = x[i], yv = y[i], w = weights[i];
            // Delta to previous mean
            Operon::Scalar deltaX = xv * sumWe - sumX;
            Operon::Scalar deltaY = yv * sumWe - sumY;
            // Increment count first
            Operon::Scalar oldWe = sumWe; // Convert to Operon::Scalar!
            sumWe += w;
            Operon::Scalar f = w / (sumWe * oldWe);
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
    Operon::Scalar sumXX;
    Operon::Scalar sumXY;
    Operon::Scalar sumYY;

    // Current mean for X and Y.
    Operon::Scalar sumX;
    Operon::Scalar sumY;

    // Weight sum
    Operon::Scalar sumWe;
};

} // namespace

#endif
