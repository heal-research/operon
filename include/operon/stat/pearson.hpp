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

#include <cmath>

#include "core/common.hpp"
#include "gsl/span"

namespace Operon {
namespace detail {
}

class PearsonsRCalculator {
public:
    PearsonsRCalculator()
        : sumXX { 0.0 }
        , sumXY { 0.0 }
        , sumYY { 0.0 }
        , sumX { 0.0 }
        , sumY { 0.0 }
        , sumWe { 0.0 }
    {
    }

    void Reset()
    {
        sumXX = sumYY = sumXY = sumX = sumY = sumWe = 0.0;
    }

    template <typename T>
    void Add(T x, T y)
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

    template <typename T>
    void Add(T x, T y, T w)
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

    template <typename T>
    void Add(gsl::span<const T> x, gsl::span<const T> y);

    // forwarding methods
    template <typename T>
    void Add(std::vector<T> const& x, std::vector<T> const& y)
    {
        return Add(gsl::span<const T>(x), gsl::span<const T>(y));
    }

    template <typename T>
    void Add(Operon::Vector<T> const& x, Operon::Vector<T> const& y)
    {
        return Add(gsl::span<const T>(x), gsl::span<const T>(y));
    }

    double Correlation() const
    {
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    double Count() const { return sumWe; }
    double MeanX() const { return sumX / sumWe; }
    double MeanY() const { return sumY / sumWe; }
    double NaiveCovariance() const { return sumXY / sumWe; }
    double SampleCovariance() const
    {
        EXPECT(sumWe > 1.);
        return sumXY / (sumWe - 1.);
    }

    double NaiveVarianceX() const { return sumXX / sumWe; }
    double SampleVarianceX() const
    {
        EXPECT(sumWe > 1.);
        return sumXX / (sumWe - 1.);
    }

    double NaiveStddevX() const { return std::sqrt(NaiveVarianceX()); }
    double SampleStddevX() const { return std::sqrt(SampleVarianceX()); }
    double NaiveVarianceY() const { return sumYY / sumWe; }
    double SampleVarianceY() const
    {
        EXPECT(sumWe > 1.);
        return sumYY / (sumWe - 1.);
    }

    double NaiveStddevY() const { return std::sqrt(NaiveVarianceY()); }
    double SampleStddevY() const { return std::sqrt(SampleVarianceY()); }

    double SumWe() const { return sumWe; }
    double SumX() const { return sumX; }
    double SumY() const { return sumY; }
    double SumXX() const { return sumXX; }
    double SumYY() const { return sumYY; }
    double SumXY() const { return sumXY; }

    template <typename T>
    static double Coefficient(gsl::span<const T> x, gsl::span<const T> y)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        EXPECT(xdim == ydim);
        EXPECT(xdim > 0);
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        double sumXX = 0., sumYY = 0., sumXY = 0.;
        double sumX = x[0], sumY = y[0];
        size_t i = 1;
        while (i < xdim) {
            double xv = x[i], yv = y[i];
            // Delta to previous mean
            double deltaX = xv * static_cast<double>(i) - sumX;
            double deltaY = yv * static_cast<double>(i) - sumY;
            // Increment count first
            double oldi = static_cast<double>(i);
            ++i;
            double f = 1. / (static_cast<double>(i) * oldi);
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
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    template <typename T>
    static double WeightedCoefficient(gsl::span<const T> x, gsl::span<const T> y, gsl::span<const T> weights)
    {
        auto xdim = x.size();
        auto ydim = y.size();
        EXPECT(xdim == ydim);
        EXPECT(xdim > 0);
        EXPECT(xdim == weights.size());
        // Inlined computation of Pearson correlation, to avoid allocating objects!
        // This is a numerically stabilized version, avoiding sum-of-squares.
        double sumXX = 0., sumYY = 0., sumXY = 0., sumWe = weights[0];
        double sumX = x[0] * sumWe, sumY = y[0] * sumWe;
        for (size_t i = 1; i < xdim; ++i) {
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
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

private:
    // Aggregation for squared residuals - we are not using sum-of-squares!
    double sumXX;
    double sumXY;
    double sumYY;

    // Current mean for X and Y.
    double sumX;
    double sumY;

    // Weight sum
    double sumWe;
};

} // namespace

#endif
