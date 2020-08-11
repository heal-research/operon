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
        : sumXX { 0.0 }
        , sumXY { 0.0 }
        , sumYY { 0.0 }
        , sumX  { 0.0 }
        , sumY  { 0.0 }
        , sumWe { 0.0 }
    {
    }

    void Reset()
    {
        sumXX = sumYY = sumXY = sumX = sumY = sumWe = 0.0;
    }

    template<typename T>
    void Add(T x, T y);

    template<typename T>
    void Add(T x, T y, T w);

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
    double SampleVarianceY() const {
        EXPECT(sumWe > 1.);
        return sumYY / (sumWe - 1.);
    }

    double NaiveStddevY() const { return std::sqrt(NaiveVarianceY()); }
    double SampleStddevY() const { return std::sqrt(SampleVarianceY()); }

    template<typename T>
    static double Coefficient(gsl::span<const T> x, gsl::span<const T> y);

    template<typename T>
    static double WeightedCoefficient(gsl::span<const T> x, gsl::span<const T> y, gsl::span<const T> weights);

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
