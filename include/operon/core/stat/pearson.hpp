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

    void Add(double x, double y)
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

    void Add(double x, double y, double w)
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

    double Correlation() const
    {
        if (!(sumXX > 0. && sumYY > 0.)) {
            return (sumXX == sumYY) ? 1. : 0.;
        }
        return sumXY / std::sqrt(sumXX * sumYY);
    }

    double Count() const
    {
        return sumWe;
    }

    double MeanX() const
    {
        return sumX / sumWe;
    }

    double MeanY() const
    {
        return sumY / sumWe;
    }

    double NaiveCovariance()
    {
        return sumXY / sumWe;
    }

    double SampleCovariance()
    {
        Expects(sumWe > 1.);
        return sumXY / (sumWe - 1.);
    }

    double NaiveVarianceX()
    {
        return sumXX / sumWe;
    }

    double SampleVarianceX()
    {
        Expects(sumWe > 1.);
        return sumXX / (sumWe - 1.);
    }

    double NaiveStddevX()
    {
        return std::sqrt(NaiveVarianceX());
    }

    double SampleStddevX()
    {
        return std::sqrt(SampleVarianceX());
    }

    double NaiveVarianceY()
    {
        return sumYY / sumWe;
    }

    double SampleVarianceY()
    {
        Expects(sumWe > 1.);
        return sumYY / (sumWe - 1.);
    }

    double NaiveStddevY()
    {
        return std::sqrt(NaiveVarianceY());
    }

    double SampleStddevY()
    {
        return std::sqrt(SampleVarianceY());
    }

    static double Coefficient(gsl::span<const double> x, gsl::span<const double> y)
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
            double oldi = i; // Convert to double!
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

    static double WeightedCoefficient(gsl::span<const double> x, gsl::span<const double> y, gsl::span<const double> weights)
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
