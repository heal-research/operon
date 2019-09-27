#ifndef METRICS_HPP
#define METRICS_HPP

#include <execution>

#include "core/common.hpp"
#include "core/stat/meanvariance.hpp"
#include "core/stat/pearson.hpp"

namespace Operon {
double NormalizedMeanSquaredError(gsl::span<const double> x, gsl::span<const double> y)
{
    Expects(x.size() == y.size());
    Expects(x.size() > 0);
    MeanVarianceCalculator ycalc;
    MeanVarianceCalculator errcalc;
    for(int i = 0; i < x.size(); ++i) {
        if (!std::isnan(y[i])) {
            ycalc.Add(y[i]);
        }
        auto e = x[i] - y[i];
        errcalc.Add(e * e);
    }
    auto yvar = ycalc.NaiveVariance();
    auto errmean = errcalc.Mean();
    return yvar > 0 ? errmean / yvar : yvar;
}

double MeanSquaredError(gsl::span<const double> x, gsl::span<const double> y)
{
    Expects(x.size() == y.size());
    Expects(x.size() > 0);
    MeanVarianceCalculator mcalc;
    for(int i = 0; i < x.size(); ++i) {
        mcalc.Add((x[i] - y[i]) * (x[i] - y[i]));
    }
    return mcalc.Mean();
}

double RootMeanSquaredError(gsl::span<const double> x, gsl::span<const double> y)
{
    return std::sqrt(MeanSquaredError(x, y));
}

double RSquared(gsl::span<const double> x, gsl::span<const double> y)
{
    Expects(x.size() == y.size());
    Expects(x.size() > 0);
    PearsonsRCalculator pcalc;
    for(int i = 0; i < x.size(); ++i) {
        pcalc.Add(x[i], y[i]);
    }
    auto r = pcalc.Correlation();
    return r * r;
}
}
#endif
