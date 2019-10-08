#include "core/metrics.hpp"

namespace Operon {
operon::scalar_t NormalizedMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y)
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

operon::scalar_t MeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y)
{
    Expects(x.size() == y.size());
    Expects(x.size() > 0);
    MeanVarianceCalculator mcalc;
    for(int i = 0; i < x.size(); ++i) {
        mcalc.Add((x[i] - y[i]) * (x[i] - y[i]));
    }
    return mcalc.Mean();
}

operon::scalar_t RootMeanSquaredError(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y)
{
    return std::sqrt(MeanSquaredError(x, y));
}

operon::scalar_t RSquared(gsl::span<const operon::scalar_t> x, gsl::span<const operon::scalar_t> y)
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
} // namespace Operon
