/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

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
