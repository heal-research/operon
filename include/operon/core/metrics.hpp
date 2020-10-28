/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef METRICS_HPP
#define METRICS_HPP

#include "core/types.hpp"
#include "core/contracts.hpp"
#include "core/stats.hpp"
#include "stat/pearson.hpp"
#include <type_traits>

namespace Operon {
template<typename T>
double NormalizedMeanSquaredError(gsl::span<const T> x, gsl::span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    PearsonsRCalculator calc;
    for(size_t i = 0; i < x.size(); ++i) {
        auto e = x[i] - y[i];
        calc.Add(e * e, y[i]);
    }
    auto yvar = calc.NaiveVarianceY();
    auto errmean = calc.MeanX();
    return yvar > 0 ? errmean / yvar : yvar;
}

template<typename T>
double MeanSquaredError(gsl::span<const T> x, gsl::span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    MeanVarianceCalculator mcalc;
    for(size_t i = 0; i < x.size(); ++i) {
        auto e = x[i] - y[i];
        mcalc.Add(e * e);
    }
    return mcalc.Mean();
}

template<typename T>
double RootMeanSquaredError(gsl::span<const T> x, gsl::span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename T>
double RSquared(gsl::span<const T> x, gsl::span<const T> y)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    auto r = PearsonsRCalculator::Coefficient(x, y);
    return r * r;
}
} // namespace
#endif
