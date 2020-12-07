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

#include "stat/meanvariance.hpp"
#include "stat/meanvariance2.hpp"
#include "stat/meanvariance3.hpp"
#include "stat/pearson.hpp"

#include <doctest/doctest.h>

#include <gsl/span>

namespace Operon::Test {
TEST_CASE("constant series")
{
    const std::vector<double> xs = {
        35874426.078924179,
        35874426.078924179,
        21524655.647354506,
        43049311.294709012,
        28699540.863139343,
        57399081.726278685,
        7174885.2157848356,
        14349770.431569671,
        14349770.431569671,
        28699540.863139343,
        7174885.2157848356,
        14349770.431569671
    };

    const std::vector<double> ys = {
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47,
        305.47
    };

    SUBCASE("pearson correlation")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\npearson correlation (schubert)\n");
        Operon::PearsonsRCalculator rCalc;

        for (size_t i = 0; i < xs.size(); ++i) {
            rCalc.Add(xs[i], ys[i]);
        }

        fmt::print("mean == xs: {}, ys: {}\n", rCalc.MeanX(), rCalc.MeanY());
        fmt::print("naive variance == xs: {}, ys: {}\n", rCalc.NaiveVarianceX(), rCalc.NaiveVarianceY());
        fmt::print("sample variance == xs: {}, ys: {}\n", rCalc.SampleVarianceX(), rCalc.SampleVarianceY());
        fmt::print("correlation: {}\n", rCalc.Correlation());
    }

    SUBCASE("welford")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\nwelford\n");
        Operon::MeanVariance2 calc;
        for (size_t i = 0; i < xs.size(); ++i) {
            calc.Add(ys[i]);
        }
        fmt::print("mean: {}\n", calc.Mean());
        fmt::print("naive variance: {}\n", calc.NaiveVariance());
        fmt::print("sample variance: {}\n", calc.SampleVariance());
    }

    SUBCASE("youngs-cramer")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\nyoungs-cramer\n");
        Operon::MeanVariance3 calc;
        for (size_t i = 0; i < xs.size(); ++i) {
            calc.Add(ys[i]);
        }
        fmt::print("mean: {}\n", calc.Mean());
        fmt::print("naive variance: {}\n", calc.NaiveVariance());
        fmt::print("sample variance: {}\n", calc.SampleVariance());
    }
}

TEST_CASE("random series")
{

    std::vector<double> xs(10000);
    std::vector<double> ys(10000);

    Operon::RandomGenerator rng(1234);
    std::uniform_real_distribution<double> dist(-100, 100);
    std::generate(xs.begin(), xs.end(), [&]() { return dist(rng); });
    std::generate(ys.begin(), ys.end(), [&]() { return dist(rng); });

    SUBCASE("pearson correlation")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\npearson correlation (schubert)\n");
        Operon::PearsonsRCalculator rCalc;

        for (size_t i = 0; i < xs.size(); ++i) {
            rCalc.Add(xs[i], ys[i]);
        }

        fmt::print("mean == xs: {}, ys: {}\n", rCalc.MeanX(), rCalc.MeanY());
        fmt::print("naive variance == xs: {}, ys: {}\n", rCalc.NaiveVarianceX(), rCalc.NaiveVarianceY());
        fmt::print("sample variance == xs: {}, ys: {}\n", rCalc.SampleVarianceX(), rCalc.SampleVarianceY());
        fmt::print("correlation: {}\n", rCalc.Correlation());
    }

    SUBCASE("welford")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\nwelford\n");
        Operon::MeanVariance2 calc;
        for (size_t i = 0; i < xs.size(); ++i) {
            calc.Add(ys[i]);
        }
        fmt::print("mean: {}\n", calc.Mean());
        fmt::print("naive variance: {}\n", calc.NaiveVariance());
        fmt::print("sample variance: {}\n", calc.SampleVariance());
    }

    SUBCASE("youngs-cramer")
    {
        fmt::print(fmt::fg(fmt::color::light_green), "\nyoungs-cramer\n");
        Operon::MeanVariance3 calc;
        for (size_t i = 0; i < xs.size(); ++i) {
            calc.Add(ys[i]);
        }
        fmt::print("mean: {}\n", calc.Mean());
        fmt::print("naive variance: {}\n", calc.NaiveVariance());
        fmt::print("sample variance: {}\n", calc.SampleVariance());
    }
}

TEST_CASE("degenerate case")
{
    Operon::PearsonsRCalculator rCalc;
    Operon::MeanVarianceCalculator xCalc;
    Operon::MeanVarianceCalculator yCalc;
    Operon::MeanVariance2 calc2;

    std::vector<double> vec { 1e20, 1, -1e20 };

    for (auto v : vec) {
        xCalc.Add(v);
        calc2.Add(v);
    }

    fmt::print("shubert variance: {}\n", xCalc.NaiveVariance());
    fmt::print("welford variance: {}\n", calc2.NaiveVariance());

    xCalc.Reset();
    calc2.Reset();

    std::vector<double> vec2 { 1e20, -1e20, 1 };

    for (auto v : vec2) {
        xCalc.Add(v);
        calc2.Add(v);
    }

    fmt::print("shubert variance: {}\n", xCalc.NaiveVariance());
    fmt::print("welford variance: {}\n", calc2.NaiveVariance());
}
}
