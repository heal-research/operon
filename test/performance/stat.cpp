// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "nanobench.h"
#include <core/types.hpp>
#include <doctest/doctest.h>

#include <random/random.hpp>
#include <stat/linearscaler.hpp>
#include <stat/meanvariance.hpp>
#include <stat/pearson.hpp>

namespace nb = ankerl::nanobench;

namespace Operon {
namespace Test {
    template <typename T>
    std::pair<double, double> TestCalculator(nb::Bench& b, T& calc, std::vector<double> const& vec, std::string const& name)
    {
        double var = 0, mean = 0;
        b.run(name, [&]() {
            calc.Reset();
            for (auto v : vec) {
                calc.Add(v);
            }
            mean = calc.Mean();
            var = calc.NaiveVariance();
        });
        return { mean, var };
    }

    TEST_CASE("Stat")
    {
        const int N = int(1e6);

        Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1> x = Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>::Random(N, 1);

        decltype(x) y = 5 * x + 3;

        nb::Bench bench;
        bench.title("Stat").batch(N).performanceCounters(true).minEpochIterations(100);

        gsl::span<Operon::Scalar const> xx(x.data(), x.size());
        gsl::span<Operon::Scalar const> yy(y.data(), y.size());

        auto [a1, b1] = Operon::LinearScalingCalculator::Calculate(xx.begin(), xx.end(), yy.begin());
        auto [a2, b2] = Operon::LinearScalingCalculator::Calculate(xx, yy);

        double a3, b3;
        Operon::PearsonsRCalculator calc;
        calc.Add(xx, yy);
        a3 = calc.SampleCovariance() / calc.SampleVarianceX();
        b3 = calc.MeanY() - a3 * calc.MeanX();

        CHECK(a1 == a2);
        CHECK(b1 == b2);

        CHECK(a1 == a3);
        CHECK(b1 == b3);

        double f = 0;

        bench.run("ls batch", [&]() {
            auto [a, b] = Operon::LinearScalingCalculator::Calculate(xx, yy);
            f += a + b;

        });

        bench.run("ls online", [&]() {
            auto [a, b] = Operon::LinearScalingCalculator::Calculate(xx.begin(), xx.end(), yy.begin());
            f += a + b;
        });

        bench.run("corr batch", [&]() {
            calc.Reset();
            calc.Add(gsl::span<Operon::Scalar const>(x.data(), x.size()), gsl::span<Operon::Scalar const>(y.data(), y.size()));
            f += calc.Correlation();
        });

        bench.run("var batch", [&]() {
            Operon::MeanVarianceCalculator mv;
            mv.Add(gsl::span<Operon::Scalar const>(x.data(), x.size()));
            f += mv.NaiveVariance();
        });

        std::vector<int> sizes { 1000, 10000 };
        int step = int(5e4);
        for (int s = step; s <= N; s += step) {
            sizes.push_back(s);
        }

        std::vector<float> xf(N), yf(N);
        std::vector<double> xd(N), yd(N);

        std::default_random_engine rng(1234);
        std::uniform_real_distribution<double> dist(-1, 1);

        std::generate(xd.begin(), xd.end(), [&]() { return dist(rng); });
        std::generate(yd.begin(), yd.end(), [&]() { return dist(rng); });

        std::copy(xd.begin(), xd.end(), xf.begin());
        std::copy(yd.begin(), yd.end(), yf.begin());

        double var = 0, count = 0;
        for (auto s : sizes) {
            bench.batch(s).run("univariate float/float " + std::to_string(s), [&]() {
                ++count;
                Operon::MeanVarianceCalculator mv;
                mv.Add(gsl::span<decltype(xf)::value_type const>(xf.data(), size_t(s)));
                var += mv.NaiveVariance();
            });
        }

        var = count = 0;
        for (auto s : sizes) {
            bench.batch(s).run("univariate double/double " + std::to_string(s), [&]() {
                ++count;
                Operon::MeanVarianceCalculator mv;
                mv.Add(gsl::span<decltype(xd)::value_type const>(xd.data(), size_t(s)));
                var += mv.NaiveVariance();
            });
        }

        double corr = 0;
        count = 0;
        for (auto s : sizes) {
            bench.batch(s).run("bivariate float/float " + std::to_string(s), [&]() {
                ++count;
                Operon::PearsonsRCalculator pc;
                pc.Add(gsl::span<decltype(xf)::value_type const>(xf.data(), s), gsl::span<decltype(yf)::value_type const>(yf.data(), s));
                corr += pc.Correlation();
            });
        }

        corr = count = 0;
        for (auto s : sizes) {
            bench.batch(s).run("bivariate double/double" + std::to_string(s), [&]() {
                ++count;
                Operon::PearsonsRCalculator pc;
                pc.Add(gsl::span<decltype(xd)::value_type const>(xd.data(), s), gsl::span<decltype(yd)::value_type const>(yd.data(), s));
                corr += pc.Correlation();
            });
        }
    }
}
}
