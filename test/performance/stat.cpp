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
        const int N = int(1e5);

        Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1> x = Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>::Random(N, 1);

        decltype(x) y = 5 * x + 3;

        nb::Bench bench;
        bench.title("Stat").relative(true).performanceCounters(true).minEpochIterations(100);

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

    }
}
}
