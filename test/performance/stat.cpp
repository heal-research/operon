#include "nanobench.h"
#include <core/types.hpp>
#include <doctest/doctest.h>

#include <stat/meanvariance.hpp>
#include <stat/meanvariance2.hpp>
#include <stat/meanvariance3.hpp>

#include <random/random.hpp>
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

    TEST_CASE("random series")
    {
        constexpr int N = 1000000;

        std::vector<double> vec(N);
        Operon::RandomGenerator rng(1234);
        std::uniform_real_distribution<double> dist(-5, 5);
        std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });

        SUBCASE("performance comparison")
        {
            nb::Bench b;
            b.relative(true);
            b.performanceCounters(true);

            Operon::MeanVarianceCalculator calc1;
            Operon::MeanVariance2 calc2;
            Operon::MeanVariance3 calc3;

            auto [m2, v2] = TestCalculator(b, calc2, vec, "welford");

            b.run("welford-batch", [&]() {
                calc2.Reset();
                calc2.Add(vec);
            });

            auto meanWelfordBatch = calc2.Mean();
            auto varWelfordBatch = calc2.NaiveVariance();

            b.run("welford-eigen", [&]() {
                calc2.Reset();
                calc2.AddSIMD(vec);
            });

            auto meanWelfordEigen = calc2.Mean();
            auto varWelfordEigen = calc2.NaiveVariance();

            auto [m1, v1] = TestCalculator(b, calc1, vec, "operon-youngs-cramer");

            b.run("operon-youngs-cramer-two-pass", [&]() {
                calc1.Reset();
                calc1.AddTwoPass(vec);
            });

            auto meanOperonTwoPass = calc1.Mean();
            auto varOperonTwoPass = calc1.NaiveVariance();

            b.run("operon-youngs-cramer-eigen", [&]() {
                calc1.Reset();
                calc1.Add(vec);
            });

            auto meanOperonEigen = calc1.Mean();
            auto varOperonEigen = calc1.NaiveVariance();

            auto [m3, v3] = TestCalculator(b, calc3, vec, "youngs-cramer");

            b.run("youngs-cramer-batch", [&]() {
                calc3.Reset();
                calc3.Add(vec);
            });

            auto meanYCBatch = calc3.Mean();
            auto varYCBatch = calc3.NaiveVariance();

            b.run("youngs-cramer-eigen", [&]() {
                calc3.Reset();
                calc3.AddSIMD(vec);
            });

            auto meanYCEigen = calc3.Mean();
            auto varYCEigen = calc3.NaiveVariance();

            fmt::print("operon:              [{}, {}]\n", m1, v1);
            fmt::print("operon-twopass:      [{}, {}]\n", meanOperonTwoPass, varOperonTwoPass);
            fmt::print("operon-eigen         [{}, {}]\n", meanOperonEigen, varOperonEigen);
            fmt::print("welford:             [{}, {}]\n", m2, v2);
            fmt::print("welford-batch        [{}, {}]\n", meanWelfordBatch, varWelfordBatch);
            fmt::print("welford-eigen        [{}, {}]\n", meanWelfordEigen, varWelfordEigen);
            fmt::print("youngs-cramer:       [{}, {}]\n", m3, v3);
            fmt::print("youngs-cramer-batch: [{}, {}]\n", meanYCBatch, varYCBatch);
            fmt::print("youngs-cramer-eigen: [{}, {}]\n", meanYCEigen, varYCEigen);
        }

        SUBCASE("schubert")
        {
            Operon::MeanVarianceCalculator calc;
            nb::Bench b;
            b.performanceCounters(true);
            auto [m, v] = TestCalculator(b, calc, vec, "schubert");
            fmt::print("[{}, {}]\n", m, v);
        }

        SUBCASE("welford")
        {
            Operon::MeanVariance2 calc;
            nb::Bench b;
            b.performanceCounters(true);
            auto [m, v] = TestCalculator(b, calc, vec, "welford");
            fmt::print("[{}, {}]\n", m, v);
            fmt::print("ss = {}\n", calc.SS());

            b.run("welford-batch", [&]() {
                calc.Reset();
                calc.Add(vec);
            });
            fmt::print("[{}, {}]\n", calc.Mean(), calc.NaiveVariance());
            fmt::print("ss = {}\n", calc.SS());

            b.run("welford-eigen", [&]() {
                calc.Reset();
                calc.AddSIMD(vec);
            });
            fmt::print("[{}, {}]\n", calc.Mean(), calc.NaiveVariance());
            fmt::print("ss = {}\n", calc.SS());
        }

        SUBCASE("youngs-cramer")
        {
            Operon::MeanVariance3 calc;
            nb::Bench b;
            b.performanceCounters(true);
            auto [m, v] = TestCalculator(b, calc, vec, "youngs-cramer");
            fmt::print("[{}, {}]\n", m, v);
        }
    }

    TEST_CASE("correlation")
    {
        Operon::PearsonsRCalculator calc;

        constexpr int N = 1000000;
        Operon::Vector<double> x(N), y(N);
        Operon::RandomGenerator rng(1234);
        std::uniform_real_distribution<double> dist(-5, 5);
        std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
        std::generate(y.begin(), y.end(), [&]() { return dist(rng); });
        //std::vector<double> x { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        //std::vector<double> y { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        SUBCASE("performance comparison")
        {
            nb::Bench b;
            b.performanceCounters(true).relative(true).minEpochIterations(100);

            calc.Reset();
            for (size_t i = 0; i < N; ++i) {
                calc.Add(x[i], y[i]);
            }
            fmt::print(" SEQ  sumWe: {:.20}, sumX: {:.20}, sumY: {:.20}, sumXX: {:.20}, sumYY: {:.20}, sumXY: {:.20}\n", calc.SumWe(), calc.SumX(), calc.SumY(), calc.SumXX(), calc.SumYY(), calc.SumXY());

            calc.Reset();
            calc.Add(x, y);
            fmt::print("SIMD  sumWe: {:.20}, sumX: {:.20}, sumY: {:.20}, sumXX: {:.20}, sumYY: {:.20}, sumXY: {:.20}\n", calc.SumWe(), calc.SumX(), calc.SumY(), calc.SumXX(), calc.SumYY(), calc.SumXY());

            b.run("correlation      ", [&]() { calc.Reset(); 
                for (size_t i = 0; i < N; ++i) {
                    calc.Add(x[i], y[i]);
                }
            });
            b.run("correlation-simd", [&]() { calc.Reset(); calc.Add(x, y); });

            SUBCASE("correlation")
            {
                nb::Bench b1;
                b1.performanceCounters(true).relative(true).minEpochIterations(10);
                b1.run("correlation      ", [&]() {
                    calc.Reset();
                    for (size_t i = 0; i < N; ++i) {
                        calc.Add(x[i], y[i]);
                    }
                });
            }

            SUBCASE("correlation-simd")
            {
                nb::Bench b1;
                b1.performanceCounters(true).relative(true).minEpochIterations(10);
                b1.run("correlation-simd ", [&]() { calc.Reset(); calc.Add(x, y); });
            }
        }
    }
}
}
