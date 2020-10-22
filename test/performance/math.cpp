#include <doctest/doctest.h>
#include <random>
#include <fmt/core.h>

#include "nanobench.h"

#include "core/types.hpp"
#include "core/common.hpp"

#include <Eigen/Core>

namespace nb = ankerl::nanobench;

namespace Operon {

TEST_CASE("Libm functions cost model")
{
    Operon::RandomGenerator rand(std::random_device{}());

    int rows = 1000, cols = 100;

    std::uniform_int_distribution<int> dist(0, cols-1);

    SUBCASE("double-precision") {
        std::uniform_real_distribution<double> uniformDouble(Operon::Numeric::Min<double>(), Operon::Numeric::Max<double>());
        Eigen::ArrayXd mDouble(rows, cols); mDouble.unaryExpr([&](double){ return uniformDouble(rand); });

        nb::Bench b;
        b.title("double-precision").relative(true).performanceCounters(true).minEpochIterations(1000);

        // binary
        b.run("+", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) += mDouble.col(dist(rand))); });
        b.run("-", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) -= mDouble.col(dist(rand))); });
        b.run("*", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) *= mDouble.col(dist(rand))); });
        b.run("/", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) /= mDouble.col(dist(rand))); });

        // unary
        b.run("exp", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).exp()); });
        b.run("log", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).log()); });
        b.run("sin", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).sin()); });
        b.run("cos", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).cos()); });
        b.run("tan", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).tan()); });
        b.run("sqrt", [&]() { nb::doNotOptimizeAway(mDouble.col(dist(rand)) = mDouble.col(dist(rand)).sqrt()); });
    }


    SUBCASE("single-precision") {
        std::uniform_real_distribution<float> uniformFloat(Operon::Numeric::Min<float>(), Operon::Numeric::Max<float>());
        Eigen::ArrayXf mFloat(rows, cols); mFloat.unaryExpr([&](float){ return uniformFloat(rand); });

        nb::Bench b;
        b.title("single-precision").relative(true).performanceCounters(true).minEpochIterations(1000);

        // binary
        b.run("+", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) += mFloat.col(dist(rand))); });
        b.run("-", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) -= mFloat.col(dist(rand))); });
        b.run("*", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) *= mFloat.col(dist(rand))); });
        b.run("/", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) /= mFloat.col(dist(rand))); });

        // unary
        b.run("exp", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).exp()); });
        b.run("log", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).log()); });
        b.run("sin", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).sin()); });
        b.run("cos", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).cos()); });
        b.run("tan", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).tan()); });
        b.run("sqrt", [&]() { nb::doNotOptimizeAway(mFloat.col(dist(rand)) = mFloat.col(dist(rand)).sqrt()); });
    }
}

}
