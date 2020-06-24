#include <doctest/doctest.h>
#include <random>
#include <fmt/core.h>

#include "nanobench.h"

#include "core/types.hpp"
#include "core/common.hpp"

namespace nb = ankerl::nanobench;

namespace Operon {

TEST_CASE("Libm functions cost model")
{

    Operon::Random rand(std::random_device{}());

    SUBCASE("double-precision") {
        nb::Bench b;
        b.title("operation").relative(true).performanceCounters(true).minEpochIterations(1e6);
        std::uniform_real_distribution<double> dist(Operon::Numeric::Min<double>(), Operon::Numeric::Max<double>());
        // binary
        b.run("+", [&]() { nb::doNotOptimizeAway(dist(rand) + dist(rand)); });
        b.run("-", [&]() { nb::doNotOptimizeAway(dist(rand) - dist(rand)); });
        b.run("*", [&]() { nb::doNotOptimizeAway(dist(rand) * dist(rand)); });
        b.run("/", [&]() { nb::doNotOptimizeAway(dist(rand) / dist(rand)); });

        // unary
        b.run("exp", [&]() { nb::doNotOptimizeAway(std::exp(dist(rand))); });
        b.run("log", [&]() { nb::doNotOptimizeAway(std::log(dist(rand))); });
        b.run("sin", [&]() { nb::doNotOptimizeAway(std::sin(dist(rand))); });
        b.run("cos", [&]() { nb::doNotOptimizeAway(std::cos(dist(rand))); });
        b.run("tan", [&]() { nb::doNotOptimizeAway(std::tan(dist(rand))); });
    }

    fmt::print("\n\n");

    SUBCASE("single-precision") {
        nb::Bench b;
        b.title("operation").relative(true).performanceCounters(true).minEpochIterations(1e6);
        std::uniform_real_distribution<float> dist(Operon::Numeric::Min<float>(), Operon::Numeric::Max<float>());
        // binary
        b.run("+", [&]() { nb::doNotOptimizeAway(dist(rand) + dist(rand)); });
        b.run("-", [&]() { nb::doNotOptimizeAway(dist(rand) - dist(rand)); });
        b.run("*", [&]() { nb::doNotOptimizeAway(dist(rand) * dist(rand)); });
        b.run("/", [&]() { nb::doNotOptimizeAway(dist(rand) / dist(rand)); });

        // unary
        b.run("exp", [&]() { nb::doNotOptimizeAway(std::exp(dist(rand))); });
        b.run("log", [&]() { nb::doNotOptimizeAway(std::log(dist(rand))); });
        b.run("sin", [&]() { nb::doNotOptimizeAway(std::sin(dist(rand))); });
        b.run("cos", [&]() { nb::doNotOptimizeAway(std::cos(dist(rand))); });
        b.run("tan", [&]() { nb::doNotOptimizeAway(std::tan(dist(rand))); });
    }
}

}
