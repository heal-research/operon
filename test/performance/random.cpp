// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "random/random.hpp"
#include "nanobench.h"
#include <doctest/doctest.h>
#include <limits>
#include <random>

#include "core/types.hpp"

namespace nb = ankerl::nanobench;

namespace Operon {

template <typename Rng>
inline void bench(ankerl::nanobench::Bench* bench, char const* name)
{
    std::random_device dev;
    Rng rng(dev());

    uint64_t sum = 0;
    bench->batch(1).run(name, [&]() {
        auto r = rng();
        ankerl::nanobench::doNotOptimizeAway(sum += r);
    });
}

TEST_CASE("Random number generators") {
    ankerl::nanobench::Bench b;
    b.title("rng name")
        //.unit("uint64_t")
        .warmup(100)
        .minEpochIterations(100);
    b.performanceCounters(true);

    SUBCASE("std::random") {
        bench<std::default_random_engine>(&b, "std::default_random_engine");
        bench<std::mt19937>(&b, "std::mt19937");
        bench<std::mt19937_64>(&b, "std::mt19937_64");
        bench<std::ranlux24_base>(&b, "std::ranlux24_base");
        bench<std::ranlux48_base>(&b, "std::ranlux48_base");
        bench<std::ranlux24>(&b, "std::ranlux24_base");
        bench<std::ranlux48>(&b, "std::ranlux48");
        bench<std::knuth_b>(&b, "std::knuth_b");
    }

    SUBCASE("Jsf64") {
        bench<Operon::Random::Jsf64>(&b, "Operon::RandomGenerator::Jsf64");
    }

    SUBCASE("RomuDuo") {
        bench<Operon::Random::RomuDuo>(&b, "Operon::RandomGenerator::RomuDuo");
    }

    SUBCASE("RomuTrio") {
        bench<Operon::Random::RomuTrio>(&b, "Operon::RandomGenerator::RomuTrio");
    }

    SUBCASE("Sfc64") {
        bench<Operon::Random::Sfc64>(&b, "Operon::RandomGenerator::Sfc64");
    }

    SUBCASE("Wyrand") {
        bench<Operon::Random::Wyrand>(&b, "Operon::RandomGenerator::Wyrand");
    }

    SUBCASE("ankerl::nanobench::Rng (RomuDuoJr)") {
        bench<nb::Rng>(&b, "ankerl::nanobench::Rng");
    }
}

}
