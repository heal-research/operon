// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "random/random.hpp"
#include "nanobench.h"
#include <doctest/doctest.h>
#include <random>

namespace nb = ankerl::nanobench;

namespace Operon {

template <typename Rng>
void bench(ankerl::nanobench::Bench* bench, char const* name)
{
    std::random_device dev;
    Rng rng(dev());

    bench->run(name, [&]() {
        auto r = std::uniform_int_distribution<uint64_t> {}(rng);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
}

TEST_CASE("Random number generators") {
    ankerl::nanobench::Bench b;
    b.title("rng name")
        .unit("uint64_t")
        .warmup(100)
        .relative(true);
    b.performanceCounters(true);

    bench<std::default_random_engine>(&b, "std::default_random_engine");
    bench<std::mt19937>(&b, "std::mt19937");
    bench<std::mt19937_64>(&b, "std::mt19937_64");
    bench<std::ranlux24_base>(&b, "std::ranlux24_base");
    bench<std::ranlux48_base>(&b, "std::ranlux48_base");
    bench<std::ranlux24>(&b, "std::ranlux24_base");
    bench<std::ranlux48>(&b, "std::ranlux48");
    bench<std::knuth_b>(&b, "std::knuth_b");
    bench<Operon::Random::Jsf64>(&b, "Operon::RandomGenerator::Jsf64");
    bench<Operon::Random::RomuDuo>(&b, "Operon::RandomGenerator::RomuDuo");
    bench<Operon::Random::RomuTrio>(&b, "Operon::RandomGenerator::RomuTrio");
    bench<Operon::Random::Sfc64>(&b, "Operon::RandomGenerator::Sfc64");
    bench<nb::Rng>(&b, "ankerl::nanobench::Rng");
}

}
