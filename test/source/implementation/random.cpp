// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cstddef>
#include <doctest/doctest.h>
#include <fmt/core.h>
#include <random>
#include <vector>

#include "operon/random/random.hpp"
#include "operon/random/romu.hpp"

namespace dt = doctest;

namespace Operon::Test {

TEST_CASE("random sampling" * dt::test_suite("[implementation]"))
{
    std::vector<size_t> vec{0,1,2,3,4,5,6,7,8,9};

    size_t samples = 1'000'000;

    Operon::Random::RomuTrio rng(std::random_device{}());

    SUBCASE("uniform") {
        std::vector<size_t> counts(vec.size(), 0);


        for (size_t i = 0; i < samples; ++i) {
            counts[*Operon::Random::Sample(rng, vec.begin(), vec.end())]++;
        }

        for (auto c : counts)
            fmt::print("{} ", c);
        fmt::print("\n");
    }

    SUBCASE("uniform with condition") {
        std::vector<size_t> counts(vec.size(), 0);

        auto condition = [](auto v) { return v % 2 == 0; };

        for (size_t i = 0; i < samples; ++i) {
            counts[*Operon::Random::Sample(rng, vec.begin(), vec.end(), condition)]++;
        }

        for (auto c : counts)
            fmt::print("{} ", c);
        fmt::print("\n");
    }
}

} // namespace
