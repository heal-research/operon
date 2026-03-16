// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <operon/core/types.hpp>
#include <operon/random/random.hpp>

namespace Operon::Test {

TEST_CASE("Uniform random sampling", "[core]")
{
    std::vector<size_t> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t samples = 1'000'000;
    Operon::RandomGenerator rng(1234);

    SECTION("All indices are sampled") {
        std::vector<size_t> counts(vec.size(), 0);

        for (size_t i = 0; i < samples; ++i) {
            counts[*Operon::Random::Sample(rng, vec.begin(), vec.end())]++;
        }

        for (size_t i = 0; i < counts.size(); ++i) {
            CHECK(counts[i] > 0);
        }
    }

    SECTION("Samples are within bounds") {
        for (size_t i = 0; i < 1000; ++i) {
            auto val = *Operon::Random::Sample(rng, vec.begin(), vec.end());
            CHECK(val < vec.size());
        }
    }
}

TEST_CASE("Conditional random sampling", "[core]")
{
    std::vector<size_t> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t samples = 1'000'000;
    Operon::RandomGenerator rng(1234);

    auto condition = [](auto v) { return v % 2 == 0; };

    std::vector<size_t> counts(vec.size(), 0);

    for (size_t i = 0; i < samples; ++i) {
        counts[*Operon::Random::Sample(rng, vec.begin(), vec.end(), condition)]++;
    }

    // Odd indices should never be sampled
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i % 2 == 0) {
            CHECK(counts[i] > 0);
        } else {
            CHECK(counts[i] == 0);
        }
    }
}

} // namespace Operon::Test
