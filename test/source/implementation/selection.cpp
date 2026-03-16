// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/selector.hpp"

namespace Operon::Test {

TEST_CASE("Tournament selection bias", "[operators]")
{
    constexpr size_t popSize = 100;
    constexpr size_t nSamples = 100'000;

    Operon::RandomGenerator rng(1234);

    Operon::Vector<Individual> individuals(popSize);
    for (size_t i = 0; i < popSize; ++i) {
        individuals[i].Fitness.resize(1);
        individuals[i][0] = static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(popSize);
    }

    TournamentSelector selector([](Individual const& a, Individual const& b) {
        return a.Fitness[0] < b.Fitness[0];
    });
    selector.Prepare(individuals);

    std::vector<size_t> hist(popSize, 0);
    for (size_t i = 0; i < nSamples; ++i) {
        hist[selector(rng)]++;
    }

    // Better fitness (lower index = lower value) should have higher selection probability
    // Check that the first quartile is selected more than the last quartile
    size_t firstQuartile = 0, lastQuartile = 0;
    for (size_t i = 0; i < popSize / 4; ++i) {
        firstQuartile += hist[i];
    }
    for (size_t i = 3 * popSize / 4; i < popSize; ++i) {
        lastQuartile += hist[i];
    }
    CHECK(firstQuartile > lastQuartile);
}

} // namespace Operon::Test
