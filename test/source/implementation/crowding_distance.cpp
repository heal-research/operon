// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"

namespace Operon::Test {
namespace {

// Replicates NSGA2::UpdateDistance logic for a single front (used to verify correctness).
auto ComputeCrowdingDistance(Operon::Span<Individual> pop, std::vector<size_t> const& front) -> void
{
    if (front.empty()) { return; }
    size_t m = pop[front.front()].Fitness.size();
    auto inf = std::numeric_limits<Operon::Scalar>::infinity();

    // Initialize Distance = 0
    for (auto idx : front) {
        pop[idx].Distance = 0;
    }

    for (size_t obj = 0; obj < m; ++obj) {
        SingleObjectiveComparison comp(obj);
        std::vector<size_t> sorted(front.begin(), front.end());
        std::stable_sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) -> bool { return comp(pop[a], pop[b]); });

        auto min = pop[sorted.front()][obj];
        auto max = pop[sorted.back()][obj];

        for (size_t j = 0; j < sorted.size(); ++j) {
            auto mPrev = j > 0 ? pop[sorted[j - 1]][obj] : inf;
            auto mNext = j < sorted.size() - 1 ? pop[sorted[j + 1]][obj] : inf;
            auto distance = (mNext - mPrev) / (max - min);

            // Boundary points get infinite distance
            if (j == 0 || j == sorted.size() - 1) {
                distance = inf;
            } else if (!std::isfinite(distance)) {
                distance = 0;
            }
            pop[sorted[j]].Distance += distance;
        }
    }
}

} // namespace

TEST_CASE("Boundary individuals get infinite crowding distance", "[nsga2][crowding]")
{
    // Front of 5 individuals with 2 objectives
    Operon::Vector<Individual> pop(5);
    for (auto& ind : pop) {
        ind.Fitness.resize(2);
    }

    // Objective 0: linearly increasing
    // Objective 1: linearly decreasing
    // This creates a classic Pareto front
    pop[0].Fitness = { 0.0f, 4.0f };
    pop[1].Fitness = { 1.0f, 3.0f };
    pop[2].Fitness = { 2.0f, 2.0f };
    pop[3].Fitness = { 3.0f, 1.0f };
    pop[4].Fitness = { 4.0f, 0.0f };

    std::vector<size_t> front = {0, 1, 2, 3, 4};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front);

    // Boundary individuals (0 and 4) should have infinite distance
    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[4].Distance));

    // Interior individuals should have finite distance
    CHECK(std::isfinite(pop[1].Distance));
    CHECK(std::isfinite(pop[2].Distance));
    CHECK(std::isfinite(pop[3].Distance));
}

TEST_CASE("Interior individuals have correct crowding distance", "[nsga2][crowding]")
{
    // Front of 3 individuals with 2 objectives on a uniform grid
    Operon::Vector<Individual> pop(3);
    for (auto& ind : pop) {
        ind.Fitness.resize(2);
    }

    pop[0].Fitness = { 0.0f, 2.0f };
    pop[1].Fitness = { 1.0f, 1.0f };
    pop[2].Fitness = { 2.0f, 0.0f };

    std::vector<size_t> front = {0, 1, 2};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front);

    // Boundary individuals — infinite distance
    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[2].Distance));

    // Middle individual: distance for obj 0 = (2-0)/(2-0) = 1.0
    //                     distance for obj 1 = (0-2)/(0-2) = 1.0
    //                     total = 2.0
    CHECK_THAT(pop[1].Distance, Catch::Matchers::WithinAbs(2.0, 1e-5));
}

TEST_CASE("Single-individual front gets infinite distance", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(1);
    pop[0].Fitness = { 1.0f, 2.0f };

    std::vector<size_t> front = {0};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front);

    CHECK(std::isinf(pop[0].Distance));
}

TEST_CASE("Two-individual front: both boundary, both infinite", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(2);
    for (auto& ind : pop) {
        ind.Fitness.resize(2);
    }

    pop[0].Fitness = { 0.0f, 1.0f };
    pop[1].Fitness = { 1.0f, 0.0f };

    std::vector<size_t> front = {0, 1};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[1].Distance));
}

TEST_CASE("Non-uniform spacing produces larger distance for isolated points", "[nsga2][crowding]")
{
    // 4 individuals with non-uniform distribution on objective 0
    Operon::Vector<Individual> pop(4);
    for (auto& ind : pop) {
        ind.Fitness.resize(1); // single objective for simplicity
    }

    pop[0].Fitness = { 0.0f };
    pop[1].Fitness = { 1.0f };
    pop[2].Fitness = { 2.0f };
    pop[3].Fitness = { 10.0f };

    std::vector<size_t> front = {0, 1, 2, 3};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[3].Distance));

    // pop[2] is isolated further from pop[3] (gap = 8) than pop[1] from neighbors (gap = 1)
    // distance[1] = (2-0)/(10-0) = 0.2
    // distance[2] = (10-1)/(10-0) = 0.9
    CHECK(std::isfinite(pop[1].Distance));
    CHECK(std::isfinite(pop[2].Distance));
    CHECK(pop[2].Distance > pop[1].Distance);
}

TEST_CASE("Front uses current front extremes, not global population", "[nsga2][crowding]")
{
    // 6 individuals: 3 in first front, 3 in second front
    // Verify that normalization uses front boundaries, not global population boundaries
    Operon::Vector<Individual> pop(6);
    for (auto& ind : pop) {
        ind.Fitness.resize(1);
    }

    // Front 1: individuals with values 1.0, 2.0, 3.0
    pop[0].Fitness = { 1.0f };
    pop[1].Fitness = { 2.0f };
    pop[2].Fitness = { 3.0f };

    // Front 2: individuals with values 100.0, 200.0, 300.0 (not in this front)
    pop[3].Fitness = { 100.0f };
    pop[4].Fitness = { 200.0f };
    pop[5].Fitness = { 300.0f };

    // Compute distance only for front 1
    std::vector<size_t> front1 = {0, 1, 2};
    ComputeCrowdingDistance(Operon::Span<Individual>{pop.data(), pop.size()}, front1);

    // Middle individual: distance = (3.0 - 1.0) / (3.0 - 1.0) = 1.0
    CHECK_THAT(pop[1].Distance, Catch::Matchers::WithinAbs(1.0, 1e-5));

    // If normalization used global boundaries (100, 300),
    // distance would be = (3-1)/(300-100) = 0.01, which is incorrect
}

} // namespace Operon::Test
