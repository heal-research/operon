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

// Replicates the corrected NSGA2::UpdateDistance logic for a single front.
auto ComputeCrowdingDistance(Operon::Span<Individual> pop, std::vector<size_t> const& front) -> void
{
    if (front.empty()) { return; }
    size_t m = pop[front.front()].Fitness.size();
    auto inf = std::numeric_limits<Operon::Scalar>::max();

    for (auto idx : front) { pop[idx].Distance = 0; }

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
    Operon::Vector<Individual> pop(5);
    for (auto& ind : pop) { ind.Fitness.resize(2); }

    pop[0].Fitness = { 0.0F, 4.0F };
    pop[1].Fitness = { 1.0F, 3.0F };
    pop[2].Fitness = { 2.0F, 2.0F };
    pop[3].Fitness = { 3.0F, 1.0F };
    pop[4].Fitness = { 4.0F, 0.0F };

    std::vector<size_t> front = { 0, 1, 2, 3, 4 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[4].Distance));
    CHECK(std::isfinite(pop[1].Distance));
    CHECK(std::isfinite(pop[2].Distance));
    CHECK(std::isfinite(pop[3].Distance));
}

TEST_CASE("Interior individual has correct crowding distance", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(3);
    for (auto& ind : pop) { ind.Fitness.resize(2); }

    pop[0].Fitness = { 0.0F, 2.0F };
    pop[1].Fitness = { 1.0F, 1.0F };
    pop[2].Fitness = { 2.0F, 0.0F };

    std::vector<size_t> front = { 0, 1, 2 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[2].Distance));
    // obj0: (2-0)/(2-0)=1, obj1: (0-2)/(0-2)=1 → total 2
    CHECK_THAT(pop[1].Distance, Catch::Matchers::WithinAbs(2.0, 1e-5));
}

TEST_CASE("Single-individual front gets infinite crowding distance", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(1);
    pop[0].Fitness = { 1.0F, 2.0F };

    std::vector<size_t> front = { 0 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front);

    CHECK(std::isinf(pop[0].Distance));
}

TEST_CASE("Two-individual front: both boundary, both infinite", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(2);
    for (auto& ind : pop) { ind.Fitness.resize(2); }

    pop[0].Fitness = { 0.0F, 1.0F };
    pop[1].Fitness = { 1.0F, 0.0F };

    std::vector<size_t> front = { 0, 1 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[1].Distance));
}

TEST_CASE("Isolated point gets larger crowding distance", "[nsga2][crowding]")
{
    Operon::Vector<Individual> pop(4);
    for (auto& ind : pop) { ind.Fitness.resize(1); }

    pop[0].Fitness = { 0.0F };
    pop[1].Fitness = { 1.0F };
    pop[2].Fitness = { 2.0F };
    pop[3].Fitness = { 10.0F };

    std::vector<size_t> front = { 0, 1, 2, 3 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front);

    CHECK(std::isinf(pop[0].Distance));
    CHECK(std::isinf(pop[3].Distance));
    // pop[1]: (2-0)/10 = 0.2; pop[2]: (10-1)/10 = 0.9
    CHECK(std::isfinite(pop[1].Distance));
    CHECK(std::isfinite(pop[2].Distance));
    CHECK(pop[2].Distance > pop[1].Distance);
}

TEST_CASE("Normalization uses current front extremes, not global population", "[nsga2][crowding]")
{
    // Front 1 spans [1,3]; front 2 spans [100,300] — normalization must use front 1's range
    Operon::Vector<Individual> pop(6);
    for (auto& ind : pop) { ind.Fitness.resize(1); }

    pop[0].Fitness = { 1.0F };
    pop[1].Fitness = { 2.0F };
    pop[2].Fitness = { 3.0F };
    pop[3].Fitness = { 100.0F };
    pop[4].Fitness = { 200.0F };
    pop[5].Fitness = { 300.0F };

    std::vector<size_t> front1 = { 0, 1, 2 };
    ComputeCrowdingDistance({ pop.data(), pop.size() }, front1);

    // With correct normalization: (3-1)/(3-1) = 1.0
    // With global normalization:  (3-1)/(300-100) = 0.01
    CHECK_THAT(pop[1].Distance, Catch::Matchers::WithinAbs(1.0, 1e-5));
}

} // namespace Operon::Test
