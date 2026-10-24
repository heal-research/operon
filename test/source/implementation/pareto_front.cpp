// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "operon/core/individual.hpp"
#include "pareto_front.hpp"

namespace Operon::Test {

namespace {
    auto Make(std::vector<std::vector<Operon::Scalar>> const& fits, std::vector<size_t> const& ranks)
        -> std::vector<Individual>
    {
        std::vector<Individual> pop;
        for (auto i = 0UL; i < fits.size(); ++i) {
            Individual ind(fits[i].size());
            for (auto j = 0UL; j < fits[i].size(); ++j) {
                ind[j] = fits[i][j];
            }
            ind.Rank = ranks[i];
            pop.push_back(std::move(ind));
        }
        return pop;
    }
} // namespace

TEST_CASE("SelectParetoFront ignores stale ranks and uses fitness", "[cli][pareto]")
{
    // Ranks as left behind by sorting parents+offspring: the rank-0 member (0,0) was discarded by the reinserter, so
    // the retained individuals carry nonzero ranks although they are mutually non-dominated or dominate each other.
    auto pop = Make({ { 1, 4 }, { 2, 2 }, { 4, 1 }, { 3, 3 }, { 2, 2 }, { 5, 5 } }, { 1, 1, 2, 3, 0, 0 });
    auto const front = SelectParetoFront({ pop.data(), pop.size() });
    // (3,3) and (5,5) are dominated; (2,2) duplicate reported once (first); ordered by first objective.
    REQUIRE(front.size() == 3);
    CHECK(front[0] == &pop[0]);
    CHECK(front[1] == &pop[1]);
    CHECK(front[2] == &pop[2]);
}

TEST_CASE("SelectParetoFront on empty and single populations", "[cli][pareto]")
{
    std::vector<Individual> none;
    CHECK(SelectParetoFront({ none.data(), none.size() }).empty());
    auto one = Make({ { 1 } }, { 7 });
    CHECK(SelectParetoFront({ one.data(), one.size() }).size() == 1);
}

} // namespace Operon::Test
