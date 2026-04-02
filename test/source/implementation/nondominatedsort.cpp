// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <functional>
#include <ranges>
#include <random>

#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace Operon::Test {

namespace {
auto InitializePop(Operon::RandomGenerator& random, auto& dist, size_t n, size_t m) {
    Operon::Vector<Individual> individuals(n);
    for (auto& individual : individuals) {
        individual.Fitness.resize(m);
        for (size_t j = 0; j < m; ++j) {
            individual[j] = dist(random);
        }
    }
    std::stable_sort(individuals.begin(), individuals.end(), [](auto const& a, auto const& b) -> auto { return std::ranges::lexicographical_compare(a.Fitness, b.Fitness); });

    for (auto i = individuals.begin(); i < individuals.end();) {
        i->Rank = 0;
        auto j = i + 1;
        for (; j < individuals.end() && i->Fitness == j->Fitness; ++j) {
            j->Rank = 1;
        }
        i = j;
    }
    auto r = std::stable_partition(individuals.begin(), individuals.end(), [](auto const& ind) -> auto { return !ind.Rank; });
    Operon::Vector<Individual> pop(individuals.begin(), r);
    return pop;
}
} // namespace

TEST_CASE("Hand-crafted Pareto fronts", "[algorithms]")
{
    SECTION("2D points with known fronts") {
        Operon::Vector<Operon::Vector<Operon::Scalar>> points = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {10, 2}, {6, 6}, {9, 5}};
        Operon::Vector<Individual> pop(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            pop[i].Fitness = points[i];
        }

        auto fronts = RankIntersectSorter{}(pop);
        CHECK(!fronts.empty());

        // First front should contain some of the Pareto-optimal points
        CHECK(!fronts[0].empty());
    }

    SECTION("3D points") {
        Operon::Vector<Operon::Vector<Operon::Scalar>> points = {{1, 2, 3}, {-2, 3, 7}, {-1, -2, -3}, {0, 0, 0}};
        Operon::Vector<Individual> pop(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            pop[i].Fitness = points[i];
        }
        std::stable_sort(pop.begin(), pop.end(), LexicographicalComparison{});

        auto fronts = DeductiveSorter{}(pop);
        CHECK(!fronts.empty());
        CHECK(!fronts[0].empty());
    }
}

TEST_CASE("All sorters produce same ranking", "[algorithms]")
{
    constexpr Operon::RandomGenerator::result_type seed{1234};

    auto compareSorters = [&](auto const& s1, auto const& s2, auto const& ns, auto const& ms) -> auto {
        Operon::RandomGenerator rd(seed); // fresh RNG per sorter pair so all pairs see the same populations
        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        for (auto n : ns) {
            for (auto m : ms) {
                auto pop = InitializePop(rd, dist, n, m);

                auto f1 = s1.Sort(pop, 0);
                auto f2 = s2.Sort(pop, 0);

                if (f1.size() != f2.size()) {
                    return false;
                }
                for (auto i = 0; i < std::ssize(f1); ++i) {
                    std::ranges::stable_sort(f1[i]);
                    std::ranges::stable_sort(f2[i]);
                    if (f1[i] != f2[i]) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    std::array ns{100, 1000, 5000};
    std::array ms{2, 3, 4, 5, 10};
    Operon::RankIntersectSorter rs;
    Operon::RankOrdinalSorter ro;
    Operon::MergeSorter mnds;
    Operon::BestOrderSorter bos;
    Operon::HierarchicalSorter hnds;
    Operon::DeductiveSorter ds;
    Operon::EfficientBinarySorter ebs;
    Operon::EfficientSequentialSorter ess;

    // DeductiveSorter is used as the reference (simplest, most obviously correct implementation).
    // All sorters including RankIntersectSorter are compared against it.
    Operon::Vector<std::reference_wrapper<NondominatedSorterBase const>> compared{rs, ro, mnds, bos, hnds, ebs, ess};
    for (auto i = 0; i < std::ssize(compared); ++i) {
        auto const& sorter = compared[i].get();
        auto res = compareSorters(ds, sorter, ns, ms);
        CHECK(res);
    }
}

TEST_CASE("Non-dominated sort edge cases", "[algorithms]")
{
    Operon::RandomGenerator rd(1234);

    SECTION("Single objective") {
        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        auto pop = InitializePop(rd, dist, 100, 1);
        auto fronts = RankIntersectSorter{}(pop);
        CHECK(!fronts.empty());
    }

    SECTION("Non-dominated 2D points in the same front") {
        // Points that are trade-offs (neither dominates the other) must all land in front 0.
        Operon::Vector<Operon::Vector<Operon::Scalar>> points = {
            {0.0F, 1.0F}, {0.5F, 0.5F}, {1.0F, 0.0F}
        };
        Operon::Vector<Individual> pop(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            pop[i].Fitness = points[i];
        }
        auto fronts = RankIntersectSorter{}(pop);
        CHECK(fronts.size() == 1);
        CHECK(fronts[0].size() == pop.size());
    }
}

// Tagged [.] so it only runs when explicitly requested: operon_test "[.][minseed]"
// Sweeps many seeds at small n to find the minimum population size that triggers
// the RankIntersectSorter vs DeductiveSorter disagreement.
TEST_CASE("RankIntersect minimum reproducer search", "[.][minseed]")
{
    RankIntersectSorter rs;
    DeductiveSorter    ds;
    std::uniform_real_distribution<Operon::Scalar> dist(0, 1);

    std::array ns{10, 20, 50, 100, 200, 500};
    std::array ms{2, 3, 4, 5};
    constexpr int nseeds{200};

    for (auto n : ns) {
        for (auto m : ms) {
            int hits{0};
            int first_seed{-1};
            Operon::RandomGenerator::result_type first_seed_val{};
            for (int s = 0; s < nseeds; ++s) {
                Operon::RandomGenerator rd(static_cast<Operon::RandomGenerator::result_type>(s));
                auto pop = InitializePop(rd, dist, n, m);
                auto f_rs = rs.Sort(pop, 0);
                auto f_ds = ds.Sort(pop, 0);
                Operon::Vector<int> rank_rs(pop.size(), -1);
                Operon::Vector<int> rank_ds(pop.size(), -1);
                for (auto fi = 0; fi < std::ssize(f_rs); ++fi)
                    for (auto idx : f_rs[fi]) rank_rs[idx] = fi;
                for (auto fi = 0; fi < std::ssize(f_ds); ++fi)
                    for (auto idx : f_ds[fi]) rank_ds[idx] = fi;
                bool any = false;
                for (size_t i = 0; i < pop.size(); ++i) {
                    if (rank_rs[i] != rank_ds[i]) { any = true; break; }
                }
                if (any) {
                    if (first_seed < 0) { first_seed = s; first_seed_val = static_cast<Operon::RandomGenerator::result_type>(s); }
                    ++hits;
                }
            }
            if (hits > 0) {
                fmt::println("n={:5d} m={}: DISAGREE in {:3d}/{} seeds  (first seed={})", n, m, hits, nseeds, first_seed_val);
                // Print the disagreeing individuals for the first triggering seed
                Operon::RandomGenerator rd(first_seed_val);
                auto pop = InitializePop(rd, dist, n, m);
                auto f_rs = rs.Sort(pop, 0);
                auto f_ds = ds.Sort(pop, 0);
                Operon::Vector<int> rank_rs(pop.size(), -1);
                Operon::Vector<int> rank_ds(pop.size(), -1);
                for (auto fi = 0; fi < std::ssize(f_rs); ++fi)
                    for (auto idx : f_rs[fi]) rank_rs[idx] = fi;
                for (auto fi = 0; fi < std::ssize(f_ds); ++fi)
                    for (auto idx : f_ds[fi]) rank_ds[idx] = fi;
                for (size_t i = 0; i < pop.size(); ++i) {
                    if (rank_rs[i] != rank_ds[i]) {
                        fmt::println("  ind {:4d}  fitness={}  RankIntersect={}  Deductive={}",
                            i, pop[i].Fitness, rank_rs[i], rank_ds[i]);
                    }
                }
            } else {
                fmt::println("n={:5d} m={}: agree in all {} seeds", n, m, nseeds);
            }
        }
    }
}

// Tagged [.] so it only runs when explicitly requested: operon_test "[.][rankdebug]"
// This test reproduces the disagreement between RankIntersectSorter and DeductiveSorter
// and prints every individual whose rank assignment differs, along with their fitness values.
// Run it to find the concrete pair that violates the algorithm's assumptions.
TEST_CASE("RankIntersect vs Deductive disagreement reproducer", "[.][rankdebug]")
{
    Operon::RandomGenerator rd(1234);
    std::uniform_real_distribution<Operon::Scalar> dist(0, 1);

    std::array ns{100, 1000, 5000};
    std::array ms{2, 3, 4, 5, 10};

    RankIntersectSorter rs;
    DeductiveSorter    ds;

    for (auto n : ns) {
        for (auto m : ms) {
            auto pop = InitializePop(rd, dist, n, m);

            auto f_rs = rs.Sort(pop, 0);
            auto f_ds = ds.Sort(pop, 0);

            // Build per-individual rank maps
            Operon::Vector<int> rank_rs(pop.size(), -1);
            Operon::Vector<int> rank_ds(pop.size(), -1);
            for (auto fi = 0; fi < std::ssize(f_rs); ++fi)
                for (auto idx : f_rs[fi]) rank_rs[idx] = fi;
            for (auto fi = 0; fi < std::ssize(f_ds); ++fi)
                for (auto idx : f_ds[fi]) rank_ds[idx] = fi;

            bool any = false;
            for (size_t i = 0; i < pop.size(); ++i) {
                if (rank_rs[i] != rank_ds[i]) {
                    if (!any) {
                        fmt::println("--- Disagreement: n={} m={} ---", n, m);
                        any = true;
                    }
                    fmt::println("  ind {:4d}  fitness={}  RankIntersect={}  Deductive={}",
                        i, pop[i].Fitness, rank_rs[i], rank_ds[i]);
                }
            }
            if (!any) {
                fmt::println("n={} m={}: agree ({} fronts)", n, m, f_rs.size());
            }
        }
    }
}

} // namespace Operon::Test
