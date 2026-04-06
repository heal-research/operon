// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <random>
#include <ranges>

#include <fmt/core.h>

#include "nanobench.h"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

namespace {

template<typename Dist>
auto InitializePop(Operon::RandomGenerator& random, Dist& dist, size_t n, size_t m) -> std::vector<Individual>
{
    std::vector<Individual> individuals(n);
    for (auto& individual : individuals) {
        individual.Fitness.resize(m);
        for (size_t j = 0; j < m; ++j) {
            individual[j] = static_cast<Operon::Scalar>(dist(random));
        }
        ENSURE(individual.Fitness.size() == m);
    }
    return individuals;
}

} // namespace

// Primary benchmark: RS and MS only (fastest across all objective counts).
// For 2 objectives, ENS-SS and ENS-BS are also included (they are competitive
// at low m but scale poorly as m grows).
// Population sizes: 1k–20k in steps of 1k.
// Run with: operon_test "[performance][ndsort]"
TEST_CASE("Non-dominated sort performance", "[performance][ndsort]")
{
    Operon::RandomGenerator rd{0};

    auto run_sorter = [&](nb::Bench& bench, std::string const& name, auto&& sorter, int n, int m) {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
        auto pop = InitializePop(rd, dist, n, m);
        bench.run(fmt::format("{}/{}", name, n), [&]() {
            auto fronts = sorter(pop);
            return fronts.size();
        });
    };

    std::vector<int> ns;
    std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i) { return 1000 * i; });

    SECTION("2 objectives") {
        nb::Bench bench;
        bench.title("2 objectives").minEpochIterations(20);
        const int m{2};
        for (auto n : ns) {
            run_sorter(bench, "RS",     Operon::RankIntersectSorter{},    n, m);
            run_sorter(bench, "MS",     Operon::MergeSorter{},            n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{},  n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("3 objectives") {
        nb::Bench bench;
        bench.title("3 objectives").minEpochIterations(20);
        const int m{3};
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{},         n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("4 objectives") {
        nb::Bench bench;
        bench.title("4 objectives").minEpochIterations(20);
        const int m{4};
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{},         n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("5 objectives") {
        nb::Bench bench;
        bench.title("5 objectives").minEpochIterations(20);
        const int m{5};
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{},         n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }
}

// Optional extended benchmark including slower sorters (RO, BOS).
// Disabled by default — only runs when explicitly requested.
// Run with: operon_test "[.][ndsort-extended]"
TEST_CASE("Non-dominated sort performance (extended)", "[.][ndsort-extended]")
{
    Operon::RandomGenerator rd{0};

    auto run_sorter = [&](nb::Bench& bench, std::string const& name, auto&& sorter, int n, int m) {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
        auto pop = InitializePop(rd, dist, n, m);
        bench.run(fmt::format("{}/{}", name, n), [&]() {
            auto fronts = sorter(pop);
            return fronts.size();
        });
    };

    std::vector<int> ns;
    std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i) { return 1000 * i; });

    SECTION("2 objectives") {
        nb::Bench bench;
        const int m{2};
        for (auto n : ns) {
            run_sorter(bench, "RS",     Operon::RankIntersectSorter{},       n, m);
            run_sorter(bench, "MS",     Operon::MergeSorter{},               n, m);
            run_sorter(bench, "RO",     Operon::RankOrdinalSorter{},         n, m);
            run_sorter(bench, "BOS",    Operon::BestOrderSorter{},           n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{},     n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("3 objectives") {
        nb::Bench bench;
        const int m{3};
        for (auto n : ns) {
            run_sorter(bench, "RS",  Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS",  Operon::MergeSorter{},         n, m);
            run_sorter(bench, "RO",  Operon::RankOrdinalSorter{},   n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{},     n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("4 objectives") {
        nb::Bench bench;
        const int m{4};
        for (auto n : ns) {
            run_sorter(bench, "RS",  Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS",  Operon::MergeSorter{},         n, m);
            run_sorter(bench, "RO",  Operon::RankOrdinalSorter{},   n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{},     n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SECTION("5 objectives") {
        nb::Bench bench;
        const int m{5};
        for (auto n : ns) {
            run_sorter(bench, "RS",  Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "MS",  Operon::MergeSorter{},         n, m);
            run_sorter(bench, "RO",  Operon::RankOrdinalSorter{},   n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{},     n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }
}

TEST_CASE("Single front benchmarks", "[performance]")
{
    auto const n = 50'000;
    std::vector<Operon::Individual> pop(n);

    for (auto i = 0; i < n; ++i) {
        pop[i].Fitness = {static_cast<float>(i), n - static_cast<float>(i) - 1};
    }

    nb::Bench bench;

    SECTION("RS") {
        bench.run("RS single front", [&]() {
            RankIntersectSorter{}(pop, 0);
        });
    }

    SECTION("MS") {
        bench.run("MS single front", [&]() {
            MergeSorter{}(pop, 0);
        });
    }
}

TEST_CASE("Non-dominated sort complexity", "[performance]")
{
    Operon::RandomGenerator rd{0};
    std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);

    auto check_complexity = [&](size_t m, auto&& sorter) {
        std::vector<size_t> sizes{500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
        nb::Bench bench;
        bench.minEpochIterations(10);

        for (auto s : sizes) {
            auto pop = InitializePop(rd, dist, s, m);
            bench.complexityN(s).run(fmt::format("n = {}", s), [&]() { return sorter(pop).size(); });
        }
        std::cout << bench.complexityBigO() << "\n";
    };

    SECTION("M=2") {
        check_complexity(2, RankIntersectSorter{});
        check_complexity(2, MergeSorter{});
    }

    SECTION("M=3") {
        check_complexity(3, RankIntersectSorter{});
        check_complexity(3, MergeSorter{});
    }
}

} // namespace Operon::Test
