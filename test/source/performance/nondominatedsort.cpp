// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>

#include "nanobench.h"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

constexpr size_t MinEpochIterations { 50 };
constexpr size_t Seed { 1234 };

template <typename Dist>
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
};

TEST_CASE("non-dominated sort performance")
{
    Operon::RandomGenerator rd{0};
    constexpr int reps{16};

    auto run_sorter = [&](nb::Bench& bench, std::string const& name, auto&& sorter, int n, int m)
    {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
        auto pop = InitializePop(rd, dist, n, m);
        bench.run(fmt::format("{};{};{}", name, n, m), [&]() {
            auto fronts = sorter(pop);
            return fronts.size();
        });
    };

    constexpr int N{20000};
    constexpr int M{20};

    std::vector<int> ns { 1000, 2500, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000 }; // NOLINT

    std::vector<int> ms;
    for (auto i = 2; i <= M; ++i) { ms.push_back(i); }

    auto test = [&](auto& bench, auto&& name, auto&& sorter) {
        for (auto n : ns) {
            for (auto m : ms) {
                run_sorter(bench, name, sorter, n, m);
            }
        }
    };

    SUBCASE("RS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "RS", Operon::RankIntersectSorter{});
        bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("RS N=25000 M=10")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        const int n{25000};
        const int m{10};
        run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
        bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("MNDS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "MNDS", Operon::MergeSorter{});
        bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("MNDS N=25000 M=10")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        const int n{25000};
        const int m{10};
        run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
        bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }
}

TEST_CASE("non-dominated sort complexity")
{
    Operon::RandomGenerator rd{0};
    std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
    //using F = std::function<std::vector<std::vector<size_t>>(Operon::Span<Operon::Individual const>)>;
    auto check_complexity = [&](size_t m, auto&& sorter) {
        std::vector<size_t> sizes { 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
        nb::Bench bench;
        bench.minEpochIterations(10);
        //Operon::RandomGenerator rd(1234);

        for (auto s : sizes) {
            auto pop = InitializePop(rd, dist, s, m);
            bench.complexityN(s).run(fmt::format("n = {}", s), [&]() { return sorter(pop).size(); });
        }
        std::cout << bench.complexityBigO() << "\n";
    };

    SUBCASE("M=2")
    {
        check_complexity(2, DeductiveSorter {});
        check_complexity(2, HierarchicalSorter {});
        check_complexity(2, RankIntersectSorter {});
        check_complexity(2, RankOrdinalSorter {});
    }

    SUBCASE("M=3")
    {
        check_complexity(3, DeductiveSorter {});
        check_complexity(3, HierarchicalSorter {});
        check_complexity(3, RankIntersectSorter {});
        check_complexity(3, RankOrdinalSorter {});
    }
}
} // namespace Operon::Test
