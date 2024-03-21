// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <chrono>
#include <doctest/doctest.h>
#include <fstream>
#include <random>

#include "nanobench.h"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

#include <filesystem>
#include <scn/scan.h>
#include <ranges>
#include <string_view>

namespace fs = std::filesystem;
namespace nb = ankerl::nanobench;

namespace Operon::Test {

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

    auto run_sorter = [&](nb::Bench& bench, std::string const& name, auto&& sorter, int n, int m)
    {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, 1.F);
        auto pop = InitializePop(rd, dist, n, m);
        bench.run(fmt::format("{};{};{}", name, n, m), [&]() {
            auto fronts = sorter(pop);
            return fronts.size();
        });
    };

    constexpr int M{40};

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

    SUBCASE("2 objectives")
    {
        nb::Bench bench;
        std::vector<int> ns;
        std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i){ return 1000 * i; });
        const int m { 2 };
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "RO", Operon::RankOrdinalSorter{}, n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{}, n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{}, n, m);
            run_sorter(bench, "DS", Operon::DeductiveSorter{}, n, m);
            run_sorter(bench, "HS", Operon::HierarchicalSorter{}, n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("3 objectives")
    {
        nb::Bench bench;
        std::vector<int> ns;
        std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i){ return 1000 * i; });
        const int m { 3 };
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "RO", Operon::RankOrdinalSorter{}, n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{}, n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{}, n, m);
            run_sorter(bench, "DS", Operon::DeductiveSorter{}, n, m);
            run_sorter(bench, "HS", Operon::HierarchicalSorter{}, n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("4 objectives")
    {
        nb::Bench bench;
        std::vector<int> ns;
        std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i){ return 1000 * i; });
        const int m { 4 };
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "RO", Operon::RankOrdinalSorter{}, n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{}, n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{}, n, m);
            run_sorter(bench, "DS", Operon::DeductiveSorter{}, n, m);
            run_sorter(bench, "HS", Operon::HierarchicalSorter{}, n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("5 objectives")
    {
        nb::Bench bench;
        std::vector<int> ns;
        std::ranges::transform(std::views::iota(1, 21), std::back_inserter(ns), [](auto i){ return 1000 * i; });
        const int m { 5 };
        for (auto n : ns) {
            run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
            run_sorter(bench, "RO", Operon::RankOrdinalSorter{}, n, m);
            run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
            run_sorter(bench, "MS", Operon::MergeSorter{}, n, m);
            run_sorter(bench, "ENS-SS", Operon::EfficientSequentialSorter{}, n, m);
            run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{}, n, m);
            run_sorter(bench, "DS", Operon::DeductiveSorter{}, n, m);
            run_sorter(bench, "HS", Operon::HierarchicalSorter{}, n, m);
        }
        bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("RS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "RS", Operon::RankIntersectSorter{});

        std::ofstream fs("./rs.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("RO")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "RO", Operon::RankOrdinalSorter{});
        std::ofstream fs("./ro.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("MNDS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "MNDS", Operon::MergeSorter{});
        std::ofstream fs("./mnds.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("BOS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "BOS", Operon::BestOrderSorter{});
        std::ofstream fs("./bos.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("HNDS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "HNDS", Operon::HierarchicalSorter{});
        std::ofstream fs("./hnds.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("DS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "BOS", Operon::DeductiveSorter{});
        std::ofstream fs("./ds.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("ENS-SS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "ENS-SS", Operon::EfficientSequentialSorter{});
        std::ofstream fs("./ens-ss.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("ENS-BS")
    {
        nb::Bench bench;
        bench.performanceCounters(true);
        test(bench, "ENS-BS", Operon::EfficientBinarySorter{});
        std::ofstream fs("./ens-bs.csv");
        bench.render(ankerl::nanobench::templates::csv(), fs);
    }

    SUBCASE("RS N=25000 M=2")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{2};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("RS N=25000 M=3")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{3};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("RS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("RS N=50000 M=20")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{20};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("RS N=10000 M=40")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{10000};
       const int m{40};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("RS N=50000 M=40")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{40};
       run_sorter(bench, "RS", Operon::RankIntersectSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("RO N=25000 M=10")
    {
        nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "RO", Operon::RankOrdinalSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("MNDS N=25000 M=3")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{3};
       run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
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

    SUBCASE("MNDS N=50000 M=20")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{20};
       run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("MNDS N=10000 M=40")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{10000};
       const int m{40};
       run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("MNDS N=50000 M=40")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{40};
       run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("BOS N=25000 M=3")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "MNDS", Operon::MergeSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("BOS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("BOS N=50000 M=2")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{2};
       run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("BOS N=50000 M=20")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{50000};
       const int m{20};
       run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("BOS N=10000 M=40")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{10000};
       const int m{40};
       run_sorter(bench, "BOS", Operon::BestOrderSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("DS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "DS", Operon::DeductiveSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("HNDS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "HNDS", Operon::HierarchicalSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("ENS-BS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "ENS-BS", Operon::EfficientBinarySorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("ENS-SS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "ENS-SS", Operon::EfficientBinarySorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }

    SUBCASE("DDS N=25000 M=10")
    {
       nb::Bench bench;
       bench.performanceCounters(true);
       const int n{25000};
       const int m{10};
       run_sorter(bench, "DDS", Operon::DominanceDegreeSorter{}, n, m);
       bench.render(ankerl::nanobench::templates::csv(), std::cout);;
    }
}

TEST_CASE("single front rs")
{
    auto const n = 50'000;
    std::vector<Operon::Individual> pop(n);

    for (auto i = 0; i < n; ++i) {
        pop[i].Fitness = {static_cast<float>(i), n-static_cast<float>(i)-1};
    }
    nb::Bench bench;
    bench.run("RS", [&]() {
        RankIntersectSorter{}(pop, 0);
    });
}

TEST_CASE("single front mnds")
{
    auto const n = 50'000;
    std::vector<Operon::Individual> pop(n);

    for (auto i = 0; i < n; ++i) {
        auto f = static_cast<Operon::Scalar>(i);
        pop[i].Fitness = {f, n-f};
    }
    nb::Bench bench;
    bench.run("MNDS", [&]() {
        MergeSorter{}(pop, 0);
    });
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

TEST_CASE("dtlz2")
{
    std::string path = "./csv";

    using Key = std::string;
    using Value = std::vector<std::vector<Operon::Individual>>;

    Operon::Map<Key, Value> values;

    nb::Bench bench;

    for (const auto& entry : fs::directory_iterator(path)) {
        std::string name = entry.path().string();
        if (name.find("rs") == std::string::npos) { continue; }
        std::cout << name << std::endl;
        auto result = scn::scan<std::size_t, std::size_t>(name, "./csv/nsga2_DTLZ2_{}_{}_rs.csv");
        ENSURE(result);
        auto [n, m] = result->values();

        // read part
        Value gen;
        std::ifstream fs(name);
        std::string line;
        while (std::getline(fs, line)) {
            std::vector<Operon::Individual> pop;
            auto c = 0UL;
            Operon::Individual ind(m);
            for (auto const sv : std::views::split(line, ',')) {
                if (c == m-1) {
                    c = 0;
                    pop.push_back(std::move(ind));
                    ind = Operon::Individual(m);
                } else {
                    std::string s(sv.begin(), sv.end());
                    auto v = scn::scan<Operon::Scalar>(s, "{}")->value();
                    ind[c++] = v;
                }
            }
            ENSURE(n == pop.size());
            gen.push_back(pop);
        }
        //values[name] = gen;

        // benchmark part
        for (auto i = 0; i < std::ssize(gen); ++i) {
            auto n = gen[i].size();
            auto m = gen[i].front().Size();

            auto individuals = gen[i];

            std::stable_sort(individuals.begin(), individuals.end(), [](auto const& a, auto const& b){ return std::ranges::lexicographical_compare(a.Fitness, b.Fitness); });

            for(auto i = individuals.begin(); i < individuals.end(); ) {
                i->Rank = 0;
                auto j = i + 1;
                for (; j < individuals.end() && i->Fitness == j->Fitness; ++j) {
                    j->Rank = 1;
                }
                i = j;
            }
            auto r = std::stable_partition(individuals.begin(), individuals.end(), [](auto const& ind) { return !ind.Rank; });
            std::vector<Individual> pop(individuals.begin(), r);

            auto fronts = Operon::RankIntersectSorter{}(pop, 0);
            fmt::print("dtlz2 n = {}, m = {}, gen = {}, fronts = {}\n", n, m, i, fronts.size());

            bench.run(fmt::format("RS;{};{};{};DTLZ2", n, m, i), [&]() {
                Operon::RankIntersectSorter sorter;
                sorter.Sort(pop, 0);
            });

            bench.run(fmt::format("MS;{};{};{};DTLZ2", n, m, i), [&]() {
                Operon::MergeSorter sorter;
                sorter.Sort(pop, 0);
            });
        }
    }


    std::ofstream out("./dtlz2-benchmark.csv");
    bench.render(nb::templates::csv(), out);
}
} // namespace Operon::Test
