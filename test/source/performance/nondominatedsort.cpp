// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <taskflow/taskflow.hpp>
#include <thread>

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

template <typename Func, typename Dist>
void Test(std::string const& str, nb::Bench& bench, Func&& func, Operon::RandomGenerator& rd, Dist& dist, std::vector<size_t> const& ns, std::vector<size_t> const& ms)
{
    for (auto m : ms) {
        for (auto n : ns) {
            auto pop = InitializePop(rd, dist, n, m);
            bench.run(fmt::format("{}: n = {}, m = {}", str, n, m), [&] {
                std::stable_sort(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return Operon::Less{}(lhs.Fitness, rhs.Fitness); });
                std::vector<Individual> dup;
                dup.reserve(pop.size());
                auto r = std::unique(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) {
                    auto res = Operon::Equal{}(lhs.Fitness, rhs.Fitness);
                    if (res) {
                        dup.push_back(rhs);
                    }
                    return res;
                });
                std::copy_n(std::make_move_iterator(dup.begin()), dup.size(), r);
                Operon::Span<Individual const> s(pop.begin(), r);
                return func(s).size();
            });
        }
    }
    //bench.render(ankerl::nanobench::templates::csv(), std::cout);
}

TEST_CASE("non-dominated sort" * doctest::test_suite("[performance]"))
{
    auto initializePop = [](Operon::RandomGenerator& random, auto& dist, size_t n, size_t m) {
        std::vector<Individual> individuals(n);
        for (auto& individual : individuals) {
            individual.Fitness.resize(m);

            for (size_t j = 0; j < m; ++j) {
                individual[j] = dist(random);
            }
            ENSURE(individual.Fitness.size() == m);
        }

        return individuals;
    };

    Operon::RandomGenerator rd(1234);
    //std::vector<size_t> ms;
    //for (size_t i = 2; i <= 20; ++i) { ms.push_back(i); }
    //std::vector<size_t> ns { 100, 500 };
    //for (size_t i = 1000; i <= 20000; i += 1000) { ns.push_back(i); }
    std::vector<size_t> ms { 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<size_t> ns { 10000 };
    std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
    std::uniform_int_distribution<size_t> integerDist(0, 10);

    nb::Bench bench;
    bench.relative(true).performanceCounters(true).minEpochIterations(10);
    //bench.output(nullptr);

    // A: dominate on equal
    // B: no dominate on equal
    DeductiveSorter ds;
    HierarchicalSorter hs;
    EfficientBinarySorter ensBs;
    EfficientSequentialSorter ensSs;
    RankOrdinalSorter ro;
    RankIntersectSorter rs;
    MergeSorter ms_;

    SUBCASE("point cloud all")
    {
        bench.minEpochIterations(10);
        Test("RS", bench, [&](auto pop) { return rs(pop); }, rd, dist, ns, ms);
        Test("RO", bench, [&](auto pop) { return ro(pop); }, rd, dist, ns, ms);
        //Test("DS", bench, [&](auto pop){ return ds.Sort(pop); }, rd, dist, ns, ms);
        //Test("HS", bench, [&](auto pop){ return ds.Sort(pop); }, rd, dist, ns, ms);
        //Test("ENS-BS", bench, [&](auto pop){ return ds.Sort(pop); }, rd, dist, ns, ms);
        //Test("ENS-SS", bench, [&](auto pop){ return ds.Sort(pop); }, rd, dist, ns, ms);
        Test("MNDS", bench, [&](auto pop){ return ms_(pop); }, rd, dist, ns, ms);

        std::ofstream of("./synthetic_point_cloud.csv");

        bench.render(ankerl::nanobench::templates::csv(), of);
    }

    SUBCASE("point cloud threaded RS")
    {
        tf::Executor ex;
        tf::Taskflow tf;

        nb::Bench b;
        b.output(nullptr);
        for (size_t i = 0; i < 16; ++i) {
            tf.emplace([&] {
                b = bench;
                b.minEpochIterations(10);
                Test("RS", b, rs, rd, dist, ns, ms);
            });
        }
        ex.run(tf).wait();
        b.render(ankerl::nanobench::templates::csv(), std::cout);
    }

    SUBCASE("statics vs dynamic M")
    {
        auto pop = InitializePop(rd, dist, 5000, 2);
        bench.run("static M", [&]() { return RankIntersectSorter {}(pop).size(); });
        bench.run("dynamic M", [&]() { return RankIntersectSorter {}(pop).size(); });
    }

    SUBCASE("point cloud RS") { Test("RS", bench, rs, rd, dist, ns, ms); }
    SUBCASE("point cloud DS") { Test("DS", bench, ds, rd, dist, ns, ms); }
    SUBCASE("point cloud HS") { Test("HS", bench, hs, rd, dist, ns, ms); }
    SUBCASE("point cloud ENS-BS") { Test("ENS-BS", bench, ensBs, rd, dist, ns, ms); }
    SUBCASE("point cloud ENS-SS") { Test("ENS-SS", bench, ensSs, rd, dist, ns, ms); }
    SUBCASE("point cloud MS") { Test("MNDS", bench, ms_, rd, dist, ns, ms); }

    //using F = std::function<std::vector<std::vector<size_t>>(Operon::Span<Operon::Individual const>)>;
    auto check_complexity = [&](size_t m, auto&& sorter) {
        std::vector<size_t> sizes { 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
        bench.minEpochIterations(10);
        //Operon::RandomGenerator rd(1234);

        for (auto s : sizes) {
            auto pop = initializePop(rd, dist, s, m);
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
        check_complexity(2, RankIntersectSorter {});
        check_complexity(2, RankOrdinalSorter {});
    }
}
} // namespace Operon::Test
