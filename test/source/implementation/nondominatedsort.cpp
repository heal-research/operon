// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <doctest/doctest.h>
#include <random>
#include <thread>

#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include "nanobench.h"

#include "taskflow/taskflow.hpp"

namespace Operon::Test {

TEST_CASE("non-dominated sort" * doctest::test_suite("[implementation]"))
{
    Operon::RandomGenerator rd(1234);

    auto initializePop = [](Operon::RandomGenerator& random, auto& dist, size_t n, size_t m) {
        std::vector<Individual> individuals(n);

        //std::uniform_real_distribution<Operon::Scalar> dist(0, 1);

        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Fitness.resize(m);

            for (size_t j = 0; j < m; ++j) {
                individuals[i][j] = dist(random);
            }
        }

        return individuals;
    };

    auto print = [&](auto& fr) {
        for (size_t i = 0; i < fr.size(); ++i) {
            fmt::print("{}: ", i);
            std::sort(fr[i].begin(), fr[i].end());
            //for (auto f : fronts[i]) { std::cout << f << ":(" << eigenMap(pop[f].Fitness).transpose() << ") "; }
            for (auto f : fr[i]) { std::cout << f << " "; }
            fmt::print("\n");
        }
        fmt::print("\n");
    };
    
    SUBCASE("dominance degree sort")
    {
        //Eigen::Array<Operon::Scalar, -1, 1> w(6);
        //w << 0.9218f, 0.7382f, 0.1763f, 0.4057f, 0.9355f, 0.9218f;
        //auto c = detail::ComputeComparisonMatrix(w);
        //std::cout << "C:\n" << c << "\n";

        //decltype(w) w1(6), w2(6), w3(6);
        //w1 << 0.9501f, 0.2311f, 0.6068f, 0.2311f, 0.8913f, 0.9501f;
        //w2 << 0.4565f, 0.0185f, 0.8214f, 0.0185f, 0.6154f, 0.4565f;
        //w3 << 0.9218f, 0.7382f, 0.1763f, 0.4057f, 0.9355f, 0.9218f;
        //
        //decltype(c) d(6, 6);
        //auto c1 = detail::ComputeComparisonMatrix(w1); d += c1;
        //std::cout << "C1:\n" << c << "\n";
        //auto c2 = detail::ComputeComparisonMatrix(w2); d += c2;
        //std::cout << "C2:\n" << c << "\n";
        //auto c3 = detail::ComputeComparisonMatrix(w3); d += c3;
        //std::cout << "C3:\n" << c << "\n";

        //std::cout << "D:\n" << d << "\n";
    }

    SUBCASE("test 1")
    {
        std::vector<std::vector<Operon::Scalar>> points = {{0, 7}, {1, 5}, {2, 3}, {4, 2}, {7, 1}, {10, 0}, {2, 6}, {4, 4}, {10, 2}, {6, 6}, {9, 5}};
        std::vector<Individual> pop(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            pop[i].Fitness = points[i];
        }
        fmt::print("DS\n");
        DeductiveSorter ds;
        auto fronts = ds(pop);
        print(fronts);

        fmt::print("HS\n");
        HierarchicalSorter hs;
        fronts = hs(pop);
        print(fronts);

        fmt::print("ENS-SS\n");
        EfficientSorter es;
        fronts = es(pop);
        print(fronts);

        fmt::print("ENS-BS\n");
        EfficientSorter<1> eb;
        fronts = eb(pop);
        print(fronts);

        fmt::print("RS\n");
        RankSorter rs;
        fronts = rs(pop);
        print(fronts);

        fmt::print("MNDS\n");
        MergeNondominatedSorter ms;
        fronts = ms(pop);
        print(fronts);
    }

    SUBCASE("test 2")
    {
        std::vector<std::vector<Operon::Scalar>> points = {{1, 2, 3}, {-2, 3, 7}, {-1, -2, -3}, {0, 0, 0}};
        std::vector<Individual> pop(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            pop[i].Fitness = points[i];
        }
        std::stable_sort(pop.begin(), pop.end(), LexicographicalComparison{});
        fmt::print("DS\n");
        DeductiveSorter ds;
        auto fronts = ds(pop);
        print(fronts);

        fmt::print("HS\n");
        HierarchicalSorter hs;
        fronts = hs(pop);
        print(fronts);

        fmt::print("ENS-SS\n");
        EfficientSorter es;
        fronts = es(pop);
        print(fronts);

        fmt::print("ENS-BS\n");
        EfficientSorter<1> eb;
        fronts = eb(pop);
        print(fronts);

        fmt::print("RS\n");
        RankSorter rs;
        fronts = rs(pop);
        print(fronts);

        fmt::print("MNDS\n");
        MergeNondominatedSorter ms;
        fronts = ms(pop);
        print(fronts);
    }

    SUBCASE("six solutions")
    {
        Operon::Vector<Individual> pop(6);
        pop[0].Fitness = { 4.1f, 4.1f };
        pop[1].Fitness = { 3.1f, 5.1f };
        pop[2].Fitness = { 2.4f, 6.0f };
        pop[3].Fitness = { 5.3f, 1.3f };
        pop[4].Fitness = { 4.2f, 2.0f };
        pop[5].Fitness = { 1.8f, 3.4f };

        DeductiveSorter ds;
        auto fronts = ds(pop);
        fmt::print("DS: {}\n", ds.Stats.DominanceComparisons);
        print(fronts);

        RankSorter rs;
        fronts = rs(pop);
        fmt::print("RS: {}\n", rs.Stats.RankComparisons);
        print(fronts);

        HierarchicalSorter hs;
        fronts = hs(pop);
        fmt::print("HS: {}\n", hs.Stats.DominanceComparisons);
        print(fronts);

        EfficientSorter es;
        fronts = es(pop);
        fmt::print("ENS-SS: {}\n", es.Stats.DominanceComparisons);
        print(fronts);

        EfficientSorter<1> eb;
        fronts = eb(pop);
        fmt::print("ENS-BS: {}\n", eb.Stats.DominanceComparisons);
        print(fronts);

        MergeNondominatedSorter ms;
        fronts = ms(pop);
        fmt::print("ENS-BS: {}\n", ms.Stats.DominanceComparisons);
        print(fronts);
    }

    SUBCASE("rank sort") {
        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        auto pop = initializePop(rd, dist, 100, 2);
        RankSorter rs;
        std::vector<std::vector<size_t>> fronts;

        fronts = rs(pop);
        fmt::print("RS comparisons: {} {} {} {}\n", rs.Stats.LexicographicalComparisons, rs.Stats.SingleValueComparisons, rs.Stats.RankComparisons, rs.Stats.InnerOps);
        print(fronts);
        rs.Reset();
    }

    //auto eigenMap = [](Operon::Vector<Operon::Scalar> const& fit) {
    //    return Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> const> (fit.data(), fit.size());
    //};
    SUBCASE("basic")
    {
        size_t n = 200;
        size_t m = 4;

        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        auto pop = initializePop(rd, dist, n, m);

        std::stable_sort(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs.LexicographicalCompare(rhs); });
        std::vector<Individual> dup; dup.reserve(pop.size());
        auto r = std::unique(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) {
            auto res = lhs == rhs;
            if (res) { dup.push_back(rhs); }
            return res;
        });
        ENSURE(std::distance(pop.begin(), r) + dup.size() == pop.size());
        std::copy_n(std::make_move_iterator(dup.begin()), dup.size(), r);
        Operon::Span<Individual const> s(pop.begin(), r);

        auto fronts = DeductiveSorter{}(s);
        std::cout << "deductive sort\n"; 
        print(fronts);

        fronts = RankSorter{}(s);
        std::cout << "rank sort\n"; 
        print(fronts);

        //fronts = FastNondominatedSorter{}(pop);
        //std::cout << "nds sort\n"; 
        //print(fronts);

        fronts = HierarchicalSorter{}(s);
        std::cout << "hierarchical sort\n";
        print(fronts);

        fronts = DominanceDegreeSorter{}(s);
        std::cout << "dominance degree sort\n";
        print(fronts);

        fronts = EfficientSorter{}(s);
        std::cout << "ens-ss\n";
        print(fronts);

        fronts = EfficientSorter<1>{}(s);
        std::cout << "ens-bs\n";
        print(fronts);

        fronts = MergeNondominatedSorter{}(s);
        std::cout << "mnds\n";
        print(fronts);

    }

    SUBCASE("bit density")
    {
        size_t n = 20000;
        size_t m = 20;

        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);
        auto pop = initializePop(rd, dist, n, m);
        auto fronts = RankSorter{}(pop);
        fmt::print("{} fronts\n", fronts.size());
    }

    auto testComparisons = [&](std::string const& name, NondominatedSorterBase& sorter)
    {
        size_t reps = 50;
        std::uniform_real_distribution<Operon::Scalar> dist(0, 1);

        fmt::print("name,n,m,lc,sv,dc,rc,ops,mean_rk,mean_nd\n");
        for (size_t n = 100; n <= 5000; n += 100) {
            for (size_t m = 2; m <= 10; ++m) {
                double lc, dc, rc, sv, ops, mean_rank, mean_nd; 
                lc = dc = rc = ops = sv = mean_rank = mean_nd = 0;
                for (size_t r = 0; r < reps; ++r) {
                    auto pop = initializePop(rd, dist, n, m);
                    sorter(pop);
                    auto [lc_, sv_, dc_, rc_, ops_, rk_, nd_] = sorter.Stats;
                    lc += (double)lc_;
                    sv += (double)sv_;
                    dc += (double)dc_;
                    rc += (double)rc_;
                    ops += (double)ops_;
                    mean_rank += rk_;
                    mean_nd += nd_;
                    sorter.Reset();
                }
                auto r = (double)reps;
                fmt::print("{},{},{},{},{},{},{},{},{},{}\n", name, n, m, lc/r, sv/r, dc/r, rc/r, ops/r, mean_rank/r, mean_nd/r);
            }
        }
    };

    SUBCASE("comparisons RS")
    {
        RankSorter sorter;
        testComparisons("RS", sorter);
    }

    SUBCASE("comparisons DS")
    {
        DeductiveSorter sorter;
        testComparisons("DS", sorter);
    }

    SUBCASE("comparisons HS")
    {
        HierarchicalSorter sorter;
        testComparisons("HS", sorter);
    }

    SUBCASE("comparisons ENS-SS")
    {
        EfficientSorter<false> sorter;
        testComparisons("ENS-SS", sorter);
    }

    SUBCASE("comparisons ENS-BS")
    {
        EfficientSorter<true> sorter;
        testComparisons("ENS-BS", sorter);
    }
}

} // namespace Operon::Test
