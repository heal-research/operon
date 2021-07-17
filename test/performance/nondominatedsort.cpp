// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>
#include <interpreter/dispatch_table.hpp>
#include <thread>

#include "algorithms/nsga2.hpp"
#include "core/dataset.hpp"
#include "core/pset.hpp"
#include "interpreter/interpreter.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"

#include "nanobench.h"

#include "taskflow/taskflow.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

namespace detail {

    template <size_t N>
    Operon::Vector<int> ComputeRanks(Operon::Span<Individual> pop)
    {
        Operon::Vector<int> ranks(pop.size(), 0);
        for (size_t i = 0; i < pop.size() - 1; ++i) {
            for (size_t j = i + 1; j < pop.size(); ++j) {
                auto d = pop[i].Compare<N>(pop[j]);
                if (d == DominanceResult::Equality) {
                    // duplicate points are banished to the last pareto front
                    ranks[i] = (int)pop.size();
                    continue;
                }
                ranks[i] += d == DominanceResult::RightDominates;
                ranks[j] += d == DominanceResult::LeftDominates;
            }
        }
        return ranks;
    }

    Operon::Vector<Operon::Vector<int>> ComputeFronts(Operon::Vector<int> const& ranks)
    {
        std::vector<int> indices(ranks.size());
        std::iota(indices.begin(), indices.end(), 0);
        pdqsort(indices.begin(), indices.end(), [&](auto i, auto j) { return ranks[i] < ranks[j]; });

        auto r = ranks[indices.front()];
        auto it0 = indices.begin();
        Operon::Vector<Operon::Vector<int>> fronts;
        while (true) {
            auto it = std::partition_point(it0, indices.end(), [&](auto i) { return ranks[i] == r; });
            Operon::Vector<int> front(it0, it);
            fronts.push_back(front);
            if (it == indices.end()) {
                break;
            }
            it0 = it;
            r = ranks[*it];
        }
        return fronts;
    }

    void ComputeCrowdingDistance(Operon::Span<Individual> pop, Operon::Vector<Operon::Vector<int>>& fronts)
    {
        // assign distance. each front is sorted for each objective
        size_t n = pop.front().Fitness.size();
        auto inf = Operon::Numeric::Max<Operon::Scalar>();
        for (size_t i = 0; i < fronts.size(); ++i) {
            auto& front = fronts[i];
            for (size_t obj = 0; obj < n; ++obj) {
                pdqsort(front.begin(), front.end(), [&](auto a, auto b) { return pop[a][obj] < pop[b][obj]; });
                auto min = pop.front()[obj];
                auto max = pop.back()[obj];
                for (size_t j = 0; j < front.size(); ++j) {
                    auto idx = front[j];

                    pop[idx].Rank = i;
                    if (obj == 0)
                        pop[idx].Distance = 0;

                    auto mPrev = j > 0 ? pop[j - 1][obj] : inf;
                    auto mNext = j < front.size() - 1 ? pop[j + 1][obj] : inf;
                    auto distance = (mNext - mPrev) / (max - min);
                    if (!std::isfinite(distance)) {
                        distance = 0;
                    }
                    pop[idx].Distance += distance;
                }
            }
        }
    }
}

TEST_CASE("non-dominated sort")
{
    constexpr size_t n = 100; // number of trees
    constexpr size_t maxLength = 100;
    constexpr size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("../data/Friedman-I.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& v) { return v.Name != target; });
    Range range = { 0, ds.Rows() };

    auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(range).TestRange(range);
    problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };

    std::vector<Tree> trees(n);
    std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

    DispatchTable dt;
    Interpreter interpreter(dt);

    Operon::RSquaredEvaluator r2eval(problem, interpreter);
    Operon::Vector<Operon::Scalar> buf(range.Size());

    std::vector<Individual> individuals(n);
    for (size_t i = 0; i < individuals.size(); ++i) {
        individuals[i].Genotype = trees[i];
        individuals[i].Fitness.resize(2);
        individuals[i].Fitness[0] = r2eval(rd, individuals[i], buf).front();
        individuals[i].Fitness[1] = static_cast<Operon::Scalar>(trees[i].Length());
    }

    nb::Bench b;
    b.minEpochIterations(10);

    Operon::Vector<int> ranks;
    Operon::Vector<Operon::Vector<int>> fronts;

    SUBCASE("Compute ranks") {
        b.run("compute ranks", [&]() {
            ranks = detail::ComputeRanks<2>(individuals);
        });
    }

    SUBCASE("Compute fronts") {
        b.run("compute fronts", [&]() {
            fronts = detail::ComputeFronts(ranks);
        });
    }
}

} // namespace Operon::Test
