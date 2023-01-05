// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <deque>
#include <numeric>
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"

namespace Operon {
    auto
    HierarchicalSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
    {
        std::deque<size_t> q(pop.size());
        std::iota(q.begin(), q.end(), 0UL);
        std::vector<size_t> dominated;
        dominated.reserve(pop.size());

        std::vector<std::vector<size_t>> fronts;
        while (!q.empty()) {
            ++Stats.InnerOps;
            std::vector<size_t> front;

            while (!q.empty()) {
                auto q1 = q.front(); q.pop_front();
                front.push_back(q1);
                auto nonDominatedCount = 0UL;
                while (q.size() > nonDominatedCount) {
                    auto qj = q.front(); q.pop_front();
                    if (ParetoDominance{}(pop[q1].Fitness, pop[qj].Fitness, eps) == Dominance::None) {
                        q.push_back(qj);
                        ++nonDominatedCount;
                    } else {
                        dominated.push_back(qj);
                    }
                }
            }
            std::copy(dominated.begin(), dominated.end(), std::back_inserter(q));
            dominated.clear();
            fronts.push_back(front);

            std::stable_sort(q.begin(), q.end(), [&](auto  a, auto b) { return LexicographicalComparison{}(pop[a], pop[b]); });
        }
        return fronts;
    }

} // namespace Operon

