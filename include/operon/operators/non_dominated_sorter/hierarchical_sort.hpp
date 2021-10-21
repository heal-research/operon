// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_HIERARCHICAL_SORT
#define OPERON_PARETO_HIERARCHICAL_SORT

#include "sorter_base.hpp"
#include "robin_hood.h"

#include <deque>

namespace Operon {

class HierarchicalSorter : public NondominatedSorterBase {
    NondominatedSorterBase::Result Sort(Operon::Span<Operon::Individual const> pop) const override;
};

namespace hidden {
    inline std::vector<std::vector<size_t>> HSortV1(Operon::Span<Operon::Individual> pop)
    {
        std::deque<size_t> q(pop.size());
        std::iota(q.begin(), q.end(), 0ul);

        Operon::Vector<size_t> vecs[2];
        auto& dominated = vecs[0];
        auto& nondominated = vecs[1];

        std::vector<std::vector<size_t>> fronts;
        while (!q.empty()) {
            // initialize new empty front
            std::vector<size_t> front;

            // sort solutions in q according to the first objective value
            std::stable_sort(q.begin(), q.end(), [&](size_t a, size_t b) { return pop[a].LexicographicalCompare(pop[b]); });

            while (q.size() > 1) {
                auto q1 = q.front();
                q.pop_front();
                front.push_back(q1);

                while (q.size() > 0) {
                    auto qj = q.front();
                    q.pop_front();
                    auto d = pop[q1].ParetoCompare(pop[qj]);
                    vecs[d == Dominance::None].push_back(qj);
                }

                // move solutions from nondominated to q
                std::copy(nondominated.begin(), nondominated.end(), std::back_inserter(q));
                nondominated.clear();
            }
            // assign the last solution of q to F_k
            if (!q.empty()) {
                front.push_back(q.back());
                q.pop_back();
            }
            std::copy(dominated.begin(), dominated.end(), std::back_inserter(q));
            dominated.clear();
            fronts.push_back(front);
        }

        return fronts;
    }
} // namespace hidden 
} // namespace operon

#endif
