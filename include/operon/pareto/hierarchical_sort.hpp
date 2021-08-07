// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_HIERARCHICAL_SORT
#define OPERON_PARETO_HIERARCHICAL_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

#include "robin_hood.h"

#include <deque>

namespace Operon {

// Bao et al. 2017 - "A novel non-dominated sorting algorithm for evolutionary multi-objective optimization"
// https://doi.org/10.1016/j.jocs.2017.09.015
struct HierarchicalSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const {
        size_t m = pop.front().Fitness.size();
        ENSURE(m > 1);
        switch (m) {
        case 2:
            return Sort<2>(pop);
        case 3:
            return Sort<3>(pop);
        case 4:
            return Sort<4>(pop);
        case 5:
            return Sort<5>(pop);
        case 6:
            return Sort<6>(pop);
        case 7:
            return Sort<7>(pop);
        default:
            return Sort<0>(pop);
        }
    }

    private:
    template <size_t N>
    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        size_t n = pop.size();

        std::deque<size_t> q(n);
        std::iota(q.begin(), q.end(), 0ul);

        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0ul);
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            //++this->Stats.LexicographicalComparisons;
            return pop[a].LexicographicalCompare(pop[b]);
        });
        std::vector<size_t> pos(n);
        for (size_t i = 0; i < n; ++i) { pos[idx[i]] = i; }

        std::vector<size_t> dominated;
        dominated.reserve(n);
        std::vector<std::vector<size_t>> fronts;
        while (!q.empty()) {
            ++this->Stats.InnerOps;
            // initialize new empty front
            std::vector<size_t> front;

            // sort solutions in q according to the first objective value
            // optimizations:
            // - use rank comparison as it is much faster
            // - use timsort as it is faster than std::stable_sort
            std::stable_sort(q.begin(), q.end(), [&](size_t a, size_t b) {
                //++this->Stats.SingleValueComparisons;
                return pos[a] < pos[b];
            });

            while (q.size() > 1) {
                //++this->Stats.InnerOps;
                auto q1 = q.front();
                q.pop_front();
                front.push_back(q1);

                auto nonDominatedCount = 0ul;
                while (q.size() > nonDominatedCount) {
                    //++this->Stats.InnerOps;
                    auto qj = q.front();
                    q.pop_front();
                    //++this->Stats.DominanceComparisons;
                    if (pop[q1].ParetoCompare<N>(pop[qj]) != Dominance::None) { 
                        dominated.push_back(qj);
                    } else {
                        q.push_back(qj);
                        ++nonDominatedCount;
                    }
                }
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
};

namespace hidden {
    // BB: this is the paper version of Hierarchical Sort (HNDS), which runs slowly because of all the std::copy.
    // this goes to prove that authors focusing on Big-O complexity (number of pareto comparisons) which looks good
    // on paper sometimes create really badly performing algorithms due to set intersections and other costly ops
    template <size_t N>
    inline std::vector<std::vector<size_t>> HSortV1(Operon::Span<Operon::Individual> pop)
    {
        size_t n = pop.size();

        std::deque<size_t> q(n);
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
                    auto d = pop[q1].ParetoCompare<N>(pop[qj]);
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

} // namespace detail

} // namespace operon

#endif
