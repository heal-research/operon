// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon/operators/non_dominated_sorter/deductive_sort.hpp"

namespace Operon {

    NondominatedSorterBase::Result
    DeductiveSorter::Sort(Operon::Span<Operon::Individual const> pop) const
    {
        size_t n = 0; // total number of sorted solutions
        std::vector<std::vector<size_t>> fronts;
        std::vector<bool> dominated(pop.size(), false);
        std::vector<bool> sorted(pop.size(), false);
        auto dominatedOrSorted = [&](size_t i) { return sorted[i] || dominated[i]; };

        while (n < pop.size()) {
            std::vector<size_t> front;

            for (size_t i = 0; i < pop.size(); ++i) {
                ++Stats.InnerOps;
                if (!dominatedOrSorted(i)) {
                    for (size_t j = i + 1; j < pop.size(); ++j) {
                        ++Stats.InnerOps;
                        if (!dominatedOrSorted(j)) {
                            ++Stats.DominanceComparisons;
                            auto const& lhs = pop[i];
                            auto const& rhs = pop[j];
                            auto res = lhs.ParetoCompare(rhs);

                            dominated[i] = (res == Dominance::Right);
                            dominated[j] = (res == Dominance::Left);

                            if (dominated[i]) {
                                break;
                            }
                        }
                    }

                    if (!dominated[i]) {
                        front.push_back(i);
                        sorted[i] = true;
                    }
                }
            }
            std::fill(dominated.begin(), dominated.end(), 0UL);
            n += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }

}
