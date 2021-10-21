// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/non_dominated_sorter/deductive_sort.hpp"

namespace Operon {

    NondominatedSorterBase::Result
    DeductiveSorter::Sort(Operon::Span<Operon::Individual const> pop) const
    {
        size_t n = 0; // total number of sorted solutions
        std::vector<std::vector<size_t>> fronts;
        std::vector<size_t> dominated(pop.size(), 0);
        std::vector<size_t> sorted(pop.size(), 0);
        auto dominated_or_sorted = [&](size_t i) { return sorted[i] || dominated[i]; };

        while (n < pop.size()) {
            std::vector<size_t> front;

            for (size_t i = 0; i < pop.size(); ++i) {
                ++Stats.InnerOps;
                if (!dominated_or_sorted(i)) {
                    for (size_t j = i + 1; j < pop.size(); ++j) {
                        ++Stats.InnerOps;
                        if (!dominated_or_sorted(j)) {
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
                        sorted[i] = 1;
                    }
                }
            }
            std::fill(dominated.begin(), dominated.end(), 0ul);
            n += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }

}
