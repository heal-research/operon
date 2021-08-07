// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DEDUCTIVE_SORT
#define OPERON_PARETO_DEDUCTIVE_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

namespace Operon {

// McClymont and Keedwell 2012 - "Deductive Sort and Climbing Sort: New Methods for Non-Dominated Sorting"
// https://doi.org/10.1162/EVCO_a_00041
struct DeductiveSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
    {
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
    template <size_t N = 0>
    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        size_t n = 0; // total number of sorted solutions
        std::vector<std::vector<size_t>> fronts;
        std::vector<size_t> dominated(pop.size(), 0);
        std::vector<size_t> sorted(pop.size(), 0);
        auto dominated_or_sorted = [&](size_t i) { return sorted[i] || dominated[i]; };

        //size_t ncomp{0ul};
        //size_t ops{0ul};
        while (n < pop.size()) {
            std::vector<size_t> front;

            for (size_t i = 0; i < pop.size(); ++i) {
                //++this->Stats.InnerOps;
                if (!dominated_or_sorted(i)) {
                    for (size_t j = i + 1; j < pop.size(); ++j) {
                        //++this->Stats.InnerOps;
                        if (!dominated_or_sorted(j)) {
                            //++this->Stats.DominanceComparisons;
                            auto res = pop[i].ParetoCompare<N>(pop[j]);

                            dominated[i] = (res == Dominance::Right);
                            dominated[j] = (res == Dominance::Left || res == Dominance::Equal);

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
        //fmt::print("ncomp: {}\n", ncomp);
        return fronts;
    }
};
} // namespace operon

#endif
