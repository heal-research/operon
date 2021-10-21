// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research
//
// This implementation is a port of the original implementation in java by Moreno et al.
// Moreno et al. - "Merge Nondominated Sorting Algorithm for Many-Objective Optimization"
// https://ieeexplore.ieee.org/document/9000950
// https://github.com/jMetal/jMetal/blob/master/jmetal-core/src/main/java/org/uma/jmetal/util/ranking/impl/MergeNonDominatedSortRanking.java
//
// The only changes to the original version are some C++-specific optimizations (e.g. using swap at line 216 and eliminating a temporary)

#ifndef OPERON_PARETO_MERGE_NONDOMINATED_SORT
#define OPERON_PARETO_MERGE_NONDOMINATED_SORT

#include <algorithm>

#include "sorter_base.hpp"

namespace Operon {
class MergeNondominatedSorter : public NondominatedSorterBase {
    mutable int SOL_ID;
    mutable int SORT_INDEX;
    mutable int m; // number of objectives
    mutable int n; // population size
    mutable int initialPopulationSize;
    mutable std::vector<int> ranking;
    mutable std::vector<std::vector<Operon::Scalar>> population;
    mutable std::vector<std::vector<Operon::Scalar>> work;
    mutable std::vector<std::pair<int, int>> duplicatedSolutions;
    //mutable detail::BitsetManager bsm;

    NondominatedSorterBase::Result
    Sort(Operon::Span<Operon::Individual const> pop) const override;

private:
    void Clear() const
    {
        ranking.clear();
        population.clear();
        work.clear();
        duplicatedSolutions.clear();
    }

};
} // namespace Operon

#endif
