// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research
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
#include "operon/operon_export.hpp"

namespace Operon {
class OPERON_EXPORT MergeNondominatedSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result override;
};
} // namespace Operon

#endif
