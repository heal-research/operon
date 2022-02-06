// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_PARETO_HIERARCHICAL_SORT
#define OPERON_PARETO_HIERARCHICAL_SORT

#include "sorter_base.hpp"
#include <deque>

namespace Operon {

struct OPERON_EXPORT HierarchicalSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result override;
};
} // namespace Operon

#endif
