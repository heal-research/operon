// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DEDUCTIVE_SORT
#define OPERON_PARETO_DEDUCTIVE_SORT

#include "sorter_base.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

class OPERON_EXPORT DeductiveSorter : public NondominatedSorterBase {
    auto
    Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result override; 
};
} // namespace operon

#endif
