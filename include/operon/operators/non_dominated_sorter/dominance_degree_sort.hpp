// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_PARETO_DOMINANCE_DEGREE_SORT
#define OPERON_PARETO_DOMINANCE_DEGREE_SORT

#include "sorter_base.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

struct OPERON_EXPORT DominanceDegreeSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result override;
};

} // namespace operon

#endif
