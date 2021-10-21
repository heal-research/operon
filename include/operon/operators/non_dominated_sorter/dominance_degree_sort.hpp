// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DOMINANCE_DEGREE_SORT
#define OPERON_PARETO_DOMINANCE_DEGREE_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"
#include "core/types.hpp"

#include "sorter_base.hpp"

namespace Operon {

class DominanceDegreeSorter : public NondominatedSorterBase {
    private:
    NondominatedSorterBase::Result Sort(Operon::Span<Operon::Individual const> pop) const override;
};

} // namespace operon

#endif
