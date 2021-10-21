// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DEDUCTIVE_SORT
#define OPERON_PARETO_DEDUCTIVE_SORT

#include "sorter_base.hpp"

namespace Operon {

class DeductiveSorter : public NondominatedSorterBase {
    private:

    NondominatedSorterBase::Result
    Sort(Operon::Span<Operon::Individual const> pop) const override; 
};
} // namespace operon

#endif
