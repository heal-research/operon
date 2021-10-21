// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_ROTATING_SORT
#define OPERON_PARETO_ROTATING_SORT

#include <algorithm>
#include <iterator>

#include "core/operator.hpp"
#include "robin_hood.h"
#include "sorter_base.hpp"

namespace Operon {

struct RankSorter : public NondominatedSorterBase {
    using Vec = Eigen::Array<size_t, -1, 1>;
    using Mat = Eigen::Array<size_t, -1, -1>;

    inline NondominatedSorterBase::Result Sort(Operon::Span<Operon::Individual const> pop) const override
    {
        return SortBit(pop);
    }

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    NondominatedSorterBase::Result SortRank(Operon::Span<Operon::Individual const> pop) const;
#endif
    NondominatedSorterBase::Result SortBit(Operon::Span<Operon::Individual const> pop) const;
};

} // namespace Operon

#endif
