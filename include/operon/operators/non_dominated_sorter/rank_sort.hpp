// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_PARETO_ROTATING_SORT
#define OPERON_PARETO_ROTATING_SORT

#include <algorithm>
#include <iterator>
#include <robin_hood.h>

#include "sorter_base.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
struct OPERON_EXPORT RankSorter : public NondominatedSorterBase {
    using Vec = Eigen::Array<Eigen::Index, -1, 1>;
    using Mat = Eigen::Array<Eigen::Index, -1, -1>;

    inline auto Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result override
    {
        return RankIntersect(pop);
    }

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    static auto RankOrdinal(Operon::Span<Operon::Individual const> pop) -> NondominatedSorterBase::Result;
#endif
    static auto RankIntersect(Operon::Span<Operon::Individual const> pop) -> NondominatedSorterBase::Result;
};

} // namespace Operon

#endif
