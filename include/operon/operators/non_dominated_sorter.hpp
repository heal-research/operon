// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPERATORS_NONDOMINATED_SORTER_HPP
#define OPERON_OPERATORS_NONDOMINATED_SORTER_HPP

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Contract relied on by NSGA2::Sort (nsga2.cpp): every implementation reads
// only Individual::Fitness, never Genotype/Rank/Distance. NSGA2 exploits
// this to sort a permutation of indices instead of the population's actual
// storage, feeding this interface lightweight fitness-only stand-ins for
// the unique individuals rather than deep-copying genotype trees. A future
// sorter that reads anything beyond Fitness would silently break that.
class NondominatedSorterBase {
public:
    using Result = Operon::Vector<Operon::Vector<size_t>>;

    // this method needs to be implemented by all deriving classes
    virtual auto Sort(Operon::Span<Operon::Individual const>, Operon::Scalar) const -> Result = 0;

    auto operator()(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps = 0) const -> Result
    {
        return Sort(pop, eps);
    }
};

struct OPERON_EXPORT DeductiveSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT DominanceDegreeSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT HierarchicalSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT EfficientBinarySorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT EfficientSequentialSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT MergeSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT RankOrdinalSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT RankIntersectSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

struct OPERON_EXPORT BestOrderSorter : public NondominatedSorterBase {
    auto Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result override;
};

} // namespace Operon
#endif
