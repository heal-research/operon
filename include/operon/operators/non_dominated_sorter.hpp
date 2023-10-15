// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPERATORS_NONDOMINATED_SORTER_HPP
#define OPERON_OPERATORS_NONDOMINATED_SORTER_HPP

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

// aggregate performancae statistics such as sort duration
#include <chrono>

namespace Operon {

enum EfficientSortStrategy : int { Binary, Sequential };

class NondominatedSorterBase {
public:
    using Result = std::vector<std::vector<size_t>>;

    mutable struct {
        size_t LexicographicalComparisons{0}; // both lexicographical and single-objective
        size_t SingleValueComparisons{0};
        size_t DominanceComparisons{0};
        size_t RankComparisons{0};
        size_t InnerOps{0};
        double MeanRank{0};
        double MeanND{0};
        std::chrono::duration<size_t> Duration{0};
    } Stats;

    void Reset() { Stats = { 0, 0, 0, 0, 0, 0., 0. }; }

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
