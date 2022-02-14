// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_OPERATORS_NONDOMINATED_SORTER_HPP
#define OPERON_OPERATORS_NONDOMINATED_SORTER_HPP

#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
struct Individual;

enum EfficientSortStrategy : int { Binary, Sequential };

class NondominatedSorterBase {
public:
    using Result = std::vector<std::vector<size_t>>;

    mutable struct {
        size_t LexicographicalComparisons = 0; // both lexicographical and single-objective
        size_t SingleValueComparisons = 0;
        size_t DominanceComparisons = 0;
        size_t RankComparisons = 0;
        size_t InnerOps = 0;
        double MeanRank = 0;
        double MeanND = 0;
    } Stats;

    void Reset() { Stats = { 0, 0, 0, 0, 0, 0., 0. }; }

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

template<EfficientSortStrategy>
struct OPERON_EXPORT EfficientSorter : public NondominatedSorterBase {
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

using EfficientBinarySorter = EfficientSorter<EfficientSortStrategy::Binary>;
using EfficientSequentialSorter = EfficientSorter<EfficientSortStrategy::Sequential>;

} // namespace Operon
#endif
