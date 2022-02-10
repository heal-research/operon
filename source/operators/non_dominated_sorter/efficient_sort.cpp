// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"

namespace Operon {

    template<EfficientSortStrategy S>
    static auto EfficientSortImpl(Operon::Span<Operon::Individual const> pop) -> NondominatedSorterBase::Result
    {
        // check if individual i is dominated by any individual in the front f
        auto dominated = [&](auto const& f, size_t i) {
            return std::any_of(f.rbegin(), f.rend(), [&](size_t j) {
                    return pop[j].ParetoCompare(pop[i]) == Dominance::Left;
                    });
        };

        std::vector<std::vector<size_t>> fronts;
        for (size_t i = 0; i < pop.size(); ++i) {
            decltype(fronts)::iterator it;
            if constexpr (S == EfficientSortStrategy::Binary) { // binary search
                it = std::partition_point(fronts.begin(), fronts.end(),
                        [&](auto const& f) { return dominated(f, i); });
            } else { // sequential search
                it = std::find_if(fronts.begin(), fronts.end(),
                        [&](auto const& f) { return !dominated(f, i); });
            }
            if (it == fronts.end()) {
                fronts.push_back({i});
            } else {
                it->push_back(i);
            }
        }
        return fronts;
    }

    template<>
    auto EfficientBinarySorter::Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result
    {
        return EfficientSortImpl<EfficientSortStrategy::Binary>(pop);
    }

    template<>
    auto EfficientSequentialSorter::Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result
    {
        return EfficientSortImpl<EfficientSortStrategy::Sequential>(pop);
    }
} // namespace Operon
