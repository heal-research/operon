// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"

#include <ranges>
#include <eve/module/algo.hpp>

namespace Operon {

    template<EfficientSortStrategy SearchStrategy>
    inline auto EfficientSortImpl(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) -> NondominatedSorterBase::Result
    {
        auto const m = static_cast<int>(std::ssize(pop[0].Fitness));

        // check if individual i is dominated by any individual in the front f
        auto dominated = [&](auto const& f, size_t i) {
            return std::ranges::any_of(std::views::reverse(f), [&](size_t j) {
                auto const& a = pop[j].Fitness;
                auto const& b = pop[i].Fitness;
                return m == 2
                    ? std::ranges::all_of(std::ranges::iota_view{0, m}, [&](auto k) { return a[k] <= b[k]; })
                    : eve::algo::all_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x <= y; });
            });
        };

        std::vector<std::vector<size_t>> fronts;
        for (size_t i = 0; i < pop.size(); ++i) {
            decltype(fronts)::iterator it;
            if constexpr (SearchStrategy == EfficientSortStrategy::Binary) { // binary search
                it = std::partition_point(fronts.begin(), fronts.end(), [&](auto const& f) { return dominated(f, i); });
            } else { // sequential search
                it = std::find_if(fronts.begin(), fronts.end(), [&](auto const& f) { return !dominated(f, i); });
            }
            if (it == fronts.end()) { fronts.push_back({i}); }
            else                    { it->push_back(i);          }
        }
        return fronts;
    }

    auto EfficientBinarySorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
    {
        return EfficientSortImpl<EfficientSortStrategy::Binary>(pop, eps);
    }

    auto EfficientSequentialSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
    {
        return EfficientSortImpl<EfficientSortStrategy::Sequential>(pop, eps);
    }
} // namespace Operon