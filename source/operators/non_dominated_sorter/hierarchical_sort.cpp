// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cpp-sort/sorters/merge_sorter.h>
#include <deque>
#include <numeric>
#include <ranges>
#include <eve/module/algo.hpp>

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"

namespace Operon {
    auto
    HierarchicalSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
    {
        auto const m = static_cast<int>(std::ssize(pop[0].Fitness));

        std::deque<size_t> q(pop.size());
        std::iota(q.begin(), q.end(), 0UL);
        std::vector<size_t> dominated;
        dominated.reserve(pop.size());

        std::vector<std::vector<size_t>> fronts;

        auto dominates = [&](auto const& a, auto const& b) {
            return m == 2
                ? std::ranges::all_of(std::ranges::iota_view{0, m}, [&](auto k) { return a[k] <= b[k]; })
                : eve::algo::all_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x <= y; });
        };

        cppsort::merge_sorter sorter;

        while (!q.empty()) {
            std::vector<size_t> front;

            while (!q.empty()) {
                auto q1 = q.front(); q.pop_front();
                front.push_back(q1);
                auto nonDominatedCount = 0UL;
                auto const& f1 = pop[q1].Fitness;
                while (q.size() > nonDominatedCount) {
                    auto qj = q.front(); q.pop_front();
                    auto const& f2 = pop[qj].Fitness;
                    ENSURE(q1 < qj);
                    if (!dominates(f1, f2)) {
                        q.push_back(qj);
                        ++nonDominatedCount; // &&&
                    } else {
                        dominated.push_back(qj);
                    }
                }
            }
            sorter(dominated); // 00
            std::copy(dominated.begin(), dominated.end(), std::back_inserter(q));
            dominated.clear();
            fronts.push_back(front);
        }
        return fronts;
    }

} // namespace Operon
