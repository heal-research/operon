// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/types.hpp"

namespace Operon {
    auto DeductiveSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
    {
        size_t n = 0; // total number of sorted solutions
        std::vector<std::vector<size_t>> fronts;

        std::size_t constexpr d = std::numeric_limits<uint64_t>::digits;
        auto const s = static_cast<int>(pop.size());
        auto const nb = s / d + static_cast<std::size_t>(s % d != 0);

        std::vector<uint64_t> dominated(nb);
        std::vector<uint64_t> sorted(nb);

        auto set = [](auto&& range, auto i) { range[i / d] |= (1UL << (d - i % d));}; // set bit i
        [[maybe_unused]] auto reset = [](auto&& range, auto i) { range[i / d] &= ~(1UL << (i % d)); }; // unset bit i
        auto get = [](auto&& range, auto i) -> bool { return range[i / d] & (1UL << (d - i % d)); }; // return bit i

        auto dominatedOrSorted = [&](std::size_t i) { return get(sorted, i) || get(dominated, i); };

        while (n < pop.size()) {
            std::vector<size_t> front;

            for (size_t i = 0; i < pop.size(); ++i) {
                if (dominatedOrSorted(i)) { continue; }

                for (size_t j = i + 1; j < pop.size(); ++j) {
                    if (dominatedOrSorted(j)) { continue; }

                    auto res = ParetoDominance{}(pop[i].Fitness, pop[j].Fitness);
                    if (res == Dominance::Right) { set(dominated, i); }
                    if (res == Dominance::Left) { set(dominated, j); }

                    if (get(dominated, i)) { break; }
                }

                if (!get(dominated, i)) {
                    front.push_back(i);
                    set(sorted, i);
                }
            }

            std::fill(dominated.begin(), dominated.end(), 0UL);
            n += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }
} // namespace Operon
