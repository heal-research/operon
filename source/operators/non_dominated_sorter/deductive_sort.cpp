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

namespace {
    constexpr std::size_t kBitsPerWord = std::numeric_limits<uint64_t>::digits;

    auto SetBit(Operon::Vector<uint64_t>& bits, std::size_t i) -> void {
        bits[i / kBitsPerWord] |= (1UL << (kBitsPerWord - i % kBitsPerWord)); // NOLINT(clang-analyzer-core.BitwiseShift)
    }
    [[maybe_unused]] auto ResetBit(Operon::Vector<uint64_t>& bits, std::size_t i) -> void {
        bits[i / kBitsPerWord] &= ~(1UL << (i % kBitsPerWord));
    }
    auto GetBit(Operon::Vector<uint64_t> const& bits, std::size_t i) -> bool {
        return (bits[i / kBitsPerWord] & (1UL << (kBitsPerWord - i % kBitsPerWord))) != 0U; // NOLINT(clang-analyzer-core.BitwiseShift)
    }
    auto IsDominatedOrSorted(Operon::Vector<uint64_t> const& dominated, Operon::Vector<uint64_t> const& sorted, std::size_t i) -> bool {
        return GetBit(sorted, i) || GetBit(dominated, i);
    }

    // Compare solution i against all candidates j > i, marking domination bits.
    // Returns true if i was dominated by some j.
    auto CheckDominance(Operon::Span<Operon::Individual const> pop,
                        Operon::Vector<uint64_t>& dominated,
                        Operon::Vector<uint64_t> const& sorted,
                        std::size_t i) -> bool
    {
        for (std::size_t j = i + 1; j < pop.size(); ++j) {
            if (IsDominatedOrSorted(dominated, sorted, j)) { continue; }
            auto res = Operon::ParetoDominance{}(pop[i].Fitness, pop[j].Fitness);
            if (res == Operon::Dominance::Right) { SetBit(dominated, i); }
            if (res == Operon::Dominance::Left)  { SetBit(dominated, j); }
            if (GetBit(dominated, i)) { break; }
        }
        return GetBit(dominated, i);
    }
} // anonymous namespace

namespace Operon {
    auto DeductiveSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
    {
        size_t n = 0; // total number of sorted solutions
        Operon::Vector<Operon::Vector<size_t>> fronts;

        auto const s = static_cast<int>(pop.size());
        auto const nb = (s / kBitsPerWord) + static_cast<std::size_t>(s % kBitsPerWord != 0);

        Operon::Vector<uint64_t> dominated(nb);
        Operon::Vector<uint64_t> sorted(nb);

        while (n < pop.size()) {
            Operon::Vector<size_t> front;

            for (size_t i = 0; i < pop.size(); ++i) {
                if (IsDominatedOrSorted(dominated, sorted, i)) { continue; }

                if (!CheckDominance(pop, dominated, sorted, i)) {
                    front.push_back(i);
                    SetBit(sorted, i);
                }
            }

            std::fill(dominated.begin(), dominated.end(), 0UL);
            n += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }
} // namespace Operon
