// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cpp-sort/sorters/merge_sorter.h>
#include <cstddef>
#include <cstdint>
#include <bit>
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <vector>

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"

namespace Operon {

namespace detail {
    class BitsetManager {
        using word_t = uint64_t; // NOLINT

        static constexpr size_t FIRST_WORD_RANGE = 0;
        static constexpr size_t LAST_WORD_RANGE = 1;
        static constexpr word_t WORD_MASK = ~word_t{0UL};
        static constexpr size_t WORD_SIZE = std::numeric_limits<word_t>::digits;

        std::vector<std::vector<word_t>> bitsets_;
        std::vector<std::array<size_t, 2>> bsRanges_;
        std::vector<int> wordRanking_; //Ranking of each bitset word. A bitset word contains 64 solutions.
        std::vector<int> ranking_, ranking0_;
        int maxRank_ = 0;
        std::vector<word_t> incrementalBitset_;
        size_t incBsFstWord_{std::numeric_limits<int>::max()};
        size_t incBsLstWord_{0};

    public:
        [[nodiscard]] auto GetRanking() const -> std::vector<int> const& { return ranking0_; }

        auto UpdateSolutionDominance(size_t solutionId) -> bool
        {
            size_t fw = bsRanges_[solutionId][FIRST_WORD_RANGE];
            size_t lw = bsRanges_[solutionId][LAST_WORD_RANGE];
            if (lw > incBsLstWord_) {
                lw = incBsLstWord_;
            }
            if (fw < incBsFstWord_) {
                fw = incBsFstWord_;
            }

            while (fw <= lw && 0 == (bitsets_[solutionId][fw] & incrementalBitset_[fw])) {
                fw++;
            }
            while (fw <= lw && 0 == (bitsets_[solutionId][lw] & incrementalBitset_[lw])) {
                lw--;
            }
            bsRanges_[solutionId][FIRST_WORD_RANGE] = fw;
            bsRanges_[solutionId][LAST_WORD_RANGE] = lw;

            if (fw > lw) {
                return false;
            }
            for (; fw <= lw; fw++) {
                bitsets_[solutionId][fw] &= incrementalBitset_[fw];
            }
            return true;
        }

        void ComputeSolutionRanking(size_t solutionId, size_t initSolId)
        {
            auto fw = bsRanges_[solutionId][FIRST_WORD_RANGE];
            auto lw = bsRanges_[solutionId][LAST_WORD_RANGE];

            if (lw > incBsLstWord_) {
                lw = incBsLstWord_;
            }
            if (fw < incBsFstWord_) {
                fw = incBsFstWord_;
            }
            if (fw > lw) {
                return;
            }
            word_t word{};
            size_t i = 0;
            int rank = 0;
            size_t offset = 0;

            for (; fw <= lw; fw++) {
                word = bitsets_[solutionId][fw] & incrementalBitset_[fw];

                if (word != 0) {
                    i = std::countr_zero(static_cast<word_t>(word));
                    offset = fw * WORD_SIZE;
                    do {
                        auto r = ranking_[offset+i];
                        if (r >= rank) { rank = ranking_[offset + i] + 1; }
                        i++;
                        i += std::countr_zero(word >> i);
                    } while (i < WORD_SIZE && rank <= wordRanking_[fw]);
                    if (rank > maxRank_) {
                        maxRank_ = rank;
                        break;
                    }
                }
            }
            ranking_[solutionId] = rank;
            ranking0_[initSolId] = rank;
            i = solutionId / WORD_SIZE;
            if (rank > wordRanking_[i]) {
                wordRanking_[i] = rank;
            }
        }

        void UpdateIncrementalBitset(size_t solutionId)
        {
            auto wordIndex = solutionId / WORD_SIZE;
            incrementalBitset_[wordIndex] |= (word_t{1} << solutionId);
            if (incBsLstWord_ < wordIndex) {
                incBsLstWord_ = static_cast<int>(wordIndex);
            }
            if (incBsFstWord_ > wordIndex) {
                incBsFstWord_ = wordIndex;
            }
        }

        auto InitializeSolutionBitset(size_t solutionId) -> bool
        {
            auto wordIndex = solutionId / WORD_SIZE;
            if (wordIndex < incBsFstWord_ || 0 == solutionId) {
                bsRanges_[solutionId][FIRST_WORD_RANGE] = std::numeric_limits<int>::max();
                return false;
            }
            if (wordIndex == incBsFstWord_) { //only 1 word in common
                bitsets_[solutionId].resize(wordIndex + 1);
                auto intersection = incrementalBitset_[incBsFstWord_] & ~(WORD_MASK << solutionId);
                if (intersection != 0) {
                    bsRanges_[solutionId][FIRST_WORD_RANGE] = wordIndex;
                    bsRanges_[solutionId][LAST_WORD_RANGE] = wordIndex;
                    bitsets_[solutionId][wordIndex] = intersection;
                }
                return intersection != 0;
            }
            // more than one word in common
            auto lw = incBsLstWord_ < wordIndex ? incBsLstWord_ : wordIndex;
            bsRanges_[solutionId][FIRST_WORD_RANGE] = incBsFstWord_;
            bsRanges_[solutionId][LAST_WORD_RANGE] = lw;
            bitsets_[solutionId] = std::vector<word_t>(lw + 1);
            std::copy_n(incrementalBitset_.data() + incBsFstWord_, lw - incBsFstWord_ + 1, bitsets_[solutionId].data() + incBsFstWord_);
            if (incBsLstWord_ >= wordIndex) { // update (compute intersection) the last word
                bitsets_[solutionId][lw] = incrementalBitset_[lw] & ~(WORD_MASK << solutionId);
                if (bitsets_[solutionId][lw] == 0) {
                    bsRanges_[solutionId][LAST_WORD_RANGE]--;
                }
            }
            return true;
        }

        void ClearIncrementalBitset()
        {
            std::fill(incrementalBitset_.begin(), incrementalBitset_.end(), 0UL);
            incBsLstWord_ = 0;
            incBsFstWord_ = std::numeric_limits<int>::max();
            maxRank_ = 0;
        }

        BitsetManager() = default;

        // constructor
        explicit BitsetManager(size_t nSolutions)
        {
            ranking_.resize(nSolutions, 0);
            ranking0_.resize(nSolutions, 0);
            wordRanking_.resize(nSolutions, 0);
            bitsets_.resize(nSolutions);
            bsRanges_.resize(nSolutions);
            incrementalBitset_.resize(nSolutions / WORD_SIZE + static_cast<uint64_t>(nSolutions % WORD_SIZE != 0));
        }
    };

    struct Item {
        int Index;
        Operon::Scalar Value;

        friend auto operator<(Item a, Item b) { return a.Value < b.Value; }
    };
} // namespace detail

    auto
    MergeSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*eps*/) const -> NondominatedSorterBase::Result {
        auto const n = static_cast<int>(pop.size());
        auto const m = static_cast<int>(pop.front().Size());

        detail::BitsetManager bsm(n);

        cppsort::merge_sorter sorter;
        std::vector<detail::Item> items(n);
        for (auto i = 0; i < n; ++i) {
            items[i] = { i, pop[i][1] };
        }
        sorter(items);

        for (auto obj = 1; obj < m; ++obj) {
            if (obj > 1) {
                for (auto& [i, v] : items) { v = pop[i][obj]; }
                sorter(items);
                bsm.ClearIncrementalBitset();
            }

            auto dominance{false};
            for (auto i = 0; i < n; ++i) {
                auto [j, v] = items[i];
                if (obj == 1) {
                    dominance |= bsm.InitializeSolutionBitset(j);
                } else if (obj < m-1) {
                    dominance |= bsm.UpdateSolutionDominance(j);
                }
                if (obj == m-1) {
                    bsm.ComputeSolutionRanking(j, j);
                }
                bsm.UpdateIncrementalBitset(j);
            }

            if (!dominance) { break; }
        }

        auto ranking = bsm.GetRanking();
        auto rmax = *std::max_element(ranking.begin(), ranking.end());
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (auto i = 0; i < n; i++) {
            fronts[ranking[i]].push_back(i);
        }

        return fronts;
    }
} // namespace Operon
