// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/collections/bitset.hpp"
#include "operon/core/individual.hpp"

namespace Operon {

namespace detail {
    constexpr int INSERTIONSORT = 7;

    class BitsetManager {
        using word_t = uint64_t; // NOLINT

        static constexpr size_t FIRST_WORD_RANGE = 0;
        static constexpr size_t LAST_WORD_RANGE = 1;
        static constexpr size_t N_BIT_ADDR = 6; // 2^6 = 64
        static constexpr word_t WORD_MASK = ~word_t { 0 };
        static constexpr size_t WORD_SIZE = std::numeric_limits<word_t>::digits;

        std::vector<std::vector<word_t>> bitsets_;
        std::vector<std::array<size_t, 2>> bsRanges_;
        std::vector<int> wordRanking_; //Ranking of each bitset word. A bitset word contains 64 solutions.
        std::vector<int> ranking_, ranking0_;
        int maxRank_ = 0;
        std::vector<word_t> incrementalBitset_;
        size_t incBsFstWord_{}, incBsLstWord_{};

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
            size_t fw = bsRanges_[solutionId][FIRST_WORD_RANGE];
            size_t lw = bsRanges_[solutionId][LAST_WORD_RANGE];
            if (lw > incBsLstWord_) {
                lw = incBsLstWord_;
            }
            if (fw < incBsFstWord_) {
                fw = incBsFstWord_;
            }
            if (fw > lw) {
                return;
            }
            word_t word = 0;
            size_t i = 0;
            int rank = 0;
            size_t offset = 0;

            for (; fw <= lw; fw++) {
                word = bitsets_[solutionId][fw] & incrementalBitset_[fw];
                if (word != 0) {
                    i = static_cast<int>(Bitset<>::CountTrailingZeros(static_cast<word_t>(word)));
                    offset = static_cast<size_t>(fw) * WORD_SIZE;
                    do {
                        if (ranking_[offset + i] >= rank) {
                            rank = ranking_[offset + i] + 1;
                        }
                        i++;
                        word_t w = word >> i; // NOLINT
                        i += static_cast<bool>(w) ? Bitset<>::CountTrailingZeros(w) : WORD_SIZE;
                    } while (i < WORD_SIZE && rank <= wordRanking_[fw]);
                    if (rank > maxRank_) {
                        maxRank_ = rank;
                        break;
                    }
                }
            }
            ranking_[solutionId] = rank;
            ranking0_[initSolId] = rank;
            i = solutionId >> N_BIT_ADDR;
            if (rank > wordRanking_[i]) {
                wordRanking_[i] = rank;
            }
        }

        void UpdateIncrementalBitset(size_t solutionId)
        {
            auto wordIndex = solutionId >> N_BIT_ADDR;
            //int shiftDistance = solutionId & 0x3f;
            auto shiftDistance = solutionId;
            incrementalBitset_[wordIndex] |= (word_t { 1 } << shiftDistance);
            if (incBsLstWord_ < wordIndex) {
                incBsLstWord_ = static_cast<int>(wordIndex);
            }
            if (incBsFstWord_ > wordIndex) {
                incBsFstWord_ = static_cast<int>(wordIndex);
            }
        }

        auto InitializeSolutionBitset(size_t solutionId) -> bool
        {
            //int const shiftDistance = solutionId & 0x3f;
            auto const shiftDistance = solutionId;
            auto wordIndex = solutionId >> N_BIT_ADDR;
            if (wordIndex < incBsFstWord_ || 0 == solutionId) {
                bsRanges_[solutionId][FIRST_WORD_RANGE] = std::numeric_limits<int>::max();
                return false;
            } 
            if (wordIndex == incBsFstWord_) { //only 1 word in common
                bitsets_[solutionId] = std::vector<word_t>(wordIndex + 1);
                auto intersection = incrementalBitset_[incBsFstWord_] & ~(WORD_MASK << shiftDistance);
                if (intersection != 0) {
                    bsRanges_[solutionId][FIRST_WORD_RANGE] = wordIndex;
                    bsRanges_[solutionId][LAST_WORD_RANGE] = wordIndex;
                    bitsets_[solutionId][wordIndex] = intersection;
                }
                return intersection != 0;
            }
            //more than one word in common
            auto lw = incBsLstWord_ < wordIndex ? incBsLstWord_ : wordIndex;
            bsRanges_[solutionId][FIRST_WORD_RANGE] = incBsFstWord_;
            bsRanges_[solutionId][LAST_WORD_RANGE] = lw;
            bitsets_[solutionId] = std::vector<word_t>(lw + 1);
            std::copy_n(incrementalBitset_.begin() + std::make_signed_t<size_t>(incBsFstWord_), lw - incBsFstWord_ + 1,
                    bitsets_[solutionId].begin() + std::make_signed_t<size_t>(incBsFstWord_));
            if (incBsLstWord_ >= wordIndex) { // update (compute intersection) the last word
                bitsets_[solutionId][lw] = incrementalBitset_[lw] & ~(WORD_MASK << shiftDistance);
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
            auto n = nSolutions - 1;
            size_t wordIndex = static_cast<int>(n >> N_BIT_ADDR);
            ranking_.resize(nSolutions, 0);
            ranking0_.resize(nSolutions, 0);
            wordRanking_.resize(nSolutions, 0);
            bitsets_.resize(nSolutions);
            bsRanges_.resize(nSolutions);
            incrementalBitset_.resize(wordIndex + 1);
            incBsLstWord_ = 0;
            incBsFstWord_ = std::numeric_limits<int>::max();
            maxRank_ = 0;
        }
    };

    inline auto CompareLex(std::vector<Operon::Scalar> const& s1, std::vector<Operon::Scalar> const& s2, size_t fromObj, size_t toObj) -> int
    {
        for (; fromObj < toObj; fromObj++) {
            if (s1[fromObj] < s2[fromObj]) {
                return -1;
            }
            if (s1[fromObj] > s2[fromObj]) {
                return 1;
            }
        }
        return 0;
    }

    inline auto MergeSort(std::vector<std::vector<Operon::Scalar>>& src, std::vector<std::vector<Operon::Scalar>>& dest, size_t low, size_t high, size_t obj, size_t toObj) -> bool
    {
        size_t i{0};
        size_t j{0};
        size_t s{0};
        size_t destLow = low;
        size_t length = high - low;

        if (length < INSERTIONSORT) {
            bool alreadySorted { true };
            for (i = low; i < high; i++) {
                for (j = i; j > low && CompareLex(dest[j - 1], dest[j], obj, toObj) > 0; j--) {
                    alreadySorted = false;
                    dest[j].swap(dest[j - 1]);
                }
            }
            return alreadySorted; // if temp==null, src is already sorted
        }
        size_t mid = (low + high) / 2;
        bool isSorted = MergeSort(dest, src, low, mid, obj, toObj) & MergeSort(dest, src, mid, high, obj, toObj); // NOLINT

        // If list is already sorted, just copy from src to dest.
        if (src[mid - 1][obj] <= src[mid][obj]) {
            std::copy_n(src.begin() + static_cast<int64_t>(low), length, dest.begin() + static_cast<int64_t>(destLow));
            return isSorted;
        }

        for (s = low, i = low, j = mid; s < high; s++) {
            if (j >= high) { // NOLINT
                dest[s] = src[i++];
            } else if (i < mid && CompareLex(src[i], src[j], obj, toObj) <= 0) {
                dest[s] = src[i++];
            } else {
                dest[s] = src[j++];
            }
        }
        return false;
    }

} // namespace detail

    auto
    MergeSorter::Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result {
        auto n = pop.size();
        auto m = pop.front().Size(); 
        detail::BitsetManager bsm(n);

        std::vector<int> ranking;

        auto const solId = m;
        auto const sortIndex = solId + 1;

        std::vector<std::vector<Operon::Scalar>> population;
        std::vector<std::vector<Operon::Scalar>> work;

        work.resize(n);
        population.resize(n);

        for (size_t i = 0; i < n; ++i) {
            population[i].resize(sortIndex + 1);
            std::copy_n(pop[i].Fitness.begin(), m, population[i].begin());
            population[i][solId] = static_cast<Operon::Scalar>(i);
            population[i][sortIndex] = static_cast<Operon::Scalar>(i); // because pop is already sorted when passed to Sort
        }

        size_t solutionId{0};
        bool dominance{ false };
        work = population;
        detail::MergeSort(population, work, 0, n, 1, 2);
        population = work;
        for (decltype(n) p = 0; p < n; p++) {
            solutionId = static_cast<size_t>(population[p][sortIndex]);
            dominance |= bsm.InitializeSolutionBitset(solutionId);
            bsm.UpdateIncrementalBitset(solutionId);
            if (2 == m) {
                auto initSolId = static_cast<size_t>(population[p][solId]);
                bsm.ComputeSolutionRanking(solutionId, initSolId);
            }
        }

        if (m > 2) {
            dominance = false;
            decltype(m) lastObjective = m - 1;
            work = population;
            for (auto obj = 2UL; obj < m; obj++) {
                if (detail::MergeSort(population, work, 0, n, obj, obj + 1UL)) { 
                    // Population has the same order as in previous objective
                    if (obj == lastObjective) {
                        for (decltype(n) p = 0; p < n; p++) {
                            bsm.ComputeSolutionRanking(static_cast<int>(population[p][sortIndex]), static_cast<int>(population[p][solId]));
                        }
                    }
                    continue;
                }
                population = work;
                bsm.ClearIncrementalBitset();
                dominance = false;
                for (decltype(n) p = 0; p < n; p++) {
                    auto initSolId = static_cast<int>(population[p][solId]);
                    solutionId = static_cast<int>(population[p][sortIndex]);
                    if (obj < lastObjective) {
                        dominance |= bsm.UpdateSolutionDominance(solutionId);
                    } else {
                        bsm.ComputeSolutionRanking(solutionId, initSolId);
                    }
                    bsm.UpdateIncrementalBitset(solutionId);
                }
            }
        }

        ranking = bsm.GetRanking();

        auto rmax = *std::max_element(ranking.begin(), ranking.end());
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (auto i = 0UL; i < n; i++) {
            fronts[ranking[i]].push_back(i);
        }

        return fronts;
    }
} // namespace Operon
