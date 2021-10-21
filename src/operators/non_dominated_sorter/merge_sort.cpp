// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/non_dominated_sorter/merge_sort.hpp"

namespace Operon {

namespace detail {
    constexpr int INSERTIONSORT = 7;

    class BitsetManager {
        using word_t = uint64_t;

        static constexpr int FIRST_WORD_RANGE = 0;
        static constexpr int LAST_WORD_RANGE = 1;
        static constexpr int N_BIT_ADDR = 6; // 2^6 = 64
        //static constexpr long WORD_MASK = 0xffffffffffffffffL;
        static constexpr word_t WORD_MASK = ~word_t { 0 };
        static constexpr int WORD_SIZE = std::numeric_limits<word_t>::digits;

        std::vector<std::vector<word_t>> bitsets;
        std::vector<std::array<int, 2>> bsRanges;
        std::vector<int> wordRanking; //Ranking of each bitset word. A bitset word contains 64 solutions.
        std::vector<int> ranking, ranking0;
        int maxRank = 0;
        std::vector<long> incrementalBitset;
        int incBsFstWord, incBsLstWord;

    public:
        std::vector<int> const& getRanking() const { return ranking0; }

        bool updateSolutionDominance(int solutionId)
        {
            int fw = bsRanges[solutionId][FIRST_WORD_RANGE];
            int lw = bsRanges[solutionId][LAST_WORD_RANGE];
            if (lw > incBsLstWord) {
                lw = incBsLstWord;
            }
            if (fw < incBsFstWord) {
                fw = incBsFstWord;
            }

            while (fw <= lw && 0 == (bitsets[solutionId][fw] & incrementalBitset[fw])) {
                fw++;
            }
            while (fw <= lw && 0 == (bitsets[solutionId][lw] & incrementalBitset[lw])) {
                lw--;
            }
            bsRanges[solutionId][FIRST_WORD_RANGE] = fw;
            bsRanges[solutionId][LAST_WORD_RANGE] = lw;

            if (fw > lw) {
                return false;
            }
            for (; fw <= lw; fw++) {
                bitsets[solutionId][fw] &= incrementalBitset[fw];
            }
            return true;
        }

        void computeSolutionRanking(int solutionId, int initSolId)
        {
            int fw = bsRanges[solutionId][FIRST_WORD_RANGE];
            int lw = bsRanges[solutionId][LAST_WORD_RANGE];
            if (lw > incBsLstWord) {
                lw = incBsLstWord;
            }
            if (fw < incBsFstWord) {
                fw = incBsFstWord;
            }
            if (fw > lw) {
                return;
            }
            word_t word;
            int i = 0, rank = 0, offset;

            for (; fw <= lw; fw++) {
                word = bitsets[solutionId][fw] & incrementalBitset[fw];
                if (word != 0) {
                    i = (int)detail::count_trailing_zeros(static_cast<word_t>(word));
                    offset = fw * WORD_SIZE;
                    do {
                        if (ranking[offset + i] >= rank) {
                            rank = ranking[offset + i] + 1;
                        }
                        i++;
                        word_t w = static_cast<word_t>(word) >> i;
                        i += w ? (int)detail::count_trailing_zeros(w) : (int)WORD_SIZE;
                    } while (i < WORD_SIZE && rank <= wordRanking[fw]);
                    if (rank > maxRank) {
                        maxRank = rank;
                        break;
                    }
                }
            }
            ranking[solutionId] = rank;
            ranking0[initSolId] = rank;
            i = solutionId >> N_BIT_ADDR;
            if (rank > wordRanking[i]) {
                wordRanking[i] = rank;
            }
        }

        void updateIncrementalBitset(int solutionId)
        {
            int wordIndex = solutionId >> N_BIT_ADDR;
            //int shiftDistance = solutionId & 0x3f;
            int shiftDistance = solutionId;
            incrementalBitset[wordIndex] |= (word_t { 1 } << shiftDistance);
            if (incBsLstWord < wordIndex)
                incBsLstWord = wordIndex;
            if (incBsFstWord > wordIndex)
                incBsFstWord = wordIndex;
        }

        bool initializeSolutionBitset(int solutionId)
        {
            //int const shiftDistance = solutionId & 0x3f;
            int const shiftDistance = solutionId;
            int wordIndex = solutionId >> N_BIT_ADDR;
            if (wordIndex < incBsFstWord || 0 == solutionId) {
                bsRanges[solutionId][FIRST_WORD_RANGE] = std::numeric_limits<int>::max();
                return false;
            } else if (wordIndex == incBsFstWord) { //only 1 word in common
                bitsets[solutionId] = std::vector<word_t>(wordIndex + 1);
                long intersection = incrementalBitset[incBsFstWord] & ~(WORD_MASK << shiftDistance);
                if (intersection != 0) {
                    bsRanges[solutionId][FIRST_WORD_RANGE] = wordIndex;
                    bsRanges[solutionId][LAST_WORD_RANGE] = wordIndex;
                    bitsets[solutionId][wordIndex] = intersection;
                }
                return intersection != 0;
            }
            //more than one word in common
            int lw = incBsLstWord < wordIndex ? incBsLstWord : wordIndex;
            bsRanges[solutionId][FIRST_WORD_RANGE] = incBsFstWord;
            bsRanges[solutionId][LAST_WORD_RANGE] = lw;
            bitsets[solutionId] = std::vector<word_t>(lw + 1);
            std::copy_n(incrementalBitset.begin() + incBsFstWord, lw - incBsFstWord + 1, bitsets[solutionId].begin() + incBsFstWord);
            if (incBsLstWord >= wordIndex) { // update (compute intersection) the last word
                bitsets[solutionId][lw] = incrementalBitset[lw] & ~(WORD_MASK << shiftDistance);
                if (bitsets[solutionId][lw] == 0) {
                    bsRanges[solutionId][LAST_WORD_RANGE]--;
                }
            }
            return true;
        }

        void clearIncrementalBitset()
        {
            std::fill(incrementalBitset.begin(), incrementalBitset.end(), 0ul);
            incBsLstWord = 0;
            incBsFstWord = std::numeric_limits<int>::max();
            maxRank = 0;
        }

        BitsetManager() = default;

        // constructor
        BitsetManager(size_t nSolutions)
        {
            int n = (int)nSolutions - 1;
            int wordIndex = static_cast<int>(static_cast<size_t>(n) >> N_BIT_ADDR);
            ranking.resize(nSolutions, 0);
            ranking0.resize(nSolutions, 0);
            wordRanking.resize(nSolutions, 0);
            bitsets.resize(nSolutions);
            bsRanges.resize(nSolutions);
            incrementalBitset.resize(wordIndex + 1);
            incBsLstWord = 0;
            incBsFstWord = std::numeric_limits<int>::max();
            maxRank = 0;
        }
    };

    inline int CompareLex(std::vector<Operon::Scalar> const& s1, std::vector<Operon::Scalar> const& s2, size_t fromObj, size_t toObj)
    {
        for (; fromObj < toObj; fromObj++) {
            if (s1[fromObj] < s2[fromObj])
                return -1;
            if (s1[fromObj] > s2[fromObj])
                return 1;
        }
        return 0;
    }

    inline bool MergeSort(std::vector<std::vector<Operon::Scalar>>& src, std::vector<std::vector<Operon::Scalar>>& dest, int low, int high, int obj, int toObj)
    {
        int i, j, s;
        int destLow = low;
        int length = high - low;

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
        int mid = (low + high) / 2;
        bool isSorted = MergeSort(dest, src, low, mid, obj, toObj) & MergeSort(dest, src, mid, high, obj, toObj);

        // If list is already sorted, just copy from src to dest.
        if (src[mid - 1][obj] <= src[mid][obj]) {
            std::copy_n(src.begin() + low, length, dest.begin() + destLow);
            return isSorted;
        }

        for (s = low, i = low, j = mid; s < high; s++) {
            if (j >= high) {
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

    NondominatedSorterBase::Result
    MergeNondominatedSorter::Sort(Operon::Span<Operon::Individual const> pop) const {
        Clear();
        initialPopulationSize = (int)pop.size();
        n = initialPopulationSize;
        m = (int)pop.front().Fitness.size();
        //detail::BitsetManager bsm(n);
        //bsm = detail::BitsetManager(n);
        detail::BitsetManager bsm(n);

        SOL_ID = m;
        SORT_INDEX = SOL_ID + 1;

        work.resize(n);
        population.resize(n);

        for (int i = 0; i < n; ++i) {
            population[i].resize(SORT_INDEX + 1);
            std::copy_n(pop[i].Fitness.begin(), m, population[i].begin());
            population[i][SOL_ID] = (Operon::Scalar)i;
            population[i][SORT_INDEX] = (Operon::Scalar)i; // because pop is already sorted when passed to Sort
        }

        int solutionId;
        bool dominance { false };
        work = population;
        detail::MergeSort(population, work, 0, n, 1, 2);
        population = work;
        for (decltype(n) p = 0; p < n; p++) {
            solutionId = (int)population[p][SORT_INDEX];
            dominance |= bsm.initializeSolutionBitset(solutionId);
            bsm.updateIncrementalBitset(solutionId);
            if (2 == m) {
                int initSolId = (int)population[p][SOL_ID];
                bsm.computeSolutionRanking(solutionId, initSolId);
            }
        }

        if (m > 2) {
            dominance = false;
            decltype(m) lastObjective = m - 1;
            work = population;
            for (int obj = 2; obj < m; obj++) {
                if (detail::MergeSort(population, work, 0, n, obj, obj + 1)) { 
                    // Population has the same order as in previous objective
                    if (obj == lastObjective) {
                        for (decltype(n) p = 0; p < n; p++)
                            bsm.computeSolutionRanking((int)population[p][SORT_INDEX], (int)population[p][SOL_ID]);
                    }
                    continue;
                }
                population = work;
                bsm.clearIncrementalBitset();
                dominance = false;
                for (decltype(n) p = 0; p < n; p++) {
                    auto initSolId = (int)population[p][SOL_ID];
                    solutionId = (int)population[p][SORT_INDEX];
                    if (obj < lastObjective) {
                        dominance |= bsm.updateSolutionDominance(solutionId);
                    } else {
                        bsm.computeSolutionRanking(solutionId, initSolId);
                    }
                    bsm.updateIncrementalBitset(solutionId);
                }
            }
        }

        ranking = bsm.getRanking();
        n = initialPopulationSize; // equivalent to n += duplicatedSolutions.size();

        auto rmax = *std::max_element(ranking.begin(), ranking.end());
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (int i = 0; i < n; i++) {
            fronts[ranking[i]].push_back(i);
        }

        return fronts;
    }

}
