// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research
//
// This implementation is a port of the original implementation in java by Moreno et al.
// Moreno et al. - "Merge Nondominated Sorting Algorithm for Many-Objective Optimization" 
// https://ieeexplore.ieee.org/document/9000950
// https://github.com/jMetal/jMetal/blob/master/jmetal-core/src/main/java/org/uma/jmetal/util/ranking/impl/MergeNonDominatedSortRanking.java
//
// The only changes to the original version are some C++-specific optimizations (e.g. using swap at line 216 and eliminating a temporary)
// The trailing zeros function was taken from https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightLinear
// Java left shift of long int uses only the six lowest-order bits, therefore (1L << x) in java is equivalent with (1L << (x & 0x3f)) in C++

#ifndef OPERON_PARETO_MERGE_NONDOMINATED_SORT
#define OPERON_PARETO_MERGE_NONDOMINATED_SORT

#include <algorithm>
#include <iterator>

#include "core/individual.hpp"
#include "core/operator.hpp"

namespace Operon {
namespace detail {
    constexpr int INSERTIONSORT = 7;

    inline int TrailingZeros(uint64_t v)
    {
        int c;
        if (v) {
            v = (v ^ (v - 1)) >> 1; // Set v's trailing 0s to 1s and zero rest
            for (c = 0; v; c++) {
                v >>= 1;
            }
        } else {
            c = CHAR_BIT * sizeof(v);
        }
        return c;
    }

    class BitsetManager {
        static constexpr int FIRST_WORD_RANGE = 0;
        static constexpr int LAST_WORD_RANGE = 1;
        static constexpr int N_BIT_ADDR = 6;
        static constexpr int WORD_SIZE = 1 << N_BIT_ADDR;
        static constexpr long WORD_MASK = 0xffffffffffffffffL;

        std::vector<std::vector<long>> bitsets;
        std::vector<std::array<int, 2>> bsRanges;
        std::vector<int> wordRanking; //Ranking of each bitset word. A bitset word contains 64 solutions.
        std::vector<int> ranking, ranking0;
        int maxRank;
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
            long word;
            int i = 0, rank = 0, offset;

            for (; fw <= lw; fw++) {
                word = bitsets[solutionId][fw] & incrementalBitset[fw];
                if (word != 0) {
                    i = (int)TrailingZeros(word);
                    offset = fw * WORD_SIZE;
                    do {
                        if (ranking[offset + i] >= rank)
                            rank = ranking[offset + i] + 1;
                        i++;
                        i += (int)TrailingZeros(word >> i);
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
            incrementalBitset[wordIndex] |= (1L << (solutionId & 0x3f));
            if (incBsLstWord < wordIndex)
                incBsLstWord = wordIndex;
            if (incBsFstWord > wordIndex)
                incBsFstWord = wordIndex;
        }

        bool initializeSolutionBitset(int solutionId)
        {
            int const shiftDistance = solutionId & 0x3f;
            int wordIndex = solutionId >> N_BIT_ADDR;
            if (wordIndex < incBsFstWord || 0 == solutionId) {
                bsRanges[solutionId][FIRST_WORD_RANGE] = std::numeric_limits<int>::max();
                return false;
            } else if (wordIndex == incBsFstWord) { //only 1 word in common
                bitsets[solutionId] = std::vector<long>(wordIndex + 1);
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
            bitsets[solutionId] = std::vector<long>(lw + 1);
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
        }

        BitsetManager() = default;

        // constructor
        BitsetManager(size_t nSolutions)
        {
            int n = (int)nSolutions - 1;
            int wordIndex = static_cast<int>(static_cast<size_t>(n) >> N_BIT_ADDR);
            ranking.resize(nSolutions);
            ranking0.resize(nSolutions);
            wordRanking.resize(nSolutions);
            bitsets.resize(nSolutions);
            bsRanges.resize(nSolutions);
            incrementalBitset.resize(wordIndex + 1);
            incBsLstWord = 0;
            incBsFstWord = std::numeric_limits<int>::max();
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

        // this special case should be removed
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

template <bool DominateOnEqual = false>
class MergeNondominatedSorter : public NondominatedSorterBase {
    mutable int SOL_ID;
    mutable int SORT_INDEX;
    mutable int m; // number of objectives
    mutable int n; // population size
    mutable int initialPopulationSize;
    mutable std::vector<int> ranking;
    mutable std::vector<std::vector<Operon::Scalar>> population;
    mutable std::vector<std::vector<Operon::Scalar>> work;
    mutable std::vector<std::vector<int>> duplicatedSolutions;
    mutable detail::BitsetManager bsm;

public:
    inline std::vector<std::vector<size_t>>
    operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
    {
        n = pop.size();
        m = pop.front().Fitness.size();
        ENSURE(m > 1);
        return Sort(pop);
    }

    inline std::vector<std::vector<size_t>>
    Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        Clear();
        initialPopulationSize = pop.size();
        bsm = detail::BitsetManager(n);

        SOL_ID = m;
        SORT_INDEX = SOL_ID + 1;

        work.resize(n);
        population.resize(n);

        for (size_t i = 0; i < n; ++i) {
            population[i].resize(SORT_INDEX + 1);
            std::copy_n(pop[i].Fitness.begin(), m, population[i].begin());
            population[i][SOL_ID] = (Operon::Scalar)i;
        }

        ranking = Sort(); // sort population
        auto rmax = *std::max_element(ranking.begin(), ranking.end());
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (size_t i = 0; i < n; i++) {
            fronts[ranking[i]].push_back(i);
        }
        return fronts;
    }

private:
    void Clear() const {
        ranking.clear();
        population.clear();
        work.clear();
        duplicatedSolutions.clear();
    }

    bool SortFirstObjective() const
    {
        int p = 0;
        std::copy_n(population.begin(), n, work.begin());
        detail::MergeSort(population, work, 0, n, 0, m);
        population[0] = work[0];
        population[0][SORT_INDEX] = 0;
        for (size_t q = 1; q < n; q++) {
            if (0 != detail::CompareLex(population[p], work[q], 0, m)) {
                p++;
                population[p] = work[q];
                population[p][SORT_INDEX] = (Operon::Scalar)p;
            } else {
                duplicatedSolutions.push_back(std::vector<int> { (int)population[p][SOL_ID], (int)work[q][SOL_ID] });
            }
        }
        n = p + 1;
        return n > 1;
    }

    bool SortSecondObjective() const
    {
        int p, solutionId;
        bool dominance { false };
        std::copy_n(population.begin(), n, work.begin());
        detail::MergeSort(population, work, 0, n, 1, 2);
        std::copy_n(work.begin(), n, population.begin());
        for (p = 0; p < n; p++) {
            solutionId = (int)population[p][SORT_INDEX];
            dominance |= bsm.initializeSolutionBitset(solutionId);
            bsm.updateIncrementalBitset(solutionId);
            if (2 == m) {
                int initSolId = (int)population[p][SOL_ID];
                bsm.computeSolutionRanking(solutionId, initSolId);
            }
        }
        return dominance;
    }

    void SortRestOfObjectives() const
    {
        int p, solutionId, initSolId, lastObjective = m - 1;
        bool dominance;
        std::copy_n(population.begin(), n, work.begin());
        for (int obj = 2; obj < m; obj++) {
            if (detail::MergeSort(population, work, 0, n, obj, obj + 1)) { // Population has the same order as in previous objective
                if (obj == lastObjective) {
                    for (p = 0; p < n; p++)
                        bsm.computeSolutionRanking((int)population[p][SORT_INDEX], (int)population[p][SOL_ID]);
                }
                continue;
            }
            std::copy_n(work.begin(), n, population.begin());
            bsm.clearIncrementalBitset();
            dominance = false;
            for (p = 0; p < n; p++) {
                initSolId = ((int)population[p][SOL_ID]);
                solutionId = ((int)population[p][SORT_INDEX]);
                if (obj < lastObjective) {
                    dominance |= bsm.updateSolutionDominance((int)solutionId);
                } else {
                    bsm.computeSolutionRanking((int)solutionId, (int)initSolId);
                }
                bsm.updateIncrementalBitset((int)solutionId);
            }
            if (!dominance) {
                return;
            }
        }
    }

    std::vector<int> Sort() const
    {
        if (SortFirstObjective()) {
            if (SortSecondObjective()) {
                if (m > 2) { SortRestOfObjectives(); }
            }
        }
        ranking = bsm.getRanking();
        // UPDATING DUPLICATED SOLUTIONS
        for (auto const& duplicated : duplicatedSolutions) {
            if (duplicated.empty())
                continue;
            ranking[duplicated[1]] = ranking[duplicated[0]]; // ranking[dup solution]=ranking[original solution]
        }

        n = initialPopulationSize; // equivalent to n += duplicatedSolutions.size();
        return ranking;
    }
};
} // namespace Operon

#endif

