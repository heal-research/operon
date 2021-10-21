// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/non_dominated_sorter/rank_sort.hpp"

namespace Operon {

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    NondominatedSorterBase::Result RankSorter::SortRank(Operon::Span<Operon::Individual const> pop) const
    {
        const size_t n = pop.size();
        const size_t m = pop.front().Fitness.size();
        // 1) sort indices according to the stable sorting rules
        Mat p(n, m); // permutation matrix
        Mat r(m, n); // ordinal rank matrix
        Vec indices = Vec::LinSpaced(n, 0, n - 1);
        p.col(0) = Eigen::Map<Vec const>(indices.data(), indices.size());
        r(0, p.col(0)) = Vec::LinSpaced(n, 0, n - 1);
        for (size_t i = 1; i < m; ++i) {
            p.col(i) = p.col(i - 1); // this is a critical part of the approach
            std::stable_sort(p.col(i).begin(), p.col(i).end(), [&](auto a, auto b) { return pop[a][i] < pop[b][i]; });
            r(i, p.col(i)) = Vec::LinSpaced(n, 0, n - 1);
        }
        // 2) save min and max positions as well as the column index for the max position
        //    hash individual fitness vectors for fast equality
        Vec maxc(n), minp(n), maxp(n);
        for (size_t i = 0; i < n; ++i) {
            auto [min_, max_] = std::minmax_element(r.col(i).begin(), r.col(i).end());
            minp(i) = *min_;
            maxp(i) = *max_;
            maxc(i) = std::distance(r.col(i).begin(), max_);
        }
        // 3) compute ranks / fronts
        Vec rank = Vec::Zero(n); // individual ranks
        size_t rmax{0};
        for (auto i : p(Eigen::seq(0, n - 2), 0)) {
            if (maxp(i) == n - 1) {
                continue;
            }
            auto rank_i = rank(i);
            auto minp_i = minp(i);

            for (auto j : p(Eigen::seq(maxp(i) + 1, n - 1), maxc(i))) {
                if (minp(j) < minp_i || rank_i < rank(j)) {
                    continue;
                }
                if ((r.col(i) < r.col(j)).all()) {
                    rank(j) = rank_i + 1;
                    rmax = std::max(rmax, rank(j));
                }
            }
        }
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (size_t i = 0; i < n; ++i) {
            fronts[rank(i)].push_back(i);
        }
        return fronts;
    }
#endif

    NondominatedSorterBase::Result RankSorter::SortBit(Operon::Span<Operon::Individual const> pop) const
    {
        using block_type = uint64_t;
        using bitset = std::vector<block_type>; // a bitset is actually just a collection of 64-bit blocks

        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();

        size_t const block_bits = std::numeric_limits<block_type>::digits;
        size_t const num_blocks = (n / block_bits) + (n % block_bits != 0);
        block_type const block_max = std::numeric_limits<block_type>::max();
        std::vector<size_t> indices(n);                   // vector of indices that get sorted according to each objective
        std::vector<block_type> b(num_blocks, block_max); // this bitset will help us compute the intersections
        b.back() >>= (block_bits * num_blocks - n);       // the bits in the last block that are over n must be set to zero
        std::vector<bitset> bs(n);                        // vector of bitsets (one for each individual)
        std::vector<std::pair<size_t, size_t>> br(n);     // vector of ranges keeping track of the first/last non-zero blocks
        for (size_t i = 0; i < n; ++i) {
            indices[i] = i;
            b[i / block_bits] &= ~(block_type{1} << (i % block_bits)); // reset the bit corresponding to the current individual
            bs[i] = b;
            br[i] = {0, num_blocks-1};
        }
        std::vector<bitset> rk(n);           // vector of sets keeping track of individuals whose rank was updated
        rk[0].resize(num_blocks, block_max); // initially all the individuals have the same rank

        std::vector<size_t> rank(n, 0);
        std::vector<Operon::Scalar> values(n);
        for (size_t k = 1; k < m; ++k) {
            std::transform(pop.begin(), pop.end(), values.begin(), [&](auto const& ind) { return ind[k]; });
            std::stable_sort(indices.begin(), indices.end(), [&](auto i, auto j) { return values[i] < values[j]; });
            std::fill(b.begin(), b.end(), block_max);
            b.back() >>= (block_bits * num_blocks - n);

            if (k < m - 1) {
                for (auto i : indices) {
                    b[i / block_bits] &= ~(block_type{1} << (i % block_bits));
                    auto [lo, hi] = br[i];
                    if (lo > hi) { continue; }

                    auto p = bs[i].data();
                    auto const q = b.data();

                    // tighten the interval around non-zero blocks
                    while(lo <= hi && !(p[lo] & q[lo])) ++lo;
                    while(lo <= hi && !(p[hi] & q[hi])) --hi;

                    // perform the blockwise intersection between sets
                    for (size_t j = lo; j <= hi; ++j) {
                        p[j] &= q[j];
                    }
                    br[i] = {lo, hi};
                }
            } else {
                for (auto i : indices) {
                    b[i / block_bits] &= ~(block_type{1} << (i % block_bits)); // reset bit for current individual
                    auto [lo, hi] = br[i];
                    if (lo > hi) { continue; }

                    auto p = bs[i].data();
                    auto const q = b.data();

                    // tighten the interval around non-zero blocks
                    while(lo <= hi && !(p[lo] & q[lo])) ++lo;
                    while(lo <= hi && !(p[hi] & q[hi])) --hi;

                    auto r = rk[rank[i]].data();
                    auto rank_ = rank[i] + 1;
                    if (rk[rank_].empty()) {
                        rk[rank_].resize(num_blocks, 0);
                    }

                    // perform the blockwise intersection between sets
                    for (size_t j = lo; j <= hi; ++j) {
                        auto o = block_bits * j;     // ordinal offset to get the individual index
                        auto v = p[j] & q[j] & r[j]; // obtain the dominance set
                        r[j] &= ~v;                  // remove dominance set individuals from current rank set

                        // iterate over the set bits in this block and update ranks
                        while (v) {
                            auto x = o + detail::count_trailing_zeros(v);
                            v &= (v - 1);
                            rank[x] = rank_;
                            rk[rank_][x / block_bits] |= (block_type{1} << (x % block_bits));
                        }
                    }
                }
            }
        }
        std::vector<std::vector<size_t>> fronts(*std::max_element(rank.begin(), rank.end()) + 1);
        for (auto i : indices) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }

} // namespace Operon
