// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <cpp-sort/sorters.h>
#include "operon/operators/non_dominated_sorter/rank_sort.hpp"

namespace Operon {

namespace detail {
    template<typename T>
    struct Item {
        T Value;
        size_t Index;

        auto operator<(Item other) const -> bool { return Value < other.Value; }
    };

    template<typename T /* block type */, size_t S = std::numeric_limits<T>::digits /* block size in bits */>
    class Bitset {
        std::vector<T> blocks_;
        size_t numBits_{};

        [[nodiscard]] inline auto BlockIndex(size_t i) const { return i / S; }
        [[nodiscard]] inline auto BitIndex(size_t i) const { return BlockIndex(i) + i % S; }

        public:
        static constexpr T ZeroBlock = T{0}; // a block with all the bits set to zero
        static constexpr T OneBlock = ~ZeroBlock; // a block with all the bits set to one
        static constexpr size_t BlockSize = S;
        using Block = T;

        Bitset() = default;

        explicit Bitset(size_t n, T blockInit)
            : numBits_(n)
        {
            size_t const nb = (n / S) + (n % S != 0); // NOLINT
            blocks_.resize(nb, blockInit);
            blocks_.back() >>= S * nb - n; // zero the bits in the last block that are over n
        }

        inline void Fill(T value) { std::fill(blocks_.begin(), blocks_.end(), value); }

        inline void Resize(size_t numBlocks, T value = T{0}) { blocks_.resize(numBlocks, value); }

        inline void Set(size_t i) {
            assert(i < numBits_);
            blocks_[i / S] |= (T{1} << (i % S));
        }

        inline void Reset(size_t i) {
            assert(i < numBits_);
            blocks_[i / S] &= ~(T{1} << (i % S));
        }

        auto View() -> Operon::Span<T> { return { blocks_ }; } 
        [[nodiscard]] auto View() const -> Operon::Span<T const> { return { blocks_ }; }

        [[nodiscard]] inline auto NumBlocks() const -> size_t { return blocks_.size(); }

        [[nodiscard]] inline auto Empty() const -> bool { return blocks_.empty(); }

        inline auto operator[](size_t i) -> T& {
            return blocks_[i];
        }

        inline auto operator[](size_t i) const -> T {
            return blocks_[i];
        }

        friend auto operator&(Bitset const& lhs, Bitset const& rhs) -> Bitset {
            auto result = lhs;
            result &= rhs;
            return result;
        }

        friend auto operator&=(Bitset& lhs, Bitset const& rhs) -> Bitset& {
            assert(lhs.blocks_.size() == rhs.blocks_.size());
            assert(lhs.numBits_ == rhs.numBits_);
            for (size_t i = 0; i < lhs.blocks_.size(); ++i) {
                lhs[i] &= rhs[i];
            }
            return lhs;
        };
    };
} // namespace detail

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    auto RankSorter::SortRank(Operon::Span<Operon::Individual const> pop)  -> NondominatedSorterBase::Result
    {
        const auto n = static_cast<Eigen::Index>(pop.size());
        const auto m = static_cast<Eigen::Index>(pop.front().Size());
        // 1) sort indices according to the stable sorting rules
        Mat p(n, m); // permutation matrix
        Mat r(m, n); // ordinal rank matrix
        Vec indices = Vec::LinSpaced(n, 0, n - 1);
        p.col(0) = Eigen::Map<Vec const>(indices.data(), indices.size());
        r(0, p.col(0)) = Vec::LinSpaced(static_cast<Eigen::Index>(n), 0, n - 1);
        for (auto i = 1; i < m; ++i) {
            p.col(i) = p.col(i - 1); // this is a critical part of the approach
            std::stable_sort(p.col(i).begin(), p.col(i).end(), [&](auto a, auto b) { return pop[a][i] < pop[b][i]; });
            r(i, p.col(i)) = Vec::LinSpaced(n, 0, n - 1);
        }
        // 2) save min and max positions as well as the column index for the max position
        //    hash individual fitness vectors for fast equality
        Vec maxc(n);
        Vec minp(n);
        Vec maxp(n);
        for (auto i = 0; i < n; ++i) {
            auto [min_, max_] = std::minmax_element(r.col(i).begin(), r.col(i).end());
            minp(i) = *min_;
            maxp(i) = *max_;
            maxc(i) = std::distance(r.col(i).begin(), max_);
        }
        // 3) compute ranks / fronts
        Vec rank = Vec::Zero(n); // individual ranks
        Eigen::Index rmax{0};
        for (auto i : p(Eigen::seq(0, n - 2), 0)) {
            if (maxp(i) == n - 1) {
                continue;
            }
            auto rankI = rank(i);
            auto minpI = minp(i);

            for (auto j : p(Eigen::seq(maxp(i) + 1, n - 1), maxc(i))) {
                if (minp(j) < minpI || rankI < rank(j)) {
                    continue;
                }
                if ((r.col(i) < r.col(j)).all()) {
                    rank(j) = rankI + 1;
                    rmax = std::max(rmax, rank(j));
                }
            }
        }
        std::vector<std::vector<size_t>> fronts(rmax + 1);
        for (auto i = 0; i < n; ++i) {
            fronts[rank(i)].push_back(i);
        }
        return fronts;
    }
#endif

     auto RankSorter::SortBit(Operon::Span<Operon::Individual const> pop) -> NondominatedSorterBase::Result
    {
        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();

        using Bitset = detail::Bitset<uint64_t>;

        Bitset b(n, Bitset::OneBlock);
        std::vector<Bitset> bs(n);                          // vector of bitsets (one for each individual)
        std::vector<std::pair<size_t, size_t>> br(n);       // vector of ranges keeping track of the first/last non-zero blocks
        std::vector<detail::Item<Operon::Scalar>> items(n); // these items hold the values to be sorted along with their associated index

        auto const numBlocks = b.NumBlocks();
        auto const blockSize = Bitset::BlockSize;
        for (size_t i = 0; i < n; ++i) {
            items[i].Index = i;
            b.Reset(i);
            bs[i] = b;
            br[i] = {0, numBlocks - 1};
        }
        std::vector<Bitset> rk(n);         // vector of sets keeping track of individuals whose rank was updated
        rk[0].Resize(n, Bitset::OneBlock);

        std::vector<size_t> rank(n, 0);
        std::vector<Operon::Scalar> values(n);

        cppsort::merge_sorter sorter;
        for (size_t k = 1; k < m; ++k) {
            for (size_t i = 0; i < n; ++i) {
                auto& item = items[i];
                item.Value = pop[item.Index][k];
            }
            sorter(items);
            b.Fill(Bitset::OneBlock);

            for (auto [_, i] : items) {
                b.Reset(i);
                auto [lo, hi] = br[i];
                if (lo > hi) { continue; }

                auto* p = bs[i].View().data();
                auto const* q = b.View().data();

                // tighten the interval around non-zero blocks
                while(lo <= hi && !(p[lo] & q[lo])) { ++lo; } // NOLINT
                while(lo <= hi && !(p[hi] & q[hi])) { --hi; } // NOLINT

                if (k < m-1) {
                    // perform the set intersection
                    for (size_t j = lo; j <= hi; ++j) {
                        p[j] &= q[j];
                    }
                    br[i] = {lo, hi};
                } else {
                    auto& r = rk[rank[i]];
                    auto newRank = rank[i] + 1;
                    if (rk[newRank].Empty()) {
                        rk[newRank].Resize(numBlocks, 0);
                    }
                    auto* s = rk[newRank].View().data();

                    // set intersection + lift ranks + update ranks 
                    for (size_t j = lo; j <= hi; ++j) {
                        auto o = blockSize * j;      // ordinal offset to get the individual index
                        auto v = p[j] & q[j] & r[j]; // obtain the dominance set
                        r[j] &= ~v;                  // remove dominance set individuals from current rank set
                        s[j] |= v;

                        // iterate over the set bits in this block and update ranks
                        while (v) { // NOLINT
                            auto x = o + detail::count_trailing_zeros(v);
                            v &= (v - 1);
                            rank[x] = newRank;
                            //rk[newRank].Set(x);
                        }
                    }
                }
            }
        }
        std::vector<std::vector<size_t>> fronts(*std::max_element(rank.begin(), rank.end()) + 1);
        for (auto [_, i] : items) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }

    NondominatedSorterBase::Result SortBit1(Operon::Span<Operon::Individual const> pop)
    {
        using block_type = uint64_t;
        using bitset = std::vector<block_type>; // a bitset is actually just a collection of 64-bit blocks

        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();

        size_t const block_bits = std::numeric_limits<block_type>::digits;
        size_t const num_blocks = (n / block_bits) + (n % block_bits != 0);
        block_type const block_max = std::numeric_limits<block_type>::max();
        std::vector<block_type> b(num_blocks, block_max);   // this bitset will help us compute the intersections
        b.back() >>= (block_bits * num_blocks - n);         // the bits in the last block that are over n must be set to zero
        std::vector<bitset> bs(n);                          // vector of bitsets (one for each individual)
        std::vector<std::pair<size_t, size_t>> br(n);       // vector of ranges keeping track of the first/last non-zero blocks
        std::vector<detail::Item<Operon::Scalar>> items(n); // these items hold the values to be sorted along with their associated index

        for (size_t i = 0; i < n; ++i) {
            items[i].Index = i;
            b[i / block_bits] &= ~(block_type{1} << (i % block_bits)); // reset the bit corresponding to the current individual
            bs[i] = b;
            br[i] = {0, num_blocks-1};
        }
        std::vector<bitset> rk(n);           // vector of sets keeping track of individuals whose rank was updated
        rk[0].resize(num_blocks, block_max); // initially all the individuals have the same rank

        std::vector<size_t> rank(n, 0);
        std::vector<Operon::Scalar> values(n);
        for (size_t k = 1; k < m; ++k) {
            for (auto i = 0ul; i < n; ++i) {
                auto& item = items[i];
                item.Value = pop[item.Index][k];
            }
            std::stable_sort(items.begin(), items.end());
            std::fill(b.begin(), b.end(), block_max);
            b.back() >>= (block_bits * num_blocks - n);

            if (k < m - 1) {
                for (auto [_, i] : items) {
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
                for (auto [_, i] : items) {
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
        for (auto [_, i] : items) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }
} // namespace Operon
