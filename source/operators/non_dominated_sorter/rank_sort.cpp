// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/operators/non_dominated_sorter/rank_sort.hpp"
#include "operon/collections/bitset.hpp"

namespace Operon {

namespace detail {
    template<typename T>
    struct Item {
        T Value;
        size_t Index;

        auto operator<(Item other) const -> bool { return Value < other.Value; }
    };
} // namespace detail

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    auto RankSorter::RankOrdinal(Operon::Span<Operon::Individual const> pop)  -> NondominatedSorterBase::Result
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
        for (auto i : p(Eigen::seq(0, n - 2), 0)) {
            if (maxp(i) == n - 1) {
                continue;
            }
            auto rankI = rank(i);
            auto minpI = minp(i);

            for (auto j : p(Eigen::seq(maxp(i) + 1, n - 1), maxc(i))) {
                if (minp(j) < minpI || rankI != rank(j)) {
                    continue;
                }
                if ((r.col(i) < r.col(j)).all()) {
                    ++rank(j);
                }
            }
        }
        std::vector<std::vector<size_t>> fronts(rank.maxCoeff() + 1);
        for (auto i = 0; i < n; ++i) {
            fronts[rank(i)].push_back(i);
        }
        return fronts;
    }
#endif

    auto RankSorter::RankIntersect(Operon::Span<Operon::Individual const> pop) -> NondominatedSorterBase::Result
    {
        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();

        using Bitset = Operon::Bitset<uint64_t>;

        Bitset b(n, Bitset::OneBlock);
        std::vector<Bitset> bs(n);                          // vector of bitsets (one for each individual)
        std::vector<std::pair<size_t, size_t>> br(n);       // vector of ranges keeping track of the first/last non-zero blocks
        std::vector<detail::Item<Operon::Scalar>> items(n); // these items hold the values to be sorted along with their associated index

        auto const nb = b.NumBlocks();
        for (size_t i = 0; i < n; ++i) {
            items[i].Index = i;
            b.Reset(i);
            bs[i] = b;
            br[i] = {0, nb - 1};
        }
        std::vector<Bitset> rk;         // vector of sets keeping track of individuals whose rank was updated
        rk.emplace_back(n, Bitset::OneBlock);

        std::vector<size_t> rank(n, 0);

        for (size_t k = 1; k < m; ++k) {
            for (auto& item : items) {
                item.Value = pop[item.Index][k];
            }
            std::stable_sort(items.begin(), items.end());
            b.Fill(Bitset::OneBlock);

            for (auto [_, i] : items) {
                b.Reset(i);
                auto [lo, hi] = br[i];
                if (lo > hi) { continue; }

                auto* p = bs[i].Data();
                auto const* q = b.Data();
                
                // tighten the bounds around empty blocks
                while(lo <= hi && !(p[lo] & q[lo])) { ++lo; } // NOLINT
                while(lo <= hi && !(p[hi] & q[hi])) { --hi; } // NOLINT
                br[i] = {lo, hi};

                if (k < m-1) {
                    // perform the set intersection
                    for (size_t j = lo; j <= hi; ++j) {
                        p[j] &= q[j];
                    }
                } else {
                    auto rnk = rank[i];
                    if (rnk + 1UL == rk.size()) {
                        rk.emplace_back(n, Bitset::ZeroBlock);
                    }
                    auto* r = rk[rnk].Data();
                    auto* s = rk[rnk + 1].Data();

                    for (size_t j = lo; j <= hi; ++j) {
                        auto v = p[j] & q[j] & r[j]; // obtain the dominance set
                        r[j] &= ~v;                  // remove dominated individuals from current rank set
                        s[j] |= v;                   // add the individuals to the next rank set

                        auto o = Bitset::BlockSize * j;
                        while(v) {
                            auto x = o + Bitset::CountTrailingZeros(v);
                            v &= (v-1);
                            ++rank[x];
                        }
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> fronts;
        fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1);
        for (size_t i = 0UL; i < n; ++i) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }
} // namespace Operon
