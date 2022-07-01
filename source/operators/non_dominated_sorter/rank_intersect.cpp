// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include <iostream>

namespace Operon {
namespace detail {
    template <typename Iter>
    inline auto TightenBounds(Iter begin, Iter end) -> std::pair<Iter, Iter>
    {
        while (begin < end && *begin == 0) { ++begin; }
        while (begin < end-1 && *(end-1) == 0) { --end; }
        return { begin, end };
    }

    template<std::ranges::random_access_range R, typename T = typename std::remove_cvref_t<R>::value_type, size_t D = std::numeric_limits<T>::digits>
    requires std::is_integral_v<T>
    inline auto SetBit(R&& r, int i)
    {
        r[i / D] |= (T{1} << (i % D));
    }

    template<std::ranges::random_access_range R, typename T = typename std::remove_cvref_t<R>::value_type, size_t D = std::numeric_limits<T>::digits>
    requires std::is_integral_v<T>
    inline auto ResetBit(R&& r, int i)
    {
        r[i / D] &= ~(T{1} << (i % D));
    }
} // namespace detail

// rank-based non-dominated sorting - intersect version - see https://arxiv.org/abs/2203.13654
auto RankIntersectSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
{
    int const n = static_cast<int>(pop.size());
    int const m = static_cast<int>(pop.front().Size());

    // bitset block type
    using Block = uint64_t;
    auto constexpr ONE = ~Block{0};
    auto constexpr ZERO = Block{0};

    // constants
    size_t constexpr DIGITS{std::numeric_limits<Block>::digits};
    size_t const nb{n / DIGITS + static_cast<size_t>(n % DIGITS != 0)};

    using Item = std::tuple<Operon::Scalar, int>;
    using Span = std::span<Block>;
    using Iter = Span::iterator;
    std::vector<std::pair<Iter, Iter>> bounds; bounds.reserve(n); // vector of ranges keeping track of the first/last non-zero blocks

    // initialization
    std::vector<Block> q(nb, ONE);
    q[nb-1] >>= (DIGITS * nb - n); // zero unused region

    std::unique_ptr<Block> bits(new Block[nb * n]); // allocate all memory at once (slightly more efficient)
    std::vector<Item> items(n); // these items hold the values to be sorted along with their associated index

    auto b = 0; // current non-zero block
    for (auto i = 0; i < n; ++i) {
        std::get<1>(items[i]) = i;
        detail::ResetBit(q, i); // reset bit i
        b += static_cast<int>(q[i / DIGITS] == 0); // advance lower bound when a block becomes zero
        Span s{bits.get() + i * nb, nb};
        std::copy_n(q.begin() + b, nb - b, s.begin() + b); // initialize bitset with q
        bounds.emplace_back(s.begin() + b, s.begin() + nb); // NOLINT
    }

    std::vector<std::vector<Block>> ranksets{ std::vector(nb, ONE) }; // vector of sets keeping track of individuals whose rank was updated
    ranksets.back()[nb-1] >>= DIGITS * nb - n; // zero unused region
    std::vector<int> rank(n, 0); // individual ranks (initially, all zero)
    auto cmp = [eps](auto a, auto b) { return Operon::Less{}(std::get<0>(a), std::get<0>(b), eps); };

    std::vector<size_t> blocks; blocks.reserve(nb);
    for (auto k = 1; k < m; ++k) {
        for (auto& [v, i] : items) { v = pop[i][k]; } // update item values with current k-th objectives
        std::stable_sort(items.begin(), items.end(), cmp); // sort items
        std::fill_n(q.begin(), nb, ONE); // reset q bitset to all ones
        q[nb-1] >>= DIGITS * nb - n; // zero unused region

        for (auto [_, i] : items) {
            detail::ResetBit(q, i); // reset bit i
            auto [lo, hi] = bounds[i]; // get bounds
            if (lo == hi) { continue; }
            Span s{bits.get() + i * nb, nb};
            // compute the intersections of the dominance sets (most runtime intensive part)
            std::transform(lo, hi, q.begin() + std::distance(s.begin(), lo), lo, std::bit_and{});
            bounds[i] = detail::TightenBounds(lo, hi); // update bounds

            if (k == m-1) { // if we're at the last objective, we can update the ranks
                if (rank[i] + 1UL == ranksets.size()) { // new rankset if necessary
                    ranksets.emplace_back(nb, ZERO); // NOLINT
                }
                auto curr = ranksets.begin() + rank[i]; // the pareto front of the current individual
                auto next = curr + 1;                   // next (worse) pareto front

                auto j = std::distance(s.begin(), bounds[i].first);
                for (auto it = bounds[i].first; it != bounds[i].second; ++it, ++j) { // iterate over bitset blocks
                    auto v = *it & (*curr)[j];                                       // final set as intersection of dominance set and rank set
                    (*curr)[j] &= ~v;                                                // remove intersection result from current rank set
                    (*next)[j] |= v;                                                 // add intersection result to next rank set
                    for (; v != 0; v &= (v - 1)) {                                   // iterate over set bits of v
                        ++rank[j * DIGITS + std::countr_zero(v)];                    // increment rank
                    }
                }
            }
        }
    }

    std::vector<std::vector<size_t>> fronts;
    fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1UL);
    for (size_t i = 0UL; i < n; ++i) {
        fronts[rank[i]].push_back(i);
    }
    return fronts;
}
} // namespace Operon
