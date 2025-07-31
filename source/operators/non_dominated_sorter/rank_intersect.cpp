// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cpp-sort/sorters/merge_sorter.h>

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include <eve/module/algo.hpp>
#include <fmt/core.h>

namespace Operon {
namespace {
    std::size_t constexpr ZEROS{uint64_t{0}};
    std::size_t constexpr ONES{~ZEROS};
    std::size_t constexpr DIGITS{std::numeric_limits<uint64_t>::digits};

    using Bitset = std::unique_ptr<uint64_t[]>;

    cppsort::merge_sorter const Sorter;

    struct Item {
        int Index;
        Operon::Scalar Value;

        friend auto operator<(Item a, Item b) { return a.Value < b.Value; }
    };

    template<typename T>
    auto MakeUnique(std::size_t n, std::optional<typename std::remove_extent_t<T>> init = std::nullopt)
    {
        ENSURE(n > 0);
        using E = typename std::remove_extent_t<T>;
        auto ptr = std::make_unique<E[]>(n);
        if (init) { std::fill_n(ptr.get(), n, init.value()); }
        return ptr;
    }

    auto InitBitsets(Operon::Span<Operon::Individual const> pop)
    {
        auto const n = static_cast<int>(std::ssize(pop));
        auto const nb { static_cast<int>(n / DIGITS) + static_cast<int>(n % DIGITS != 0) };
        ENSURE(nb > 0);
        std::size_t const ub = (DIGITS * nb) - n;

        Operon::Vector<Item> items(n);
        for (auto i = 0; i < n; ++i) {
            items[i] = { .Index=i, .Value=pop[i][1] };
        }
        Sorter(items);

        auto mask = MakeUnique<uint64_t[]>(nb, ONES); // NOLINT
        mask[nb-1] >>= ub;

        Operon::Vector<std::tuple<Bitset, int, int>> bitsets(n);
        for (auto i = 0; i < n; ++i) {
            auto const j = items[i].Index;
            auto [q, r] = std::div(j, DIGITS);
            mask[q] &= ~(1UL << static_cast<uint>(r)); // reset bit j

            auto lo = 0;
            auto hi = nb-q-1;
            Bitset p;
            if (n-1 == i || n-1 == j) {
                lo = hi+1;
                bitsets[j] = { std::move(p), lo, hi };
                continue;
            }
            auto* ptr = mask.get() + q;
            while(hi >= lo && *(ptr + hi) == ZEROS) { --hi; }

            auto sz = hi-lo+1;
            if (sz == 0) { lo = hi+1; } else {
                p = MakeUnique<uint64_t[]>(sz);
                p[0] = (ONES << static_cast<uint>(r)) & mask[q];
                std::copy_n(mask.get()+q+1, sz-1, p.get()+1);
                while (lo <= hi && (p[lo] == ZEROS)) { ++lo; }
                while (lo <= hi && (p[hi] == ZEROS)) { --hi; }
            }
            bitsets[j] = { std::move(p), lo, hi };
        }

        return std::tuple{std::move(bitsets), std::move(items), std::move(mask)};
    }

    auto UpdateRanks(auto i, auto const& item, auto& rank, auto& rankset)
    {
        // if we're at the last objective, we can update the ranks
        auto const& [s, lo, hi] = item;
        auto r = rank[i];
        auto const n = std::ssize(rank);
        using E = std::remove_extent_t<typename decltype(s)::element_type>;
        auto constexpr D = std::numeric_limits<E>::digits;
        auto const nb = n / D + static_cast<std::size_t>(n % D != 0);
        if (r+1UL == rankset.size()) {                 // new rankset if necessary
            auto p = MakeUnique<uint64_t[]>(nb, E{0}); // NOLINT
            rankset.push_back(std::move(p));
        }
        auto& curr = rankset[r];                                           // the pareto front of the current individual
        auto& next = rankset[r+1UL];                                       // next (worse rank) pareto front

        auto b = static_cast<int>(i / D) + lo;

        for (auto j = b, k = lo; k <= hi; ++k, ++j) {                      // iterate over bitset blocks
            auto x = s[k] & curr[j];                                       // final set as intersection of dominance set and rank set
            if (x == 0UL) { continue; }
            auto o = j * D;
            curr[j] &= ~x;                                                 // remove intersection result from current rank set
            next[j] |= x;                                                  // add intersection result to next rank set
            for (; x != 0; x &= (x - 1)) {                                 // iterate over set bits of v
                ++rank[static_cast<std::size_t>(o) + std::countr_zero(x)]; // increment rank
            }
        }
    }

    auto GetFronts(Operon::Vector<int> const& rank)
    {
        Operon::Vector<Operon::Vector<std::size_t>> fronts;
        auto rmax = *std::max_element(rank.begin(), rank.end());
        fronts.resize(rmax + 1UL);
        for (std::size_t i = 0UL; i < rank.size(); ++i) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }
} // namespace

// rank-based non-dominated sorting - intersect version - see https://arxiv.org/abs/2203.13654
auto RankIntersectSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
{
    int const n = static_cast<int>(pop.size());
    int const m = static_cast<int>(pop.front().Size());

    // constants
    auto const nb { static_cast<int>(n / DIGITS) + static_cast<int>(n % DIGITS != 0) };
    std::size_t const ub = DIGITS * nb - n; // number of unused bits at the end of the last block (must be set to zero)

    Operon::Vector<Bitset> rs;
    rs.push_back(MakeUnique<uint64_t[]>(nb, ONES)); // vector of sets keeping track of individuals whose rank was updated NOLINT
    rs[0][nb-1] >>= ub; // zero unused region

    auto [bitsets, items, mask] = InitBitsets(pop);

    for (auto obj = 2; obj < m; ++obj) {
        for (auto& [i, v] : items) { v = pop[i][obj]; }
        Sorter(items);

        std::fill_n(mask.get(), nb, ONES); // reset q bitset to all ones
        mask[nb-1] >>= ub;                 // zero unused region

        auto done = 0;
        auto first = items.front().Index;
        auto last = items.back().Index;

        std::get<1>(bitsets[last]) = std::get<2>(bitsets[last])+1; // lo = hi+1
        mask[first / DIGITS] &= ~(1UL << static_cast<uint>(first % DIGITS));

        auto mmin = static_cast<int>(first / DIGITS);
        auto mmax = static_cast<int>(first / DIGITS);

        // [items.begin()+1, items.end()-1] must be a valid range
        for (auto [i, _] : std::span{items.begin()+1, items.end()-1}) {
            auto [q, r] = std::div(i, DIGITS);
            mask[q] &= ~(1UL << static_cast<uint>(r)); // reset bit i
            auto& [bits, lo, hi] = bitsets[i];
            if (lo > hi) { ++done; continue; }

            mmin = std::min(q, mmin);
            mmax = std::max(q, mmax);
            auto a = std::max(mmin, lo + q);
            auto b = std::min(mmax, hi + q);
            if (b < a) { continue; }

            std::span<uint64_t> pb(bits.get() + a-q, b-a+1);
            std::span<uint64_t const> pm(mask.get() + a, b-a+1);
            // eve::algo::transform_to(eve::views::zip(pb, pm), pb, [](auto t) { return kumi::apply(std::bit_and{}, t); });
            std::ranges::transform(pb, pm, std::begin(pb), std::bit_and{});
            while (lo <= hi && (bits[lo] == ZEROS)) { ++lo; }
            while (lo <= hi && (bits[hi] == ZEROS)) { --hi; }
        }
        if (done == n) { break; }
    }

    Operon::Vector<int> rank(n, 0);
    for (auto i = 0; i < n; ++i) {
        UpdateRanks(i, bitsets[i], rank, rs);
    }
    return GetFronts(rank);
}
} // namespace Operon
