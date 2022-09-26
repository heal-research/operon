// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include <cpp-sort/sorters/merge_sorter.h>
#include <optional>

namespace Operon {
namespace detail {
    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(std::size_t n, std::optional<typename std::remove_extent_t<T>> init = std::nullopt)
    {
        using E = typename std::remove_extent_t<T>;
        using Ptr = std::unique_ptr<T, std::add_pointer_t<void(E*)>>;
        auto ptr = Ptr(static_cast<E*>(::operator new[](n * sizeof(E), A)), [](E* ptr){ ::operator delete[](ptr, A); });
        if (init) { std::fill_n(ptr.get(), n, init.value()); }
        return ptr;
    }

    inline auto UpdateRanks(auto const& item, auto& rank, auto& rankset)
    {
        // if we're at the last objective, we can update the ranks
        auto const& [i, o, v, s] = item;
        auto r = rank[i];
        auto const n = rank.size();
        using E = std::remove_extent_t<typename decltype(s)::element_type>;
        auto constexpr D = std::numeric_limits<E>::digits;
        auto const nb = n / D + static_cast<std::size_t>(n % D != 0);
        if (r+1UL == rankset.size()) {                // new rankset if necessary
            auto p = detail::MakeUnique<uint64_t[]>(nb, E{0}); // NOLINT
            rankset.push_back(std::move(p));
        }
        auto& curr = rankset[r];                      // the pareto front of the current individual
        auto& next = rankset[r+1UL];                  // next (woranksete) pareto front

        for (std::size_t j = o; j < nb; ++j) {        // iterate over bitset blocks
            auto x = s[j] & curr[j];                  // final set as intersection of dominance set and rank set
            curr[j] &= ~x;                            // remove intersection result from current rank set
            next[j] |= x;                             // add intersection result to next rank set
            for (; x != 0; x &= (x - 1)) {            // iterate over set bits of v
                auto k = j * D + std::countr_zero(x); // get index of dominated individual
                ++rank[k];                            // increment rank
            }
        }
    }

    inline auto GetFronts(std::vector<int> const& rank)
    {
        std::vector<std::vector<std::size_t>> fronts;
        fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1UL);
        for (std::size_t i = 0UL; i < rank.size(); ++i) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }
} // namespace detail

// rank-based non-dominated sorting - intersect version - see https://arxiv.org/abs/2203.13654
auto RankIntersectSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
{
    int const n = static_cast<int>(pop.size());
    int const m = static_cast<int>(pop.front().Size());
    // constants
    std::size_t constexpr ZEROS{uint64_t{0}};
    std::size_t constexpr ONES{~ZEROS};
    std::size_t constexpr D{std::numeric_limits<uint64_t>::digits};
    std::size_t const nb{n / D + static_cast<std::size_t>(n % D != 0)};
    std::size_t const ub = D * nb - n; // number of unused bits at the end of the last block (must be set to zero) 

    using Ptr = std::unique_ptr<uint64_t[], std::add_pointer_t<void(uint64_t*)>>; // NOLINT
    using Operon::Scalar;

    auto set = [](auto&& range, int i) { range[i / D] |= (1UL << (D - i % D));}; // set bit i
    auto reset = [](auto&& range, int i) { range[i / D] &= ~(1UL << (i % D)); }; // unset bit i

    // initialization
    auto q = detail::MakeUnique<uint64_t[]>(nb, ONES); // NOLINT
    q[nb-1] >>= ub; // zero unused region

    std::vector<std::tuple<int, int, Scalar, Ptr>> items;
    items.reserve(n);

    auto b = 0; // current non-zero block
    for (auto i = 0; i < n; ++i) {
        reset(q, i); // reset bit i
        b += static_cast<int>(q[i / D] == 0); // advance lower bound when a block becomes zero
        auto p = detail::MakeUnique<uint64_t[]>(nb); // NOLINT
        std::copy_n(q.get() + b, nb - b, p.get() + b); // initialize bitset with q
        items.emplace_back(i, b, pop[i][0], std::move(p));
    }

    std::vector<Ptr> rs; // NOLINT
    rs.push_back(detail::MakeUnique<uint64_t[]>(nb, ONES)); // vector of sets keeping track of individuals whose rank was updated NOLINT
    rs[0][nb-1] >>= ub; // zero unused region

    cppsort::merge_sorter sorter;
    std::vector<int> rank(n, 0);

    for (auto obj = 1; obj < m; ++obj) {
        for (auto i = 0; i < n; ++i) { std::get<2>(items[i]) = pop[i][obj]; }
        sorter(items, [&](auto const& it) { return std::get<2>(it); });
        std::fill_n(q.get(), nb, ONES); // reset q bitset to all ones
        q[nb-1] >>= ub; // zero unused region

        for (auto& it : items) {
            auto& [i, o, v, s] = it;
            reset(q, i); // reset bit i
            if (i == std::get<0>(items.back())) {
               // last individual cannot dominate anyone else
               o = static_cast<int>(nb);
               continue;
            }
            if (o == static_cast<int>(nb)) { continue; }
            // compute the intersections of the dominance sets (most runtime intensive part)
            if (i != std::get<0>(items.front())) {
                for (std::size_t j = o; j < nb; ++j) { s[j] &= q[j]; }
                while (o < static_cast<int>(nb) && s[o] == ZEROS) { ++o; }
            }
            if (obj == m-1) { detail::UpdateRanks(it, rank, rs); }
        }
    }
    return detail::GetFronts(rank);
}
} // namespace Operon
