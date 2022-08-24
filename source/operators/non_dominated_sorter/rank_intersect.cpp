// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include <iostream>
#include <eve/wide.hpp>
#include <cpp-sort/sorters/merge_sorter.h>

namespace Operon {
namespace detail {
    template<std::ranges::random_access_range R, typename T = typename std::remove_cvref_t<R>::value_type, size_t D = std::numeric_limits<T>::digits>
    requires std::is_integral_v<T>
    inline auto SetBit(R&& r, int i)
    {
        r[i / D] |= (T{1} << (D - i % D));
    }

    template<std::ranges::random_access_range R, typename T = typename std::remove_cvref_t<R>::value_type, size_t D = std::numeric_limits<T>::digits>
    requires std::is_integral_v<T>
    inline auto ResetBit(R&& r, int i)
    {
        r[i / D] &= ~(T{1} << (i % D));
    }

    template<std::ranges::random_access_range R, typename T = typename std::remove_cvref_t<R>::value_type, size_t D = std::numeric_limits<T>::digits>
    inline auto Print(R&& r)
    {
        for (auto v : r) { std::cout << std::bitset<D>(v) << " "; }
        std::cout << "\n";
    }

    template<typename T, typename X = std::remove_extent_t<T>, std::align_val_t ALIGNMENT = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(size_t n)
    {
        return std::unique_ptr<T>(new (ALIGNMENT) X[n]);
    }

    template<typename T, typename X = std::remove_extent_t<T>, std::align_val_t ALIGNMENT = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(size_t n, X value)
    {
        auto p = std::unique_ptr<T>(new (ALIGNMENT) X[n]);
        std::fill_n(p.get(), n, value);
        return p;
    }
} // namespace detail

// rank-based non-dominated sorting - intersect version - see https://arxiv.org/abs/2203.13654
auto RankIntersectSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
{
    int const n = static_cast<int>(pop.size());
    int const m = static_cast<int>(pop.front().Size());

    // bitset block type
    auto constexpr ONES = ~uint64_t{0};
    auto constexpr ZEROS = uint64_t{0};

    // constants
    size_t constexpr DIGITS{std::numeric_limits<uint64_t>::digits};
    size_t const nb{n / DIGITS + static_cast<size_t>(n % DIGITS != 0)};
    size_t const ub = DIGITS * nb - n; // unused bit region (must be zeroed)

    using Span = std::span<uint64_t>;
    std::vector<int> bs(n);

    // initialization
    auto aux = detail::MakeUnique<uint64_t[]>(nb, ONES); // NOLINT
    std::span<uint64_t> q(aux.get(), nb);
    q[nb-1] >>= ub; // zero unused region

    auto bits = detail::MakeUnique<uint64_t[]>(nb * n); // NOLINT
    std::vector<int> indices(n);

    auto b = 0; // current non-zero block
    for (auto i = 0; i < n; ++i) {
        indices[i] = i;
        detail::ResetBit(q, i); // reset bit i
        b += static_cast<int>(q[i / DIGITS] == 0); // advance lower bound when a block becomes zero
        Span s{bits.get() + i * nb, nb};
        std::copy(q.begin() + b, q.end(), s.begin() + b); // initialize bitset with q
        bs[i] = b;
    }

    std::vector<std::unique_ptr<uint64_t[]>> rs; // NOLINT
    rs.push_back(detail::MakeUnique<uint64_t[]>(nb, ONES)); // vector of sets keeping track of individuals whose rank was updated NOLINT
    rs.back().get()[nb-1] >>= ub; // zero unused region
    std::vector<int> rank(n, 0); // individual ranks (initially, all zero)
    Operon::Less cmp{};

    std::vector<Operon::Scalar> values(n);
    cppsort::merge_sorter sorter;

    for (auto obj = 1; obj < m; ++obj) {
        std::transform(pop.begin(), pop.end(), values.begin(), [obj](auto const& ind) { return ind[obj]; });
        sorter(indices, [&](auto i) { return values[i]; });
        std::fill(q.begin(), q.end(), ONES); // reset q bitset to all ones
        q[nb-1] >>= ub; // zero unused region

        for (auto i : indices) {
            detail::ResetBit(q, i); // reset bit i

            if (i == indices.back()) {
                // last individual cannot dominate anyone else
                bs[i] = static_cast<int>(nb);
                continue;
            }
            auto b = bs[i];
            if (b == nb) { continue; }

            // compute the intersections of the dominance sets (most runtime intensive part)
            Span s{bits.get() + i * nb, nb};
            if (i != indices.front()) {
                auto* ss = s.data();
                auto* qq = q.data();

                for (auto j = b; j < static_cast<int>(nb); ++j) {
                    s[j] &= q[j];
                    b += !s[j]; // update lower bound in the same loop NOLINT
                }
                bs[i] = b;
            }

            if (obj == m-1) {
                // if we're at the last objective, we can update the ranks
                if (rank[i] + 1UL == rs.size()) {                         // new rankset if necessary
                    rs.push_back(detail::MakeUnique<uint64_t[]>(nb, ZEROS)); // NOLINT
                }
                auto* curr = (rs.begin() + rank[i])->get();               // the pareto front of the current individual
                auto* next = (rs.begin() + rank[i] + 1)->get();           // next (worse) pareto front

                for (auto j = b; j != static_cast<int>(nb); ++j) {              // iterate over bitset blocks
                    auto v = s[j] & curr[j];                                    // final set as intersection of dominance set and rank set
                    curr[j] &= ~v;                                              // remove intersection result from current rank set
                    next[j] |= v;                                               // add intersection result to next rank set
                    for (; v != 0; v &= (v - 1)) {                              // iterate over set bits of v
                        ++rank[j * DIGITS + std::countr_zero(v)];               // increment rank
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
