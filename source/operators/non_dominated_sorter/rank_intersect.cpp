// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include <iostream>
#include <new>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace Operon {
namespace detail {
    template<typename T, size_t D = std::numeric_limits<T>::digits>
    inline auto SetBit(Operon::Span<T> r, int i)
    {
        r[i / D] |= (T{1} << (D - i % D));
    }

    template<typename T, size_t D = std::numeric_limits<T>::digits>
    inline auto ResetBit(Operon::Span<T> r, int i)
    {
        r[i / D] &= ~(T{1} << (i % D));
    }

    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto AlignedNew(std::size_t count)
    {
        return static_cast<T*>(::operator new[](count * sizeof(T), A));
    }

    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto AlignedDelete(T* ptr)
    {
        ::operator delete[](ptr, A);
    }

    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(size_t n)
    {
        using E = typename std::remove_extent_t<T>;
        return std::unique_ptr<T, std::add_pointer_t<void(E*)>>(AlignedNew<E, A>(n), AlignedDelete<E, A>);
    }

    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(size_t n, typename std::remove_extent_t<T> value)
    {
        auto p = MakeUnique<T, A>(n);
        std::fill_n(p.get(), n, value);
        return p;
    }

    inline auto CountTrailingZeros(uint64_t value) {
#if defined(_MSC_VER)
        unsigned long result;
    #if defined(_M_X64)
        _BitScanForward64(&result, value);
    #else
        _BitScanForward(&result, value);
    #endif
        return result;
#else
        return __builtin_ctzl(value);
#endif
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

    using Span = Operon::Span<uint64_t>;
    std::vector<int> bs(n);

    // initialization
    auto aux = detail::MakeUnique<uint64_t[]>(nb, ONES); // NOLINT
    Operon::Span<uint64_t> q(aux.get(), nb);
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

    std::vector<std::unique_ptr<uint64_t[], std::add_pointer_t<void(uint64_t*)>>> rs; // NOLINT
    rs.push_back(detail::MakeUnique<uint64_t[]>(nb, ONES)); // vector of sets keeping track of individuals whose rank was updated NOLINT
    rs.back().get()[nb-1] >>= ub; // zero unused region
    std::vector<int> rank(n, 0); // individual ranks (initially, all zero)
    Operon::Less cmp{};

    std::vector<Operon::Scalar> values(n);

    for (auto obj = 1; obj < m; ++obj) {
        std::transform(pop.begin(), pop.end(), values.begin(), [obj](auto const& ind) { return ind[obj]; });
        std::stable_sort(indices.begin(), indices.end(), [&](auto i, auto j) { return values[i] < values[j]; });
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
                        ++rank[j * DIGITS + detail::CountTrailingZeros(v)];               // increment rank
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
