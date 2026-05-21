// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

#include <eve/module/algo.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <cstdint>
#include <numeric>

namespace Operon {
namespace {
    uint64_t constexpr ZEROS{0UL};
    uint64_t constexpr ONES{~ZEROS};
    int constexpr DIGITS{std::numeric_limits<uint64_t>::digits};

    // Packed pool budget: ~16MB allows the packed pool to cover n up to ~16000
    std::size_t constexpr POOL_BUDGET_WORDS{2UL << 20};

    struct Item {
        int Index;
        Operon::Scalar Value;
    };

    // IEEE 754 float -> sortable unsigned integer
    // Positive floats map to the upper half, negatives to the lower half, both in order.
    using SortKey = std::conditional_t<sizeof(Operon::Scalar) == 4, uint32_t, uint64_t>;
    static constexpr int RADIX_BITS = 11;
    static constexpr int RADIX_SIZE = 1 << RADIX_BITS;
    static constexpr int RADIX_MASK = RADIX_SIZE - 1;
    static constexpr int KEY_BITS = sizeof(SortKey) * 8;
    static constexpr int NUM_PASSES = (KEY_BITS + RADIX_BITS - 1) / RADIX_BITS;

    inline auto FloatToSortable(Operon::Scalar f) -> SortKey {
        SortKey bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // If sign bit is set, flip all bits; otherwise flip only sign bit
        SortKey mask = -static_cast<SortKey>(bits >> (KEY_BITS - 1)) | (SortKey{1} << (KEY_BITS - 1));
        return bits ^ mask;
    }

    // LSB radix sort for Item array, keyed on Value field.
    // scratch must be same size as items.
    void RadixSort(Item* items, Item* scratch, int n) {
        Item* src = items;
        Item* dst = scratch;

        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int const shift = pass * RADIX_BITS;

            int counts[RADIX_SIZE] = {};
            for (int i = 0; i < n; ++i) {
                auto key = FloatToSortable(src[i].Value);
                ++counts[(key >> shift) & RADIX_MASK];
            }

            int offsets[RADIX_SIZE];
            offsets[0] = 0;
            for (int i = 1; i < RADIX_SIZE; ++i) {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }

            for (int i = 0; i < n; ++i) {
                auto key = FloatToSortable(src[i].Value);
                int bucket = (key >> shift) & RADIX_MASK;
                dst[offsets[bucket]++] = src[i];
            }

            std::swap(src, dst);
        }

        // If odd number of passes, result is in scratch; copy back
        if constexpr (NUM_PASSES % 2 != 0) {
            std::copy_n(src, n, items);
        }
    }

    struct BitsetRef {
        int lo;
        int hi;
    };

    // Variable-length packed pool: individual j gets (nb - j/64) words
    struct PackedPool {
        Operon::Vector<uint64_t> data;
        Operon::Vector<int> offsets;

        auto Get(int individual) -> uint64_t* {
            return data.data() + offsets[individual];
        }
        auto Get(int individual) const -> uint64_t const* {
            return data.data() + offsets[individual];
        }

        static auto Create(int n, int nb) -> PackedPool {
            Operon::Vector<int> offs(n);
            int offset = 0;
            for (int j = 0; j < n; ++j) {
                offs[j] = offset;
                offset += nb - j / DIGITS;
            }
            return PackedPool{
                Operon::Vector<uint64_t>(static_cast<std::size_t>(offset), ZEROS),
                std::move(offs)
            };
        }

        static auto TotalWords(int n, int nb) -> std::size_t {
            std::size_t total = 0;
            for (int j = 0; j < n; ++j) {
                total += static_cast<std::size_t>(nb - j / DIGITS);
            }
            return total;
        }
    };

    // Fallback: individual heap allocations for very large n
    struct BitsetIndividual {
        Operon::Vector<std::unique_ptr<uint64_t[]>> data;

        auto Get(int individual) -> uint64_t* { return data[individual].get(); }
        auto Get(int individual) const -> uint64_t const* { return data[individual].get(); }

        void Alloc(int individual, int sz) {
            data[individual] = std::make_unique<uint64_t[]>(sz);
        }
    };

    template<typename Storage>
    auto InitBitsets(Operon::Span<Operon::Scalar const> fvals, int n, int nb,
                     Storage& store, Operon::Vector<BitsetRef>& refs)
    {
        auto const ub = static_cast<std::size_t>(DIGITS * nb) - static_cast<std::size_t>(n);

        Operon::Vector<Item> items(n);
        Operon::Vector<Item> scratch(n);
        auto const* obj1 = fvals.data() + static_cast<std::size_t>(1) * n;
        for (auto i = 0; i < n; ++i) {
            items[i] = {.Index = i, .Value = obj1[i]};
        }
        RadixSort(items.data(), scratch.data(), n);

        Operon::Vector<uint64_t> mask(nb, ONES);
        mask[nb - 1] >>= ub;

        for (auto i = 0; i < n; ++i) {
            auto const j = items[i].Index;
            auto [q, r] = std::div(j, DIGITS);
            mask[q] &= ~(1UL << static_cast<unsigned>(r));

            auto lo = 0;
            auto hi = nb - q - 1;

            if (n - 1 == i || n - 1 == j) {
                lo = hi + 1;
                refs[j] = {lo, hi};
                continue;
            }

            auto const* ptr = mask.data() + q;
            while (hi >= lo && *(ptr + hi) == ZEROS) { --hi; }

            auto sz = hi - lo + 1;
            if (sz == 0) {
                lo = hi + 1;
            } else {
                uint64_t* bits;
                if constexpr (std::is_same_v<Storage, PackedPool>) {
                    bits = store.Get(j);
                } else {
                    store.Alloc(j, sz);
                    bits = store.Get(j);
                }
                bits[0] = (ONES << static_cast<unsigned>(r)) & mask[q];
                std::copy_n(mask.data() + q + 1, sz - 1, bits + 1);
                while (lo <= hi && bits[lo] == ZEROS) { ++lo; }
                while (lo <= hi && bits[hi] == ZEROS) { --hi; }
            }
            refs[j] = {lo, hi};
        }

        return std::tuple{std::move(items), std::move(scratch), std::move(mask)};
    }

    template<typename T>
    auto MakeUnique(std::size_t n, std::optional<typename std::remove_extent_t<T>> init = std::nullopt)
    {
        ENSURE(n > 0);
        using E = typename std::remove_extent_t<T>;
        auto p = std::make_unique<E[]>(n);
        if (init) { std::fill_n(p.get(), n, init.value()); }
        return p;
    }

    template<typename Storage>
    void ObjectiveLoop(Operon::Span<Operon::Scalar const> fvals, int n, int m,
                       Storage& store, Operon::Vector<BitsetRef>& refs,
                       Operon::Vector<Item>& items, Operon::Vector<Item>& scratch,
                       Operon::Vector<uint64_t>& mask,
                       int nb, std::size_t ub)
    {
        for (auto obj = 2; obj < m; ++obj) {
            auto const* obj_vals = fvals.data() + static_cast<std::size_t>(obj) * n;
            for (auto& [i, v] : items) { v = obj_vals[i]; }
            RadixSort(items.data(), scratch.data(), n);

            std::fill_n(mask.data(), nb, ONES);
            mask[nb - 1] >>= ub;

            auto done = 0;
            auto first = items.front().Index;
            auto last = items.back().Index;

            refs[last].lo = refs[last].hi + 1;
            ++done; // last is worst on this objective; it can't dominate anyone

            mask[first / DIGITS] &= ~(1UL << static_cast<unsigned>(first % DIGITS));
            done += refs[first].lo > refs[first].hi;

            auto mmin = first / DIGITS;
            auto mmax = first / DIGITS;

            for (auto [i, _] : std::span{items.begin() + 1, items.end() - 1}) {
                auto [q, r] = std::div(i, DIGITS);
                mask[q] &= ~(1UL << static_cast<unsigned>(r));
                mmin = std::min(q, mmin);
                mmax = std::max(q, mmax);

                auto& [lo, hi] = refs[i];
                if (lo > hi) { ++done; continue; }

                auto a = std::max(mmin, lo + q);
                auto b = std::min(mmax, hi + q);
                if (b < a) { continue; }

                auto* bits = store.Get(i);
                std::span<uint64_t> pb(bits + a - q, b - a + 1);
                std::span<uint64_t const> pm(mask.data() + a, b - a + 1);
                eve::algo::transform_to(eve::views::zip(pb, pm), pb, [](auto t) {
                    return kumi::apply(std::bit_and{}, t);
                });
                while (lo <= hi && bits[lo] == ZEROS) { ++lo; }
                while (lo <= hi && bits[hi] == ZEROS) { --hi; }
            }
            if (done == n) { break; }
        }
    }

    template<typename Storage>
    void UpdateRanks(Storage const& store, Operon::Vector<BitsetRef> const& refs,
                     Operon::Vector<int>& rank,
                     Operon::Vector<std::unique_ptr<uint64_t[]>>& rankset, int n, int nb)
    {
        for (int i = 0; i < n; ++i) {
            auto const [lo, hi] = refs[i];
            if (lo > hi) { continue; }
            auto r = rank[i];
            if (static_cast<std::size_t>(r + 1) == rankset.size()) {
                rankset.push_back(MakeUnique<uint64_t[]>(nb, uint64_t{0}));
            }
            auto* curr = rankset[r].get();
            auto* next = rankset[r + 1].get();
            auto const* bits = store.Get(i);
            auto const b = i / DIGITS + lo;

            for (int k = lo, j = b; k <= hi; ++k, ++j) {
                auto x = bits[k] & curr[j];
                if (x == 0UL) { continue; }
                curr[j] &= ~x;
                next[j] |= x;
                auto const o = static_cast<std::size_t>(j) * DIGITS;
                for (; x != 0; x &= (x - 1)) {
                    ++rank[o + std::countr_zero(x)];
                }
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

auto RankIntersectSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
{
    int const n = static_cast<int>(pop.size());
    int const m = static_cast<int>(pop.front().Size());
    auto const nb = n / DIGITS + static_cast<int>(n % DIGITS != 0);
    auto const ub = static_cast<std::size_t>(DIGITS * nb) - static_cast<std::size_t>(n);

    // SoA fitness transpose: fvals[obj * n + i] = pop[i][obj]
    Operon::Vector<Operon::Scalar> fvals(static_cast<std::size_t>(n) * m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fvals[static_cast<std::size_t>(j) * n + i] = pop[i][j];
        }
    }
    Operon::Span<Operon::Scalar const> fspan(fvals);

    Operon::Vector<BitsetRef> refs(n);
    Operon::Vector<int> rank(n, 0);
    Operon::Vector<std::unique_ptr<uint64_t[]>> rs;
    rs.push_back(MakeUnique<uint64_t[]>(nb, ONES));
    rs[0][nb - 1] >>= ub;

    auto const packed_words = PackedPool::TotalWords(n, nb);

    if (packed_words <= POOL_BUDGET_WORDS) {
        auto pool = PackedPool::Create(n, nb);
        auto [items, scratch, mask] = InitBitsets(fspan, n, nb, pool, refs);
        ObjectiveLoop(fspan, n, m, pool, refs, items, scratch, mask, nb, ub);
        UpdateRanks(pool, refs, rank, rs, n, nb);
    } else {
        BitsetIndividual store;
        store.data.resize(n);
        auto [items, scratch, mask] = InitBitsets(fspan, n, nb, store, refs);
        ObjectiveLoop(fspan, n, m, store, refs, items, scratch, mask, nb, ub);
        UpdateRanks(store, refs, rank, rs, n, nb);
    }

    return GetFronts(rank);
}
} // namespace Operon
