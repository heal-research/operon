// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <utility>

namespace Operon {

struct Zobrist::TranspositionTable {
    ZobristCache<FitnessEntry> Cache;
};

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes, std::size_t maxAge)
    : table_(static_cast<int>(variableHashes.size()) + 1, maxLength)
    , tt_(std::make_unique<TranspositionTable>())
    , maxAge_(maxAge)
{
    std::generate(table_.container().begin(), table_.container().end(), std::ref(rng));
    for (int i = 0; std::cmp_less(i, variableHashes.size()); ++i) {
        varIndex_[variableHashes[i]] = i;
    }
}

Zobrist::~Zobrist() = default;

auto Zobrist::TryGet(Operon::Hash hash, Value& val) const -> bool
{
    // relaxed: these are statistics counters with no ordering requirement
    // on anything else, and TryGet is in the per-individual evaluation hot
    // path - the default seq_cst RMW would tax every lookup for no benefit.
    lookups_.fetch_add(1, std::memory_order_relaxed);

    bool found = false;
    bool stale = false;
    std::uint32_t observedGen = 0;
    tt_->Cache.IfContains(hash, [&](FitnessEntry const& e) {
        observedGen = e.InsertGeneration;
        auto const now = clock_.load(std::memory_order_relaxed);
        stale = maxAge_ > 0 && static_cast<std::size_t>(now - e.InsertGeneration) > maxAge_;
        if (!stale) { val = e.Value; found = true; }
    });

    if (stale) {
        // Conditional erase: only remove if it's STILL the same stale
        // generation we observed. This is not about a concurrent Insert()
        // on the entry we just saw - Insert()'s onExisting branch is a
        // no-op, so it can't refresh InsertGeneration in place. The race
        // this guards against is: another thread's TryGet erases this same
        // stale entry first, then a concurrent Insert() recreates it fresh
        // (onNew, new InsertGeneration) before we reach EraseIf here -
        // without the generation recheck we'd delete that fresh entry
        // instead of the stale one we actually observed.
        tt_->Cache.EraseIf(hash, [&](FitnessEntry const& e) {
            return e.InsertGeneration == observedGen;
        });
        return false; // treat as a miss - caller re-evaluates
    }
    if (found) { hits_.fetch_add(1, std::memory_order_relaxed); }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Value const& val) -> void
{
    // Insert is only ever called after a TryGet miss on this same hash, so
    // the "already exists" branch only fires on a genuine race (another
    // thread inserted the same newly-seen hash first) - keep that entry's
    // value (first writer wins), nothing else to do.
    auto const gen = clock_.load(std::memory_order_relaxed);
    tt_->Cache.LazyEmplace(hash,
        [](FitnessEntry&) -> void { },
        [&](FitnessEntry& e) -> void { e.Value = val; e.InsertGeneration = gen; }
    );
}

auto Zobrist::Clear() -> void
{
    tt_->Cache.Clear();
    hits_.store(0, std::memory_order_relaxed);
    lookups_.store(0, std::memory_order_relaxed);
    // Otherwise a subsequent run's entries get stamped against a clock left
    // over from before this Clear(), immediately reading as stale (or, on
    // wraparound if the new run's generation is smaller, falsely fresh).
    clock_.store(0, std::memory_order_relaxed);
}

auto Zobrist::SetGeneration(std::size_t generation) -> void
{
    clock_.store(static_cast<std::uint32_t>(generation), std::memory_order_relaxed);
}

auto Zobrist::Size() const -> std::size_t
{
    return tt_->Cache.Size();
}

} // namespace Operon
