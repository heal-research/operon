// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/hash/zobrist.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <compare>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Operon {

namespace {
    // Innermost open SlotScope of the calling thread. Scopes of different caches (and of the same cache, nested) form a
    // chain through SlotScope::prev_, so a scope is only ever visible to its owning thread.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    thread_local Zobrist::SlotScope const* tlsSlotHead = nullptr;

    // Strict total order on fitness vectors (lexicographic, std::strong_order per component so that NaN and signed
    // zeros are ordered too): the minimum of a set does not depend on the order in which its elements are visited.
    auto FitnessLess(Operon::Vector<Operon::Scalar> const& a, Operon::Vector<Operon::Scalar> const& b) -> bool
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end(),
                   [](Operon::Scalar x, Operon::Scalar y) { return std::strong_order(x, y); })
            < 0;
    }
} // namespace

struct Zobrist::TranspositionTable {
    ZobristCache<FitnessEntry> Cache;
};

Zobrist::SlotScope::SlotScope(Zobrist const* cache)
    : owner_(cache != nullptr && cache->visibility_ == CacheVisibility::Committed ? cache : nullptr)
{
    if (owner_ != nullptr) {
        prev_ = tlsSlotHead;
        tlsSlotHead = this;
    }
}

Zobrist::SlotScope::~SlotScope()
{
    if (owner_ != nullptr) {
        EXPECT(tlsSlotHead == this); // scopes are destroyed on their own thread, innermost first
        tlsSlotHead = prev_;
    }
}

Zobrist::Zobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes,
    ZobristOptions options)
    : table_(static_cast<int>(variableHashes.size()) + 1, maxLength)
    , tt_(std::make_unique<TranspositionTable>())
    , maxAge_(options.MaxAge)
    , visibility_(options.Visibility)
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

    if (visibility_ == CacheVisibility::Committed) {
        bool found = false;
        auto const now = clock_.load(std::memory_order_relaxed);
        auto const* slot = LocalSlot();
        // Inside a slot: an entry stamped with the current clock value was inserted by some slot of this generation.
        // Which slot got there first is a race, so it is a miss for everybody. SetGeneration() handles expiry.
        // Outside any slot (code that does not run the GA loops) there is no such race to hide, and every entry is
        // visible.
        tt_->Cache.IfContains(hash, [&](FitnessEntry const& e) {
            if (slot == nullptr || e.InsertGeneration != now) {
                val = e.Value;
                found = true;
            }
        });
        if (!found && slot != nullptr) {
            if (auto const it = slot->local_.find(hash); it != slot->local_.end()) {
                val = it->second;
                found = true;
            }
        }
        if (found) {
            hits_.fetch_add(1, std::memory_order_relaxed);
        }
        return found;
    }

    bool found = false;
    bool stale = false;
    std::uint32_t observedGen = 0;
    tt_->Cache.IfContains(hash, [&](FitnessEntry const& e) {
        observedGen = e.InsertGeneration;
        auto const now = clock_.load(std::memory_order_relaxed);
        stale = maxAge_ > 0 && static_cast<std::size_t>(now - e.InsertGeneration) > maxAge_;
        if (!stale) {
            val = e.Value;
            found = true;
        }
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
        tt_->Cache.EraseIf(hash, [&](FitnessEntry const& e) { return e.InsertGeneration == observedGen; });
        return false; // treat as a miss - caller re-evaluates
    }
    if (found) {
        hits_.fetch_add(1, std::memory_order_relaxed);
    }
    return found;
}

auto Zobrist::Insert(Operon::Hash hash, Value const& val) -> void
{
    auto const gen = clock_.load(std::memory_order_relaxed);
    if (visibility_ == CacheVisibility::Committed) {
        // Insert follows a TryGet miss on this hash, so an existing entry is normally one of this generation (a
        // committed one would have hit). Several slots may insert the same structure in one generation: keep the
        // smallest fitness vector, done under the submap lock. An entry of an earlier generation is left alone: it
        // is visible to lookups, so changing it mid-generation would be order dependent.
        tt_->Cache.LazyEmplace(
            hash,
            [&](FitnessEntry& e) -> void {
                if (e.InsertGeneration == gen && FitnessLess(val, e.Value)) {
                    e.Value = val;
                }
            },
            [&](FitnessEntry& e) -> void {
                e.Value = val;
                e.InsertGeneration = gen;
            });
        // What this slot computed itself, visible to this slot only.
        if (auto const* slot = LocalSlot(); slot != nullptr) {
            slot->local_.insert_or_assign(hash, val);
        }
        return;
    }

    // Insert is only ever called after a TryGet miss on this same hash, so
    // the "already exists" branch only fires on a genuine race (another
    // thread inserted the same newly-seen hash first) - keep that entry's
    // value (first writer wins), nothing else to do.
    tt_->Cache.LazyEmplace(
        hash, [](FitnessEntry&) -> void {},
        [&](FitnessEntry& e) -> void {
            e.Value = val;
            e.InsertGeneration = gen;
        });
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
    // A run with more than 2^32-1 generations is not realistic, but a
    // silent truncation here would otherwise be a very hard-to-diagnose
    // wraparound bug in the age-based expiry.
    ENSURE(generation <= std::numeric_limits<std::uint32_t>::max());
    auto const now = static_cast<std::uint32_t>(generation);
    clock_.store(now, std::memory_order_relaxed);
    if (visibility_ == CacheVisibility::Committed && maxAge_ > 0) {
        // Same staleness rule as the lazy expiry in TryGet (Immediate): the unsigned age wraps around for an entry
        // stamped later than `now` (clock moved back), which therefore expires. Applied to the whole table at the
        // generation boundary instead of during lookups in arrival order.
        tt_->Cache.EraseWhere(
            [&](FitnessEntry const& e) { return static_cast<std::size_t>(now - e.InsertGeneration) > maxAge_; });
    }
}

auto Zobrist::AbortGeneration() -> void
{
    if (visibility_ != CacheVisibility::Committed) {
        return;
    }
    auto const now = clock_.load(std::memory_order_relaxed);
    tt_->Cache.EraseWhere([&](FitnessEntry const& e) { return e.InsertGeneration == now; });
}

auto Zobrist::TableFingerprint() const -> std::uint64_t
{
    auto const& cells = table_.container();
    std::uint64_t h {};
    if constexpr (std::endian::native == std::endian::little) {
        h = Operon::Hasher {}(reinterpret_cast<std::uint8_t const*>(cells.data()), // NOLINT
            cells.size() * sizeof(cells[0]));
    } else {
        std::vector<std::uint64_t> encoded(cells.size());
        std::ranges::transform(cells, encoded.begin(), [](auto cell) { return detail::LittleEndian(cell); });
        h = Operon::Hasher {}(reinterpret_cast<std::uint8_t const*>(encoded.data()), // NOLINT
            encoded.size() * sizeof(std::uint64_t));
    }
    // the rows are meaningful only together with the variable each of them belongs to
    std::vector<std::pair<int, Hash>> rows;
    rows.reserve(varIndex_.size());
    for (auto const& [variable, row] : varIndex_) {
        rows.emplace_back(row, variable);
    }
    std::ranges::sort(rows);
    for (auto const& [row, variable] : rows) {
        std::array<Hash, 2> const buf { detail::LittleEndian(static_cast<Hash>(row)), detail::LittleEndian(variable) };
        auto const part = Operon::Hasher {}(reinterpret_cast<std::uint8_t const*>(buf.data()), sizeof(buf)); // NOLINT
        h ^= part + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    }
    return h;
}

auto Zobrist::Export() const -> ZobristState
{
    ZobristState state;
    state.TableFingerprint = TableFingerprint();
    state.Clock = clock_.load(std::memory_order_relaxed);
    state.MaxAge = maxAge_;
    state.Visibility = visibility_;
    state.Entries.reserve(tt_->Cache.Size());
    tt_->Cache.ForEach([&](Hash key, FitnessEntry const& e) {
        state.Entries.push_back({ .Key = key, .InsertGeneration = e.InsertGeneration, .Value = e.Value });
    });
    std::ranges::sort(state.Entries, [](auto const& a, auto const& b) { return a.Key < b.Key; });
    return state;
}

auto Zobrist::Import(ZobristState const& state, std::size_t objectiveCount) -> void
{
    if (state.TableFingerprint != TableFingerprint()) {
        throw std::invalid_argument("the cache state was built with a different random table or an incompatible "
                                    "representation: use the same seed, dataset and maximum length on the same kind "
                                    "of machine");
    }
    if (state.Visibility != visibility_) {
        throw std::invalid_argument("the cache state was written with a different cache visibility");
    }
    if (state.MaxAge != maxAge_) {
        throw std::invalid_argument("the cache state was written with a different maximum entry age");
    }
    if (objectiveCount == 0) {
        throw std::invalid_argument("the objective count of a cache import must be at least 1");
    }
    for (auto const& record : state.Entries) {
        if (record.Value.size() != objectiveCount) {
            throw std::invalid_argument("a cache entry has " + std::to_string(record.Value.size())
                + " fitness values but the evaluator has " + std::to_string(objectiveCount) + " objectives");
        }
        if (record.InsertGeneration > state.Clock) {
            throw std::invalid_argument("a cache entry was inserted in generation "
                + std::to_string(record.InsertGeneration) + ", after the generation clock of the cache state ("
                + std::to_string(state.Clock) + ")");
        }
    }
    // Built aside, exchanged in at the end: nothing of this cache changes unless the whole state was accepted.
    ZobristCache<FitnessEntry> staged;
    for (auto const& record : state.Entries) {
        staged.LazyEmplace(
            record.Key, [](FitnessEntry&) -> void {},
            [&](FitnessEntry& e) -> void {
                e.Value = record.Value;
                e.InsertGeneration = record.InsertGeneration;
            });
    }
    tt_->Cache.Swap(staged);
    clock_.store(state.Clock, std::memory_order_relaxed);
}

auto Zobrist::BeginSlot() const -> SlotScope { return SlotScope { this }; }

auto Zobrist::LocalSlot() const -> SlotScope const*
{
    for (auto const* slot = tlsSlotHead; slot != nullptr; slot = slot->prev_) {
        if (slot->owner_ == this) {
            return slot;
        }
    }
    return nullptr;
}

auto Zobrist::Size() const -> std::size_t { return tt_->Cache.Size(); }

} // namespace Operon
