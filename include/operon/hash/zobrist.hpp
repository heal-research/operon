// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_ZOBRIST_HPP
#define OPERON_HASH_ZOBRIST_HPP

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtl/phmap.hpp>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

namespace detail {
    // Hash inputs are fed to the hasher as little-endian integers, so that keys and the random-table fingerprint of a
    // cache (and with them a checkpoint that carries it) do not depend on the byte order of the machine. A no-op on
    // little-endian machines.
    [[nodiscard]] constexpr auto LittleEndian(std::uint64_t value) noexcept -> std::uint64_t
    {
        if constexpr (std::endian::native == std::endian::big) {
            return std::byteswap(value);
        } else {
            return value;
        }
    }
} // namespace detail

// ── Cache primitives ─────────────────────────────────────────────────────────

// Variadic mixin: assembles a cache entry from multiple data components.
// Example:  using FitnessEntry = CacheEntry<FitnessData>;
//           using JitEntry     = CacheEntry<VisitData, MetaData>;
template <typename... Data> struct CacheEntry : Data... {};

// Fitness value produced by the evaluator after local search.
struct FitnessData {
    Vector<Scalar> Value;
    std::uint32_t InsertGeneration { 0 };
};

using FitnessEntry = CacheEntry<FitnessData>;

// When an entry inserted into the transposition cache becomes visible to lookups.
//
// Immediate: at once, to every slot (first writer wins; lazy age-based expiry inside TryGet). Concurrent slots can
// observe each other's entries in arrival order, so results may differ between thread counts and between runs.
//
// Committed: inside a SlotScope (opened by the GA loops around every offspring slot), an entry is visible to other
// slots only from the generation after the one that inserted it; a slot additionally sees what it inserted itself
// (see Zobrist::BeginSlot). A lookup made with no scope open (code outside the GA loops) sees every entry.
// Duplicate inserts in one generation keep the smallest fitness vector, and age-based expiry runs at the generation
// boundary. Hit/miss decisions then do not depend on the order in which slots run.
//
// That makes a run independent of the thread count only if no stop condition cuts a generation: evaluation budget,
// offspring-selection pressure (the `os` generator), time limit, stop request. How far a cut generation got depends
// on the thread count. JIT evaluation is not covered: JIT results differ between runs even at one thread,
// independent of the cache, and a JitZobrist visit threshold above 1 makes compilation depend on arrival order.
//
// Cost: lookups that used to hit an entry of the same generation become evaluations. The search trajectory then
// changes, so the number of evaluations a run consumes changes too, in either direction; measure it with the
// configuration at hand.
enum class CacheVisibility : std::uint8_t { Immediate, Committed };

struct ZobristOptions {
    // 0 disables age-based expiry (unbounded). Otherwise an entry older than MaxAge generations is expired, a
    // pragmatic freshness/evolvability mitigation rather than a memory bound.
    std::size_t MaxAge { 0 };
    CacheVisibility Visibility { CacheVisibility::Committed };
};

// One cache entry in a ZobristState.
struct ZobristEntryRecord {
    Hash Key { 0 };
    std::uint32_t InsertGeneration { 0 };
    Vector<Scalar> Value;
};

// The content of a Zobrist cache that a checkpoint carries: every entry with its insertion stamp and the generation
// clock, plus what has to agree between the run that wrote it and the run that reads it (see Zobrist::Import).
struct ZobristState {
    std::uint64_t TableFingerprint { 0 }; // identifies the random table the keys were computed with
    std::uint32_t Clock { 0 };
    std::size_t MaxAge { 0 };
    CacheVisibility Visibility { CacheVisibility::Committed };
    std::vector<ZobristEntryRecord> Entries; // ordered by Key, so the encoding does not depend on the table's layout
};

// Thread-safe cache keyed by Operon::Hash, backed by gtl::parallel_flat_hash_map_m.
template <typename Entry> class ZobristCache {
    gtl::parallel_flat_hash_map_m<Hash, Entry> map_;

public:
    template <typename Fn> auto IfContains(Hash h, Fn&& fn) const -> bool
    {
        bool found = false;
        map_.if_contains(h, [&](auto const& kv) {
            std::forward<Fn>(fn)(kv.second);
            found = true;
        });
        return found;
    }

    template <typename Fn> auto ModifyIf(Hash h, Fn&& fn) -> bool
    {
        return map_.modify_if(h, [&](auto& kv) { std::forward<Fn>(fn)(kv.second); });
    }

    template <typename OnExisting, typename OnNew>
    auto LazyEmplace(Hash h, OnExisting&& onExisting, OnNew&& onNew) -> void
    {
        map_.lazy_emplace_l(
            h, [&](auto& kv) { std::forward<OnExisting>(onExisting)(kv.second); },
            [&](auto const& ctor) {
                Entry e {};
                std::forward<OnNew>(onNew)(e);
                ctor(h, std::move(e));
            });
    }

    template <typename Pred> auto EraseIf(Hash h, Pred&& pred) -> std::size_t
    {
        return map_.erase_if(h, [&](auto const& kv) { return pred(kv.second); });
    }

    // Erases every entry matching `pred` (called with the entry), one submap at a time under that submap's lock.
    // Returns the number of erased entries.
    template <typename Pred> auto EraseWhere(Pred&& pred) -> std::size_t
    {
        std::size_t erased = 0;
        for (std::size_t i = 0; i < map_.subcnt(); ++i) {
            map_.with_submap_m(i, [&](auto& submap) {
                for (auto it = submap.begin(); it != submap.end();) {
                    if (pred(it->second)) {
                        it = submap.erase(it);
                        ++erased;
                    } else {
                        ++it;
                    }
                }
            });
        }
        return erased;
    }

    // Calls fn(key, entry) for every entry, one submap at a time under that submap's lock. fn must not call back
    // into the cache.
    template <typename Fn> auto ForEach(Fn&& fn) -> void
    {
        for (std::size_t i = 0; i < map_.subcnt(); ++i) {
            map_.with_submap_m(i, [&](auto& submap) {
                for (auto const& kv : submap) {
                    fn(kv.first, kv.second);
                }
            });
        }
    }

    [[nodiscard]] auto Size() const -> std::size_t { return map_.size(); }
    auto Clear() -> void { map_.clear(); }
    // Exchanges the content with `other`; does not allocate and does not throw.
    auto Swap(ZobristCache& other) noexcept -> void { map_.swap(other.map_); }
};

// ─────────────────────────────────────────────────────────────────────────────

// Position-aware Zobrist hash for GP trees, plus a transposition table that
// caches fitness vectors keyed by structural hash.  The hash captures tree
// topology, variable identity, and the Optimize flag of each node, but not
// coefficient values — by design, so that a cached fitness (computed after
// local search) can be reused for any structurally identical tree with the
// same set of trainable parameters, regardless of current coefficient values.
//
// The Optimize flag is included because it determines the number and layout
// of trainable coefficients; two trees that differ only in which nodes are
// optimizable are functionally distinct (different coefficient counts) and
// must not share a cache entry.  IsEnabled is NOT hashed because disabled
// nodes are erased by Tree::Reduce() before they reach ComputeHash.
//
// Ownership: the caller constructs one Zobrist per experiment (or per run) and
// passes a raw pointer into GeneticAlgorithmConfig::Cache.  The algorithm
// borrows the pointer; the caller is responsible for lifetime.
//
// Sharing: TryGet, Insert and the slot scopes are thread-safe, and slot state is per scope, so a cache shared by
// concurrent runs (or by several caches' runs on one thread) is memory-safe. The generation clock is per cache,
// though: concurrent runs each advance it, which makes entry visibility and age-based expiry meaningless for both.
// Use one cache per run when results must be reproducible; Clear() between sequential runs.
//
// Subclassing: JitZobrist (in jit_evaluator.hpp) is the only intended subclass.
// The virtual destructor exists to support nanobind exposure of both types.
class OPERON_EXPORT Zobrist {
    using Value = Vector<Scalar>;
    using Extents = std::extents<int, std::dynamic_extent, std::dynamic_extent>;
    using Table = MDArray<Hash, Extents>;

    // table_ holds one precomputed random row per variable (known and finite
    // upfront, from variableHashes) plus one row for the Optimize marker.
    // Non-variable node identity (built-in or user-registered function hash)
    // is NOT table-indexed: that identity space is open-ended by design (any
    // number of functions can be registered, at any time, not necessarily
    // before a Zobrist is constructed), so there is no safe upper bound to
    // size a table row-count from. Instead it's combined directly from the
    // node's HashValue - see ComputeHash.
    Table table_;
    Map<Hash, int> varIndex_;

    struct TranspositionTable;
    std::unique_ptr<TranspositionTable> tt_;

    mutable std::atomic<std::size_t> hits_ { 0 };
    mutable std::atomic<std::size_t> lookups_ { 0 };

    // Generation clock set by the GA loop (see gp.cpp/nsga2.cpp) once per
    // generation. Entries are stamped with it on Insert; it drives age-based
    // expiry (lazily in TryGet for Immediate, in SetGeneration for Committed)
    // and the Committed visibility rule. Not related to hits_/lookups_.
    mutable std::atomic<std::uint32_t> clock_ { 0 };

    // 0 disables age-based expiry (default, today's unbounded behavior).
    // Otherwise, an entry older than maxAge_ generations is expired - see
    // ZobristOptions::MaxAge for why this is a pragmatic freshness/evolvability
    // mitigation, not a memory bound.
    std::size_t maxAge_ { 0 };
    CacheVisibility visibility_ { CacheVisibility::Committed };

public:
    // Slot-local fitness values of one offspring slot (Committed mode); see BeginSlot. Owned by the thread that
    // created it and reachable only through a thread-local chain, so no synchronisation is needed.
    class OPERON_EXPORT SlotScope {
        friend class Zobrist;

        Zobrist const* owner_ { nullptr }; // null: inert
        SlotScope const* prev_ { nullptr };
        // Mutable so that a scope bound as `const` (the natural way to hold an RAII guard) can still record inserts:
        // the thread-local chain reaches the scope through a pointer to const, and writing a non-mutable member of
        // an object that was defined const would be undefined behaviour.
        mutable Map<Hash, Vector<Scalar>> local_;

    public:
        // Opens a slot on `cache` for the calling thread; inert when `cache` is null or in Immediate mode.
        explicit SlotScope(Zobrist const* cache);
        ~SlotScope();
        SlotScope(SlotScope const&) = delete;
        SlotScope(SlotScope&&) = delete;
        auto operator=(SlotScope const&) -> SlotScope& = delete;
        auto operator=(SlotScope&&) -> SlotScope& = delete;
    };

private:
    // Innermost SlotScope of the calling thread that belongs to this cache, or null.
    [[nodiscard]] auto LocalSlot() const -> SlotScope const*;

public:
    // variableHashes must include every variable hash that can appear in a tree.
    // Each variable gets its own row of independent random values so that
    // permuting variables at different positions always yields a different hash.
    Zobrist(RandomGenerator& rng, int maxLength, Span<Hash const> variableHashes, ZobristOptions options = {});
    virtual ~Zobrist();
    Zobrist(Zobrist const&) = delete;
    Zobrist(Zobrist&&) = delete;
    auto operator=(Zobrist const&) -> Zobrist& = delete;
    auto operator=(Zobrist&&) -> Zobrist& = delete;

    [[nodiscard]] auto Rows() const { return table_.extent(0); }
    [[nodiscard]] auto Cols() const { return table_.extent(1); }
    [[nodiscard]] auto OptimizeRow() const { return static_cast<int>(Rows()) - 1; }

    [[nodiscard]] auto ComputeHash(Node const& n, int pos) const -> Hash
    {
        Hash h {};
        if (n.IsVariable()) {
            auto const it = varIndex_.find(n.HashValue);
            ENSURE(it != varIndex_.end());
            h = table_[it->second, pos];
        } else {
            // Combine the node's own identity hash with its tree position,
            // rather than a table lookup - see the comment on table_ above
            // for why. Ref nodes additionally encode their target: two DAGs
            // with the same node sequence but different RefTo values compute
            // different functions and must never share a cache entry.
            std::array<Hash, 2> const buf { detail::LittleEndian(n.HashValue),
                detail::LittleEndian(static_cast<Hash>(pos)) };
            h = Operon::Hasher {}(reinterpret_cast<uint8_t const*>(buf.data()),
                sizeof(buf)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            if (n.IsRef()) {
                auto const target = static_cast<Hash>(n.RefTo);
                h ^= target + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
            }
        }
        if (n.Optimize) {
            h ^= table_[OptimizeRow(), pos];
        }
        return h;
    }

    [[nodiscard]] auto ComputeHash(Tree const& tree) const -> Hash
    {
        EXPECT(std::ssize(tree.Nodes()) <= Cols());
        Hash h {};
        auto const& nodes = tree.Nodes();
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            h ^= ComputeHash(nodes[i], i);
        }
        return h;
    }

    // Returns true and fills `val` if the hash is found; thread-safe.
    //
    // Immediate: any entry in the table is visible at once, whichever slot or generation inserted it.
    // Committed: with a SlotScope of this cache open on the calling thread, the shared table is consulted for entries
    // of earlier generations only (an entry stamped with the current clock value is a miss, see SetGeneration), then
    // the slot-local map of the innermost such scope. The outcome therefore depends on neither the thread count nor
    // the order in which concurrent slots run. With no scope open, every entry of the shared table is visible.
    [[nodiscard]] auto TryGet(Hash hash, Value& val) const -> bool;

    // Inserts a newly-computed value for `hash`; thread-safe.
    //
    // Immediate: a concurrent race inserting the same hash first is not an error - the existing entry's value is
    // kept (first writer wins).
    // Committed: the entry is stamped with the current clock value and stays invisible to shared lookups until the
    // clock advances. Concurrent inserts of the same hash within one generation keep the lexicographically smallest
    // value (std::strong_order per component, so NaN and signed zeros are ordered too), which does not depend on
    // arrival order. An entry stamped with an earlier generation is never overwritten. The value is also recorded in
    // the calling thread's innermost SlotScope on this cache, if any.
    auto Insert(Hash hash, Value const& val) -> void;

    // Clears the transposition table and resets the hit counter and the generation clock.
    // NOT safe to call concurrently with TryGet or Insert — call only after
    // the algorithm has fully stopped (e.g. after GeneticAlgorithm::Run returns).
    auto Clear() -> void;

    // Sets the generation clock that entries are stamped with by Insert. The GA loop (gp.cpp/nsga2.cpp) sets it to
    // Generation() + 1 before the offspring of a generation are generated, so the entries of generation g become
    // visible to scoped lookups from the next call with a different value on. A fresh or cleared cache has clock 0:
    // entries inserted before the first call are stamped 0 and, in Committed mode, are hidden from scoped shared
    // lookups until the clock has moved past 0 (the slot-local map still serves the inserting scope). The initial
    // population (generation 0) is scored without the cache.
    //
    // Committed with MaxAge > 0: entries older than MaxAge generations (clock - InsertGeneration > MaxAge) are erased
    // here, at the generation boundary, so expiry cannot depend on arrival order. Entries stamped later than the new
    // clock value (the clock moved back, e.g. a second Run() on the same cache) are expired too, as in Immediate,
    // where the unsigned age wraps around. Immediate expires lazily in TryGet.
    // Takes the submap locks, so a concurrent TryGet/Insert is not a data race, but whether it sees an entry the
    // sweep erases is then arbitrary: the GA loop calls it from a single task between generations.
    auto SetGeneration(std::size_t generation) -> void;

    // Rolls back the generation being generated: erases every entry stamped with the current clock value, so the
    // work of a discarded generation (see GeneticAlgorithmBase::IncompleteGeneration) cannot become visible once the
    // clock advances, or take part in the minimum of a retry's duplicate inserts. Committed mode only; in Immediate
    // mode entries are visible at once and are kept, and this is a no-op. Expiry done by SetGeneration, the hit and
    // lookup counters and consumed worker RNG state are not rolled back. Takes the submap locks: the GA loops call it
    // from a single task after the offspring workers have finished.
    auto AbortGeneration() -> void;

    // Opens a slot: a thread-confined scope in which the calling thread's own Insert calls are visible to its own
    // TryGet calls even though scoped shared lookups ignore the current generation (Committed mode). The algorithm
    // opens one around each offspring slot, so a slot reuses what it computed itself in an earlier attempt of the
    // same slot. The scope must be destroyed on the thread that created it, in reverse order of creation. State
    // lives in the scope and is tied to this cache instance: other caches and concurrent runs sharing this cache
    // never see each other's scopes. Without a scope (or in Immediate mode, where the scope is inert), nothing is
    // slot-local and, in Committed mode, a lookup sees every entry of the shared table.
    [[nodiscard]] auto BeginSlot() const -> SlotScope;

    [[nodiscard]] auto Visibility() const -> CacheVisibility { return visibility_; }
    [[nodiscard]] auto MaxAge() const -> std::size_t { return maxAge_; }

    // The entries and the generation clock, for a checkpoint. A cache restored from it with Import() answers every
    // lookup exactly as this one does now (visibility and age-based expiry included) and, with the same random
    // numbers in the offspring generation, makes the same decisions from then on. Hit/lookup counters are not part of
    // it. Call it only while no run is using the cache (e.g. from the report callback, between generations).
    [[nodiscard]] auto Export() const -> ZobristState;

    // Replaces the content of this cache (entries and generation clock) by `state`. Throws std::invalid_argument, and
    // leaves the cache unchanged, if the state cannot be continued by this cache: a different random table (another
    // seed, maximum length or variable set, so the keys mean something else, or a representation that hashes
    // differently), a different MaxAge or visibility, an entry whose fitness vector does not have exactly
    // `objectiveCount` (>= 1) values (zero-length entries are always rejected), or an entry inserted after the
    // state's own clock (InsertGeneration > Clock: no run can have written it). The new content is built aside and
    // exchanged in only when it is complete, so any failure, including an allocation failure, leaves the cache as it
    // was. The table fingerprint says nothing about what the fitness values mean (dataset, objective, optimizer
    // settings): that is for the caller to establish before importing (the command-line programs compare a run
    // fingerprint stored in the checkpoint). Counters are left alone. Same concurrency rule as Clear().
    auto Import(ZobristState const& state, std::size_t objectiveCount) -> void;

    // Identifies the random table (and the variable-to-row mapping) the keys of this cache are computed with.
    [[nodiscard]] auto TableFingerprint() const -> std::uint64_t;

    // Zeroes Hits()/Lookups() only; the table's entries and the generation clock are left alone.
    auto ResetCounters() -> void
    {
        hits_.store(0, std::memory_order_relaxed);
        lookups_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] auto Hits() const -> std::size_t { return hits_.load(std::memory_order_relaxed); }
    // Total TryGet() calls regardless of outcome - the denominator Hits()
    // needs to express an actual hit *rate* rather than a raw count.
    [[nodiscard]] auto Lookups() const -> std::size_t { return lookups_.load(std::memory_order_relaxed); }
    [[nodiscard]] auto Size() const -> std::size_t;
};

} // namespace Operon

#endif
