// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_ZOBRIST_HPP
#define OPERON_HASH_ZOBRIST_HPP

#include <atomic>
#include <memory>

#include <gtl/phmap.hpp>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// ── Cache primitives ─────────────────────────────────────────────────────────

// Variadic mixin: assembles a cache entry from multiple data components.
// Example:  using FitnessEntry = CacheEntry<FitnessData>;
//           using JitEntry     = CacheEntry<VisitData, MetaData>;
template<typename... Data>
struct CacheEntry : Data... {};

// Fitness value produced by the evaluator after local search.
struct FitnessData {
    Vector<Scalar>  Value;
    std::size_t     Visits{1};
};

using FitnessEntry = CacheEntry<FitnessData>;

// Thread-safe cache keyed by Operon::Hash, backed by gtl::parallel_flat_hash_map_m.
template<typename Entry>
class ZobristCache {
    gtl::parallel_flat_hash_map_m<Hash, Entry> map_;

public:
    template<typename Fn>
    auto IfContains(Hash h, Fn&& fn) const -> bool {
        bool found = false;
        map_.if_contains(h, [&](auto const& kv) { std::forward<Fn>(fn)(kv.second); found = true; });
        return found;
    }

    template<typename Fn>
    auto ModifyIf(Hash h, Fn&& fn) -> bool {
        return map_.modify_if(h, [&](auto& kv) { std::forward<Fn>(fn)(kv.second); });
    }

    template<typename OnExisting, typename OnNew>
    auto LazyEmplace(Hash h, OnExisting&& onExisting, OnNew&& onNew) -> void {
        map_.lazy_emplace_l(
            h,
            [&](auto& kv)         { std::forward<OnExisting>(onExisting)(kv.second); },
            [&](auto const& ctor) { Entry e{}; std::forward<OnNew>(onNew)(e); ctor(h, std::move(e)); }
        );
    }

    [[nodiscard]] auto Size() const -> std::size_t { return map_.size(); }
    auto Clear() -> void { map_.clear(); }
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
// Subclassing: JitZobrist (in jit_evaluator.hpp) is the only intended subclass.
// The virtual destructor exists to support nanobind exposure of both types.
class OPERON_EXPORT Zobrist {
    using Value   = Vector<Scalar>;
    using Extents = std::extents<int, std::dynamic_extent, std::dynamic_extent>;
    using Table   = MDArray<Hash, Extents>;

    Table table_;
    Map<Hash, int> varIndex_;

    struct TranspositionTable;
    std::unique_ptr<TranspositionTable> tt_;

    mutable std::atomic<std::size_t> hits_{0};

public:
    // variableHashes must include every variable hash that can appear in a tree.
    // Each variable gets its own row of independent random values so that
    // permuting variables at different positions always yields a different hash.
    Zobrist(RandomGenerator& rng, int maxLength, Span<Hash const> variableHashes);
    virtual ~Zobrist();
    Zobrist(Zobrist const&)            = delete;
    Zobrist(Zobrist&&)                 = delete;
    auto operator=(Zobrist const&)     -> Zobrist& = delete;
    auto operator=(Zobrist&&)          -> Zobrist& = delete;

    [[nodiscard]] auto Rows() const { return table_.extent(0); }
    [[nodiscard]] auto Cols() const { return table_.extent(1); }
    [[nodiscard]] auto OptimizeRow() const { return static_cast<int>(Rows()) - 1; }

    [[nodiscard]] auto ComputeHash(Node const& n, int pos) const -> Hash
    {
        Hash h{};
        if (n.IsVariable()) {
            auto const it = varIndex_.find(n.HashValue);
            ENSURE(it != varIndex_.end());
            auto const row = static_cast<int>(NodeTypes::Count) + it->second;
            h = table_(row, pos);
        } else {
            h = table_(NodeTypes::GetIndex(n.Type), pos);
        }
        if (n.Optimize) {
            h ^= table_(OptimizeRow(), pos);
        }
        return h;
    }

    [[nodiscard]] auto ComputeHash(Tree const& tree) const -> Hash
    {
        EXPECT(std::ssize(tree.Nodes()) <= Cols());
        Hash h{};
        auto const& nodes = tree.Nodes();
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            h ^= ComputeHash(nodes[i], i);
        }
        return h;
    }

    // Returns true and fills `val` if the hash is found; thread-safe.
    [[nodiscard]] auto TryGet(Hash hash, Value& val) const -> bool;

    // Inserts or increments the visit counter; thread-safe.
    auto Insert(Hash hash, Value const& val) -> void;

    // Clears the transposition table and resets the hit counter.
    // NOT safe to call concurrently with TryGet or Insert — call only after
    // the algorithm has fully stopped (e.g. after GeneticAlgorithm::Run returns).
    auto Clear() -> void;

    [[nodiscard]] auto Hits() const -> std::size_t { return hits_.load(); }
    [[nodiscard]] auto Size() const -> std::size_t;
};

} // namespace Operon

#endif
