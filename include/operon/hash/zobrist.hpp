// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_HASH_ZOBRIST_HPP
#define OPERON_HASH_ZOBRIST_HPP

#include <atomic>
#include <memory>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Position-aware Zobrist hash for GP trees, plus a transposition table that
// caches fitness vectors keyed by structural hash.  The hash captures tree
// topology and variable identity but not coefficient values — by design, so
// that a cached fitness (computed after local search) can be reused for any
// structurally identical tree regardless of its current coefficients.
//
// Ownership: the caller constructs one Zobrist per experiment (or per run) and
// passes a raw pointer into GeneticAlgorithmConfig::Cache.  The algorithm
// borrows the pointer; the caller is responsible for lifetime.
class OPERON_EXPORT Zobrist {
    using Value = Operon::Vector<Operon::Scalar>;
    using Extents = std::extents<int, std::dynamic_extent, std::dynamic_extent>;
    using Table   = Operon::MDArray<Operon::Hash, Extents>;

    Table table_;
    Operon::Map<Operon::Hash, int> varIndex_;

    struct TranspositionTable;
    std::unique_ptr<TranspositionTable> tt_;

    mutable std::atomic<std::size_t> hits_{0};

public:
    // variableHashes must include every variable hash that can appear in a tree.
    // Each variable gets its own row of independent random values so that
    // permuting variables at different positions always yields a different hash.
    Zobrist(Operon::RandomGenerator& rng, int maxLength, Operon::Span<Operon::Hash const> variableHashes);
    ~Zobrist();
    Zobrist(Zobrist const&)            = delete;
    Zobrist(Zobrist&&)                 = delete;
    auto operator=(Zobrist const&)     -> Zobrist& = delete;
    auto operator=(Zobrist&&)          -> Zobrist& = delete;

    [[nodiscard]] auto Rows() const { return table_.extent(0); }
    [[nodiscard]] auto Cols() const { return table_.extent(1); }

    [[nodiscard]] auto ComputeHash(Operon::Node const& n, int pos) const -> Operon::Hash
    {
        if (n.IsVariable()) {
            auto const it = varIndex_.find(n.HashValue);
            ENSURE(it != varIndex_.end());
            auto const row = static_cast<int>(NodeTypes::Count) + it->second;
            return table_(row, pos);
        }
        return table_(NodeTypes::GetIndex(n.Type), pos);
    }

    [[nodiscard]] auto ComputeHash(Operon::Tree const& tree) const -> Operon::Hash
    {
        EXPECT(std::ssize(tree.Nodes()) <= Cols());
        Operon::Hash h{};
        auto const& nodes = tree.Nodes();
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            h ^= ComputeHash(nodes[i], i);
        }
        return h;
    }

    // Returns true and fills `val` if the hash is found; thread-safe.
    [[nodiscard]] auto TryGet(Operon::Hash hash, Value& val) const -> bool;

    // Inserts or increments the visit counter; thread-safe.
    auto Insert(Operon::Hash hash, Value const& val) -> void;

    // Clears the transposition table and resets the hit counter.
    // NOT safe to call concurrently with TryGet or Insert — call only after
    // the algorithm has fully stopped (e.g. after GeneticAlgorithm::Run returns).
    auto Clear() -> void;

    [[nodiscard]] auto Hits() const -> std::size_t { return hits_.load(); }
    [[nodiscard]] auto Size() const -> std::size_t;
};


} // namespace Operon

#endif
