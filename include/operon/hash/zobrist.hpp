// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_HASH_ZOBRIST_HPP
#define OPERON_HASH_ZOBRIST_HPP

#include <atomic>
#include <memory>

#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Position-aware Zobrist hash for GP trees, plus a transposition table that
// caches evaluated individuals keyed by structural hash.  The hash captures
// tree topology and variable identity but not coefficient values — by design,
// so that a cached fitness (computed after local search) can be reused for any
// structurally identical tree regardless of its current coefficients.
//
// Ownership: the caller constructs one Zobrist per experiment (or per run) and
// passes a raw pointer into GeneticAlgorithmConfig::Cache.  The algorithm
// borrows the pointer; the caller is responsible for lifetime.
class OPERON_EXPORT Zobrist {
    using Extents = std::extents<int, static_cast<int>(NodeTypes::Count), std::dynamic_extent>;
    using Table   = Operon::MDArray<Operon::Hash, Extents>;

    Table table_;

    struct TranspositionTable;
    std::unique_ptr<TranspositionTable> tt_;

    mutable std::atomic<std::size_t> hits_{0};

public:
    Zobrist(Operon::RandomGenerator& rng, int maxLength);
    ~Zobrist();
    Zobrist(Zobrist const&)            = delete;
    Zobrist(Zobrist&&)                 = delete;
    auto operator=(Zobrist const&)     -> Zobrist& = delete;
    auto operator=(Zobrist&&)          -> Zobrist& = delete;

    [[nodiscard]] auto ComputeHash(Operon::Node const& n, int pos) const -> Operon::Hash
    {
        auto const i = NodeTypes::GetIndex(n.Type);
        auto h = table_(i, pos);
        if (n.IsVariable()) { h ^= n.HashValue; }
        return h;
    }

    [[nodiscard]] auto ComputeHash(Operon::Tree const& tree) const -> Operon::Hash
    {
        Operon::Hash h{};
        auto const& nodes = tree.Nodes();
        for (auto i = 0; i < std::ssize(nodes); ++i) {
            h ^= ComputeHash(nodes[i], i);
        }
        return h;
    }

    // Returns true and fills `ind` if the hash is found; thread-safe.
    [[nodiscard]] auto TryGet(Operon::Hash hash, Operon::Individual& ind) const -> bool;

    // Inserts or increments the visit counter; thread-safe.
    auto Insert(Operon::Hash hash, Operon::Individual const& ind) -> void;

    // Clears the transposition table and resets the hit counter.
    // Call between runs when sharing a cache across an experiment.
    auto Clear() -> void;

    [[nodiscard]] auto Hits() const -> std::size_t { return hits_.load(); }
    [[nodiscard]] auto Size() const -> std::size_t;
};

} // namespace Operon

#endif
