// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_CONTENT_HASH_HPP
#define OPERON_HASH_CONTENT_HASH_HPP

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Cheap, position-independent (content-addressed) structural hash for a Tree,
// intended for subexpression memoization/dedup rather than Zobrist's whole-tree
// transposition-cache role (which is deliberately position-*dependent* and has
// no notion of combining children into a parent's hash at all).
//
// Computed bottom-up like Tree::Hash(): each node's hash folds in its children's
// already-computed hashes (sorted first when the node IsCommutative(), so e.g.
// x+y and y+x hash identically), seeded per node by zobrist.ComputeHash(node, 0)
// - reusing Zobrist's existing random table purely as a per-node-type/per-
// variable salt (pos is always 0: this hash has no array-position dimension).
// The combine step is a cheap splitmix64/boost::hash_combine-style mixer rather
// than Tree::Hash()'s concatenate-then-XXH64, since this is called on a fixed,
// tiny number of already-64-bit values per node - a general-purpose streaming
// hash function is unnecessary overhead here.
//
// Two semantic decisions carried over from Tree::Hash(): a node's Optimize
// flag is masked out (weights are re-fit, not distinct symbols for dedup
// purposes), and NodeType::Ref nodes inherit their target's hash rather than
// hashing the Ref node itself, so structurally equivalent subexpressions hash
// identically regardless of whether they're expressed via Ref sharing.

// Working buffers for the scratch overload below, both sized >= tree.Nodes().size()
// (Indices holds a node's child indices - a node's Arity is always < its own node
// count). Bundled into one struct so callers processing many candidates can carry
// and reuse both as a unit instead of allocating per call.
struct ContentHashScratch {
    Operon::Span<Operon::Hash> Hashes;
    Operon::Span<std::size_t> Indices;
};

// Computes one content-hash per node into `scratch.Hashes`, bottom-up; returns the
// root's (last node's) hash.
[[nodiscard]] OPERON_EXPORT auto ComputeContentHash(
    Operon::Tree const& tree,
    Operon::Zobrist const& zobrist,
    ContentHashScratch scratch
) noexcept -> Operon::Hash;

// Convenience overload allocating its own scratch buffer.
[[nodiscard]] OPERON_EXPORT auto ComputeContentHash(
    Operon::Tree const& tree,
    Operon::Zobrist const& zobrist
) -> Operon::Hash;

} // namespace Operon

#endif
