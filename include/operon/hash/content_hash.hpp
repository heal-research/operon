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

// Computes one content-hash per node into `scratch` (sized >= tree.Nodes().size()),
// bottom-up; returns the root's (last node's) hash. Exposed so callers processing
// many candidates can reuse one scratch buffer instead of allocating per call.
[[nodiscard]] OPERON_EXPORT auto ComputeContentHash(
    Operon::Tree const& tree,
    Operon::Zobrist const& zobrist,
    Operon::Span<Operon::Hash> scratch
) noexcept -> Operon::Hash;

// Convenience overload allocating its own scratch buffer.
[[nodiscard]] OPERON_EXPORT auto ComputeContentHash(
    Operon::Tree const& tree,
    Operon::Zobrist const& zobrist
) -> Operon::Hash;

} // namespace Operon

#endif
