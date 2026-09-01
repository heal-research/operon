// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_TREE_HASH_HPP
#define OPERON_CORE_TREE_HASH_HPP

#include <cstdint>
#include <cstring>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon::detail {

// Pure, non-mutating, coefficient-aware hash for memoization caches that may be
// queried concurrently on shared trees. Used only as the key for the in-memory,
// per-generation Feasible/Measure caches in shape_constrained_evaluator.cpp, so
// it is never persisted or compared across runs and may change freely.
//
// The coefficient's raw bit pattern is folded in inline rather than dispatched
// through the exported XXH64 Hasher -- profiling a shape-constrained run had
// Hasher::operator() as the #1 self-time symbol almost entirely from this
// site, one cross-DSO call per tree node to hash a 4-byte float where the
// call setup cost dwarfed the work. The multiply by the golden ratio is
// bijective mod 2^64, so distinct coefficient bit patterns map to distinct
// valueHash before the same boost-style combine as before -- no collision
// regression vs the prior per-node 4-byte XXH64 (which was never injective on
// more than a 32-bit input anyway).
inline auto HashTreeForMemo(Tree const& tree) -> Operon::Hash
{
    Operon::Hash h{};
    for (auto const& n : tree.Nodes()) {
        Operon::Hash valueHash{}; // zero-init so the high bytes are stable when sizeof(Scalar) < sizeof(Hash)
        std::memcpy(&valueHash, &n.Value, sizeof(n.Value)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        valueHash *= 0x9e3779b97f4a7c15ULL;
        auto const nodeHash = n.HashValue ^ (valueHash + 0x9e3779b97f4a7c15ULL + (n.HashValue << 6U) + (n.HashValue >> 2U));
        h ^= nodeHash + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        if (n.IsRef()) {
            auto const target = static_cast<Operon::Hash>(n.RefTo);
            h ^= target + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        }
    }
    return h;
}

} // namespace Operon::detail

#endif
