// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_TREE_HASH_HPP
#define OPERON_CORE_TREE_HASH_HPP

#include <cstdint>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon::detail {

// Pure, non-mutating, coefficient-aware hash for memoization caches that may be
// queried concurrently on shared trees.
inline auto HashTreeForMemo(Tree const& tree) -> Operon::Hash
{
    Operon::Hasher const hasher;
    Operon::Hash h{};
    for (auto const& n : tree.Nodes()) {
        auto const valueHash = hasher(reinterpret_cast<std::uint8_t const*>(&n.Value), sizeof(n.Value)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        auto const nodeHash = n.HashValue ^ (valueHash + 0x9e3779b97f4a7c15ULL + (n.HashValue << 6U) + (n.HashValue >> 2U));
        h ^= nodeHash + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    }
    return h;
}

} // namespace Operon::detail

#endif
