// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/hash/content_hash.hpp"

#include <algorithm>
#include <vector>

namespace Operon {

namespace {
    // splitmix64/boost::hash_combine-style: cheap, no allocation, no table
    // lookups - unlike Tree::Hash()'s combine step, which concatenates children's
    // hashes into a byte buffer and runs a general-purpose streaming hash
    // (Operon::Hasher/XXH64) over it, overkill for folding in one 64-bit value.
    // Internal-only: not part of the public API, just ComputeContentHash's combine step.
    auto MixHash(Operon::Hash a, Operon::Hash b) noexcept -> Operon::Hash
    {
        constexpr Operon::Hash Mul = 0x9E3779B97F4A7C15ULL; // golden-ratio odd constant
        a ^= b + Mul + (a << 6UL) + (a >> 2UL);
        return a;
    }
} // namespace

auto ComputeContentHash(Tree const& tree, Zobrist const& zobrist, Operon::Span<Operon::Hash> scratch) noexcept -> Operon::Hash
{
    auto const& nodes = tree.Nodes();
    EXPECT(scratch.size() >= nodes.size());

    if (nodes.empty()) { return 0; } // matches Tree::HashValue()'s empty-tree convention

    std::vector<std::size_t> childIndices;
    childIndices.reserve(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto const& n = nodes[i];
        auto h = zobrist.ComputeHash(n, /*pos=*/0);

        if (!n.IsLeaf()) {
            std::ranges::copy(Tree::Indices(nodes, i), std::back_inserter(childIndices));
            auto begin = childIndices.begin();
            auto end = begin + n.Arity;
            if (n.IsCommutative()) {
                std::sort(begin, end, [&](auto a_, auto b_) { return scratch[a_] < scratch[b_]; });
            }
            for (auto it = begin; it != end; ++it) { h = MixHash(h, scratch[*it]); }
            childIndices.clear();
        }

        scratch[i] = h;
    }

    return scratch[nodes.size() - 1];
}

auto ComputeContentHash(Tree const& tree, Zobrist const& zobrist) -> Operon::Hash
{
    std::vector<Operon::Hash> scratch(tree.Nodes().size());
    return ComputeContentHash(tree, zobrist, scratch);
}

} // namespace Operon
