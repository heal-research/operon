// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_DETAIL_SUBTREE_REWRITE_HPP
#define OPERON_DETAIL_SUBTREE_REWRITE_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

namespace Operon::detail {

struct SubtreeSpan {
    std::size_t First;
    std::size_t Root;
    std::size_t Size;
};

[[nodiscard]] inline auto DescribeSubtree(Operon::Span<Node const> nodes, std::size_t root) -> SubtreeSpan
{
    EXPECT(root < nodes.size());
    auto const size = static_cast<std::size_t>(nodes[root].Length) + 1U;
    EXPECT(size <= root + 1U);
    return { root + 1U - size, root, size };
}

[[nodiscard]] inline auto IsSelfContainedSubtree(Operon::Span<Node const> nodes, SubtreeSpan span) -> bool
{
    return std::ranges::all_of(nodes.subspan(span.First, span.Size), [first = span.First, root = span.Root](Node const& node) {
        return !node.IsRef() || (node.RefTo >= first && node.RefTo <= root);
    });
}

// Replacement and copied subtrees must not reference nodes outside their span.
[[nodiscard]] inline auto CanRewriteSubtree(Operon::Span<Node const> replacement) -> bool
{
    return !replacement.empty() && IsSelfContainedSubtree(replacement, { 0U, replacement.size() - 1U, replacement.size() });
}

[[nodiscard]] inline auto RewriteSubtree(Operon::Span<Node const> source, SubtreeSpan target,
    Operon::Span<Node const> replacement) -> Operon::Vector<Node>
{
    if (!CanRewriteSubtree(replacement)) {
        throw std::invalid_argument("replacement subtree has external Ref targets");
    }

    Operon::Vector<Node> rewritten;
    rewritten.reserve(source.size() - target.Size + replacement.size());
    auto const first = source.begin() + static_cast<std::ptrdiff_t>(target.First);
    auto const after = first + static_cast<std::ptrdiff_t>(target.Size);
    std::copy(source.begin(), first, std::back_inserter(rewritten));
    std::copy(replacement.begin(), replacement.end(), std::back_inserter(rewritten));
    std::copy(after, source.end(), std::back_inserter(rewritten));

    auto const replacementRoot = target.First + replacement.size() - 1U;
    auto const delta = static_cast<std::ptrdiff_t>(replacement.size()) - static_cast<std::ptrdiff_t>(target.Size);
    for (std::size_t i = target.First; i < target.First + replacement.size(); ++i) {
        if (rewritten[i].IsRef()) {
            rewritten[i].RefTo = static_cast<uint16_t>(rewritten[i].RefTo + target.First);
        }
    }
    for (std::size_t old = target.Root + 1U; old < source.size(); ++old) {
        auto& node = rewritten[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(old) + delta)];
        if (!node.IsRef()) {
            continue;
        }
        if (node.RefTo >= target.First && node.RefTo <= target.Root) {
            node.RefTo = static_cast<uint16_t>(replacementRoot);
        } else if (node.RefTo > target.Root) {
            node.RefTo = static_cast<uint16_t>(static_cast<std::ptrdiff_t>(node.RefTo) + delta);
        }
    }
    return rewritten;
}

[[nodiscard]] inline auto CopySubtree(Operon::Span<Node const> source, SubtreeSpan target) -> Operon::Vector<Node>
{
    if (!IsSelfContainedSubtree(source, target)) {
        throw std::invalid_argument("cannot splice a subtree with external Ref targets");
    }
    Operon::Vector<Node> copy(source.begin() + static_cast<std::ptrdiff_t>(target.First),
        source.begin() + static_cast<std::ptrdiff_t>(target.Root) + 1);
    for (auto& node : copy) {
        if (node.IsRef()) {
            node.RefTo = static_cast<uint16_t>(node.RefTo - target.First);
        }
    }
    return copy;
}

} // namespace Operon::detail

#endif
