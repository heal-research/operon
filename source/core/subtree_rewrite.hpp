// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#ifndef OPERON_DETAIL_SUBTREE_REWRITE_HPP
#define OPERON_DETAIL_SUBTREE_REWRITE_HPP
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
namespace Operon::detail {
struct SubtreeSpan { std::size_t First; std::size_t Root; std::size_t Size; };
[[nodiscard]] inline auto DescribeSubtree(Operon::Span<Node const> nodes, std::size_t root) -> SubtreeSpan {
    EXPECT(root < nodes.size());
    auto const size = static_cast<std::size_t>(nodes[root].Length) + 1U;
    EXPECT(size <= root + 1U);
    return {root + 1U - size, root, size};
}
[[nodiscard]] inline auto RewriteSubtree(Operon::Span<Node const> source, SubtreeSpan target, Operon::Span<Node const> replacement) -> Operon::Vector<Node> {
    EXPECT(target.Root < source.size()); EXPECT(target.Size > 0U);
    EXPECT(target.First + target.Size == target.Root + 1U); EXPECT(target.First + target.Size <= source.size());
    Operon::Vector<Node> rewritten;
    rewritten.reserve(source.size() - target.Size + replacement.size());
    auto const first = source.begin() + static_cast<std::ptrdiff_t>(target.First);
    auto const after = first + static_cast<std::ptrdiff_t>(target.Size);
    std::copy(source.begin(), first, std::back_inserter(rewritten));
    std::copy(replacement.begin(), replacement.end(), std::back_inserter(rewritten));
    std::copy(after, source.end(), std::back_inserter(rewritten));
    if (std::ranges::any_of(source, [](auto const& node) { return node.IsRef(); })) {
        auto const delta = static_cast<std::ptrdiff_t>(replacement.size()) - static_cast<std::ptrdiff_t>(target.Size);
        for (std::size_t old = 0; old < source.size(); ++old) {
            if (old >= target.First && old <= target.Root) continue;
            auto const next = old < target.First ? old : static_cast<std::size_t>(static_cast<std::ptrdiff_t>(old) + delta);
            auto& node = rewritten[next];
            if (!node.IsRef()) continue;
            EXPECT(node.RefTo < target.First || node.RefTo > target.Root);
            if (node.RefTo > target.Root) node.RefTo = static_cast<uint16_t>(static_cast<std::ptrdiff_t>(node.RefTo) + delta);
        }
    }
    return rewritten;
}
[[nodiscard]] inline auto CopySubtree(Operon::Span<Node const> source, SubtreeSpan target) -> Operon::Vector<Node> {
    return RewriteSubtree(source, {0U, source.size() - 1U, source.size()}, source.subspan(target.First, target.Size));
}
}
#endif
