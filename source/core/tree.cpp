// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iterator>
#include <span>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"
#include "operon/core/constants.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

namespace Operon {
auto Tree::UpdateNodes() -> Tree&
{
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];

        s.Depth = 1;
        s.Length = s.Arity;
        s.Parent = 0;

        if (s.IsLeaf()) {
            continue;
        }

        for (auto& p : Tree::Nodes(nodes_, i)) {
            s.Length += p.Length;
            s.Depth = std::max(s.Depth, p.Depth);
            p.Parent = i;
        }

        ++s.Depth;
    }
    nodes_.back().Level = 1;

    for (auto it = nodes_.rbegin() + 1; it < nodes_.rend(); ++it) {
        it->Level = static_cast<uint16_t>(nodes_[it->Parent].Level + 1);
    }

    return *this;
}

auto Tree::Reduce() -> Tree&
{
    bool reduced = false;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];
        if (s.IsLeaf() || !s.IsCommutative()) {
            continue;
        }
        auto arity{ s.Arity };
        for (auto& p : Children(i)) {
            if (s.HashValue == p.HashValue) {
                p.IsEnabled = false;
                arity = static_cast<uint16_t>(arity + p.Arity - 1);
                reduced = true;
            }
        }
        s.Arity = arity;
    }

    // if anything was reduced (nodes were disabled), copy remaining enabled nodes
    if (reduced) {
        std::erase_if(nodes_, [](auto const& n) { return !n.IsEnabled; });
    }
    // else, nothing to do
    return this->UpdateNodes();
}

// Sort each function node's children according to node type and hash value
// - note that entire child subtrees / subarrays are reordered inside the nodes array
// - this method assumes node hashes are computed, usually it is preceded by a call to tree.Hash()
auto Tree::Sort() -> Tree&
{
    // preallocate memory to reduce fragmentation
    Operon::Vector<Operon::Node> sorted = nodes_;

    Operon::Vector<size_t> children;
    children.reserve(nodes_.size());

    auto start = nodes_.begin();

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto& s = nodes_[i];

        if (s.IsLeaf()) {
            continue;
        }

        auto arity = s.Arity;
        auto size = s.Length;

        if (s.IsCommutative()) {
            if (arity == size) {
                std::stable_sort(start + i - size, start + i); // NOLINT
            } else {
                std::ranges::copy(Indices(i), std::back_inserter(children));
                std::stable_sort(children.begin(), children.end(), [&](auto a, auto b) { return nodes_[a] < nodes_[b]; }); // sort child indices

                auto pos = sorted.begin() + i - size; // NOLINT
                for (auto j : children) {
                    auto& c = nodes_[j];
                    std::copy_n(start + static_cast<int64_t>(j) - c.Length, c.Length + 1, pos);
                    pos += c.Length + 1;
                }
                children.clear();
            }
        }
    }
    nodes_.swap(sorted);
    return this->UpdateNodes();
}

auto Tree::GetCoefficients() const -> std::vector<Operon::Scalar>
{
    std::vector<Operon::Scalar> coefficients;
    for (auto const& n : nodes_) {
        if (n.Optimize) {
            coefficients.push_back(n.Value);
        }
    }
    return coefficients;
}

void Tree::SetCoefficients(Operon::Span<Operon::Scalar const> coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes_) {
        if (s.Optimize) { s.Value = coefficients[idx++]; }
    }
}

auto Tree::Depth() const noexcept -> size_t
{
    return Empty() ? 0 : nodes_.back().Depth;
}

auto Tree::VisitationLength() const noexcept -> size_t
{
    return std::transform_reduce(nodes_.begin(), nodes_.end(), 0UL, std::plus<> {}, [](const auto& node) { return node.Length + 1; });
}

auto Tree::Hash(Operon::HashMode mode) const -> Tree const&
{
    std::vector<size_t> childIndices;
    childIndices.reserve(nodes_.size());

    std::vector<Operon::Hash> hashes;
    hashes.reserve(nodes_.size());

    Operon::Hasher hasher;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto const& n = nodes_[i];

        if (n.IsLeaf()) {
            n.CalculatedHashValue = n.HashValue;
            if (mode == Operon::HashMode::Strict) {
                n.CalculatedHashValue += hasher(std::bit_cast<uint8_t const*>(&n.Value), sizeof(n.Value));
            }
            continue;
        }

        std::ranges::copy(Indices(i), std::back_inserter(childIndices));

        auto begin = childIndices.begin();
        auto end = begin + n.Arity;

        if (n.IsCommutative()) {
            std::stable_sort(begin, end, [&](auto a, auto b) { return nodes_[a] < nodes_[b]; });
        }
        std::transform(begin, end, std::back_inserter(hashes), [&](auto j) { return nodes_[j].CalculatedHashValue; });
        hashes.push_back(n.HashValue);

        n.CalculatedHashValue = hasher(std::bit_cast<uint8_t*>(hashes.data()), sizeof(Operon::Hash) * hashes.size()); // NOLINT
        childIndices.clear();
        hashes.clear();
    }

    return *this;
}

} // namespace Operon
