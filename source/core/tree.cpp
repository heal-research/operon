// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <algorithm>
#include <numeric>

#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"

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
        auto j = i - 1;
        for (size_t k = 0; k < s.Arity; ++k) {
            auto& p = nodes_[j];
            s.Length = static_cast<uint16_t>(s.Length + p.Length);
            s.Depth = std::max(s.Depth, p.Depth);
            p.Parent = static_cast<uint16_t>(i);
            j -= p.Length + 1;
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

        for (auto it = Children(i); it.HasNext(); ++it) {
            if (s.HashValue == it->HashValue) {
                it->IsEnabled = false;
                s.Arity = static_cast<uint16_t>(s.Arity + it->Arity - 1);
                reduced = true;
            }
        }
    }

    // if anything was reduced (nodes were disabled), copy remaining enabled nodes
    if (reduced) {
        // erase-remove idiom https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
        nodes_.erase(remove_if(nodes_.begin(), nodes_.end(), [](Node const& s) { return !s.IsEnabled; }), nodes_.end());
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
                for (auto it = Children(i); it.HasNext(); ++it) {
                    children.push_back(it.Index());
                }
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

auto Tree::ChildIndices(size_t i) const -> std::vector<size_t>
{
    if (nodes_[i].IsLeaf()) {
        return std::vector<size_t> {};
    }
    std::vector<size_t> indices(nodes_[i].Arity);
    size_t j = 0;
    for (auto it = Children(i); it.HasNext(); ++it) {
        indices[j++] = it.Index();
    }
    return indices;
}

auto Tree::GetCoefficients() const -> std::vector<Operon::Scalar>
{
    std::vector<Operon::Scalar> coefficients;
    for (auto const& s : nodes_) {
        if (s.Optimize) {
            coefficients.push_back(s.Value);
        }
    }
    return coefficients;
}

void Tree::SetCoefficients(Operon::Span<Operon::Scalar const> coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes_) {
        if (s.Optimize) {
            s.Value = static_cast<Operon::Scalar>(coefficients[idx++]);
        }
    }
}

auto Tree::Depth() const noexcept -> size_t
{
    return nodes_.back().Depth;
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
                const size_t s1 = sizeof(Operon::Hash);
                const size_t s2 = sizeof(Operon::Scalar);
                std::array<uint8_t, s1 + s2> key {};
                auto* ptr = key.data();
                std::memcpy(ptr, &n.HashValue, s1);
                std::memcpy(ptr + s1, &n.Value, s2);
                n.CalculatedHashValue = hasher(key.data(), key.size());
            }
            continue;
        }

        for (auto it = Children(i); it.HasNext(); ++it) {
            childIndices.push_back(it.Index());
        }

        auto begin = childIndices.begin();
        auto end = begin + n.Arity;

        if (n.IsCommutative()) {
            std::stable_sort(begin, end, [&](auto a, auto b) { return nodes_[a] < nodes_[b]; });
        }
        std::transform(begin, end, std::back_inserter(hashes), [&](auto j) { return nodes_[j].CalculatedHashValue; });
        hashes.push_back(n.HashValue);

        n.CalculatedHashValue = hasher(reinterpret_cast<uint8_t*>(hashes.data()), sizeof(Operon::Hash) * hashes.size()); // NOLINT
        childIndices.clear();
        hashes.clear();
    }

    return *this;
}

} // namespace Operon
