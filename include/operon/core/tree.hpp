// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

#include "contracts.hpp"
#include "subtree.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
class OPERON_EXPORT Tree { // NOLINT
public:
    Tree() = default;
    Tree(std::initializer_list<Node> list)
        : nodes_(list)
    {
    }
    explicit Tree(Operon::Vector<Node> vec)
        : nodes_(std::move(vec))
    {
    }
    Tree(Tree const& rhs) // NOLINT
        : nodes_(rhs.nodes_)
    {
    }
    Tree(Tree&& rhs) noexcept
        : nodes_(std::move(rhs.nodes_))
    {
    }

    ~Tree() = default;

    auto operator=(Tree rhs) -> Tree&
    {
        Swap(*this, rhs);
        return *this;
    }
    // no need for move assignment operator because we use the copy-swap idiom

    friend void Swap(Tree& lhs, Tree& rhs) noexcept
    {
        std::swap(lhs.nodes_, rhs.nodes_);
    }

    auto UpdateNodes() -> Tree&;
    auto Sort() -> Tree&;
    auto Reduce() -> Tree&;
    auto Simplify() -> Tree&;

    // performs hashing in a manner similar to Merkle trees
    // aggregating hash values from the leafs towards the root node
    [[nodiscard]] auto Hash(Operon::HashMode mode) const -> Tree const&;

    // splice a subtree rooted at node with index i as a new tree
    [[nodiscard]] auto Splice(size_t i) const -> Tree {
        EXPECT(i < Length());
        auto const& n = nodes_[i];
        auto it = nodes_.begin() + static_cast<int64_t>(i);
        return Tree({it - n.Length, it + 1}).UpdateNodes();
    }

    inline void SetEnabled(size_t i, bool enabled)
    {
        for (auto j = i - nodes_[i].Length; j <= i; ++j) {
            nodes_[j].IsEnabled = enabled;
        }
    }

    auto Nodes() & -> Operon::Vector<Node>& { return nodes_; }
    auto Nodes() && -> Operon::Vector<Node>&& { return std::move(nodes_); }
    [[nodiscard]] auto Nodes() const& -> Operon::Vector<Node> const& { return nodes_; }

    [[nodiscard]] inline auto CoefficientsCount() const
    {
        return std::count_if(nodes_.cbegin(), nodes_.cend(), [](auto const& s) { return s.IsLeaf(); });
    }

    void SetCoefficients(Operon::Span<Operon::Scalar const> coefficients);
    [[nodiscard]] auto GetCoefficients() const -> std::vector<Operon::Scalar>;

    [[nodiscard]] auto ApplyCoefficients(Operon::Span<Operon::Scalar const> coefficients) const
    {
        auto tree{ *this };
        tree.SetCoefficients(coefficients);
        return tree;
    }

    inline auto operator[](size_t i) noexcept -> Node& { return nodes_[i]; }
    inline auto operator[](size_t i) const noexcept -> Node const& { return nodes_[i]; }

    [[nodiscard]] auto Length() const noexcept -> size_t { return nodes_.size(); }
    [[nodiscard]] auto VisitationLength() const noexcept -> size_t;
    [[nodiscard]] auto Depth() const noexcept -> size_t;
    [[nodiscard]] auto Empty() const noexcept -> bool { return nodes_.empty(); }

    [[nodiscard]] auto HashValue() const -> Operon::Hash { return nodes_.empty() ? 0 : nodes_.back().CalculatedHashValue; }

    [[nodiscard]] auto Children(size_t i) { return Subtree<Node>{nodes_, i}.Nodes(); }
    [[nodiscard]] auto Children(size_t i) const { return Subtree<Node const>{nodes_, i}.Nodes(); }
    [[nodiscard]] auto Indices(size_t i) const { return Subtree<Node const>{nodes_, i}.Indices(); }

    // convenience methods
    static inline auto Indices(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.Indices();
    }

    static inline auto EnumerateIndices(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.EnumerateIndices();
    }

    static inline auto Nodes(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.Nodes();
    }

    static inline auto Nodes(auto& nodes, auto i) {
        return Subtree<Node> { nodes, static_cast<std::size_t>(i) }.Nodes();
    }

    static inline auto EnumerateNodes(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.EnumerateNodes();
    }

    static inline auto EnumerateNodes(auto& nodes, auto i) {
        return Subtree<Node>{ nodes, static_cast<std::size_t>(i) }.EnumerateNodes();
    }

private:
    Operon::Vector<Node> nodes_;
};
} // namespace Operon
#endif // TREE_H
