// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <cstdint>
#include <vector>
#include <numeric>
#include <type_traits>

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

    void SetEnabled(size_t i, bool enabled)
    {
        for (auto j = i - nodes_[i].Length; j <= i; ++j) {
            nodes_[j].IsEnabled = enabled;
        }
    }

    // One deducing-this member replaces the former &/&&/const& overload
    // triplet: Self deduces the caller's value category and cv-qualifier,
    // and forwarding through it onto nodes_ reproduces exactly the same
    // three return types (Vector<Node>&, Vector<Node> const&,
    // Vector<Node>&&) the overloads gave.
    //
    // Return type is `auto&&`, not `decltype(auto)`. For the lvalue and
    // const-lvalue cases, decltype(auto) on an unparenthesized member-access
    // return expression yields the *declared* member type by value
    // (Operon::Vector<Node>, a silent copy) rather than a reference - the
    // real defect: it loses reference identity, not just "the value
    // category." For the rvalue case specifically, decltype(auto) would
    // still move (the expression is still an xvalue, so the by-value result
    // is move-constructed, not copied) - it only changes the return *type*
    // from Vector<Node>&& to a by-value Vector<Node> prvalue. `auto&&`
    // performs ordinary forwarding-reference deduction instead, which
    // reproduces all three original reference types exactly.
    template<typename Self>
    [[nodiscard]] auto&& Nodes(this Self&& self) { return std::forward<Self>(self).nodes_; }

    [[nodiscard]] auto CoefficientsCount() const
    {
        return std::count_if(nodes_.cbegin(), nodes_.cend(), [](auto const& s) { return s.Optimize; });
    }

    void SetCoefficients(Operon::Span<Operon::Scalar const> coefficients);
    [[nodiscard]] auto GetCoefficients() const -> std::vector<Operon::Scalar>;
    void GetCoefficients(std::vector<Operon::Scalar>& out) const;

    [[nodiscard]] auto ApplyCoefficients(Operon::Span<Operon::Scalar const> coefficients) const
    {
        auto tree{ *this };
        tree.SetCoefficients(coefficients);
        return tree;
    }

    template<typename Self>
    auto operator[](this Self& self, size_t i) noexcept -> decltype(auto) { return (self.nodes_[i]); }

    [[nodiscard]] auto Length() const noexcept -> size_t { return nodes_.size(); }
    [[nodiscard]] auto AdjustedLength() const noexcept -> size_t {
        auto length = [](auto const& n) {
            if (n.IsConstant() || n.IsRef()) { return 1; }
            return n.Value == Operon::Scalar{1} ? 1 : 3;
        };
        return std::transform_reduce(nodes_.begin(), nodes_.end(), 0UL, std::plus{}, length);
    }
    [[nodiscard]] auto VisitationLength() const noexcept -> size_t;
    [[nodiscard]] auto Depth() const noexcept -> size_t;
    [[nodiscard]] auto Empty() const noexcept -> bool { return nodes_.empty(); }

    [[nodiscard]] auto HashValue() const -> Operon::Hash { return nodes_.empty() ? 0 : nodes_.back().CalculatedHashValue; }

    template<typename Self>
    [[nodiscard]] auto Children(this Self& self, size_t i) {
        return Subtree<std::conditional_t<std::is_const_v<Self>, Node const, Node>>{self.nodes_, i}.Nodes();
    }
    [[nodiscard]] auto Indices(size_t i) const { return Subtree<Node const>{nodes_, i}.Indices(); }

    // convenience methods
    static auto Indices(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.Indices();
    }

    static auto EnumerateIndices(auto const& nodes, auto i) {
        return Subtree<Node const>{ nodes, static_cast<std::size_t>(i) }.EnumerateIndices();
    }

    static auto Nodes(auto&& nodes, auto i) {
        using NodeT = std::conditional_t<std::is_const_v<std::remove_reference_t<decltype(nodes)>>, Node const, Node>;
        return Subtree<NodeT>{ nodes, static_cast<std::size_t>(i) }.Nodes();
    }

    static auto EnumerateNodes(auto&& nodes, auto i) {
        using NodeT = std::conditional_t<std::is_const_v<std::remove_reference_t<decltype(nodes)>>, Node const, Node>;
        return Subtree<NodeT>{ nodes, static_cast<std::size_t>(i) }.EnumerateNodes();
    }

private:
    Operon::Vector<Node> nodes_;
};
} // namespace Operon
#endif // TREE_H
