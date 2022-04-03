// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "operon/core/node.hpp"
#include "operon/operon_export.hpp"
#include "contracts.hpp"

namespace Operon {

template<typename T>
class SubtreeIterator {
public:
    // iterator traits
    using value_type = std::conditional_t<std::is_const_v<T>, Node const, Node>;// NOLINT
    using difference_type = std::ptrdiff_t;// NOLINT
    using pointer = value_type*;// NOLINT
    using reference = value_type&;// NOLINT
    using iterator_category = std::forward_iterator_tag;// NOLINT

    explicit SubtreeIterator(T& tree, size_t i)
        : nodes_(tree.Nodes())
        , parentIndex_(i)
        , index_(i - 1)
    {
        EXPECT(i > 0);
    }

    inline auto operator*() -> value_type& { return nodes_[index_]; }
    inline auto operator*() const -> value_type const& { return nodes_[index_]; }
    inline auto operator->() -> value_type* { return &**this; }
    inline auto operator->() const -> value_type const* { return &**this; }

    auto operator++() -> SubtreeIterator& // pre-increment
    {
        index_ -= nodes_[index_].Length + 1UL;
        return *this;
    }

    auto operator++(int) -> SubtreeIterator // post-increment
    {
        auto t = *this;
        ++t;
        return t;
    }

    auto operator==(SubtreeIterator const& rhs) -> bool
    {
        return std::tie(index_, parentIndex_, nodes_.data()) == std::tie(rhs.index_, rhs.parentIndex_, rhs.nodes_.data());
    }

    auto operator!=(SubtreeIterator const& rhs) -> bool
    {
        return !(*this == rhs);
    }

    auto operator<(SubtreeIterator const& rhs) -> bool
    {
        // this looks a little strange, but correct: we use a postfix representation and trees are iterated from right to left
        // (thus the lower index will be the more advanced iterator)
        return std::tie(parentIndex_, nodes_.data()) == std::tie(rhs.parentIndex_, rhs.nodes_.data()) && index_ > rhs.index_;
    }

    inline auto HasNext() -> bool { return index_ < parentIndex_ && index_ >= (parentIndex_ - nodes_[parentIndex_].Length); }
    [[nodiscard]] inline auto Index() const -> size_t { return index_; } // index of current child

private:
    Operon::Span<value_type> nodes_;
    size_t parentIndex_; // index of parent node
    size_t index_;       // index of current child node
};

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

    [[nodiscard]] auto Subtree(size_t i) const -> Tree {
        EXPECT(i < Length());
        auto const& n = nodes_[i];
        auto it = nodes_.begin() + std::make_signed_t<size_t>(i);
        return Tree({it - n.Length, it + 1}).UpdateNodes();
    }

    [[nodiscard]] auto ChildIndices(size_t i) const -> std::vector<size_t>;
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

    inline auto operator[](size_t i) noexcept -> Node& { return nodes_[i]; }
    inline auto operator[](size_t i) const noexcept -> Node const& { return nodes_[i]; }

    [[nodiscard]] auto Length() const noexcept -> size_t { return nodes_.size(); }
    [[nodiscard]] auto VisitationLength() const noexcept -> size_t;
    [[nodiscard]] auto Depth() const noexcept -> size_t;
    [[nodiscard]] auto Empty() const noexcept -> bool { return nodes_.empty(); }

    [[nodiscard]] auto HashValue() const -> Operon::Hash { return nodes_.empty() ? 0 : nodes_.back().CalculatedHashValue; }

    auto Children(size_t i) -> SubtreeIterator<Tree> { return SubtreeIterator(*this, i); }
    [[nodiscard]] auto Children(size_t i) const -> SubtreeIterator<Tree const> { return SubtreeIterator(*this, i); }

private:
    Operon::Vector<Node> nodes_;
};
} // namespace Operon
#endif // TREE_H

