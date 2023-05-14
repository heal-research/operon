// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_SUBTREE_HPP
#define OPERON_SUBTREE_HPP

#include "node.hpp"

namespace Operon {
// a non-owning view over a subtree (part of a tree) with the ability
// to iterate over child nodes or child node indices
template<typename T>
requires std::is_same_v<Node, T> || std::is_same_v<Node const, T>
struct Subtree {
    Subtree(Operon::Span<T> nodes, std::size_t parent)
        : nodes_(nodes), parent_(parent)
    {}

    struct Sentinel {};

    template<typename Iterator>
    struct Enumerator {
        using value_type = std::tuple<std::size_t, typename Iterator::value_type>; // NOLINT
        using reference = std::tuple<std::size_t, typename Iterator::reference>; // NOLINT
        using difference_type = std::ptrdiff_t; // NOLINT
        using iterator_tag = std::forward_iterator_tag; // NOLINT

        Enumerator(Operon::Span<T> nodes, std::size_t parent)
            : Iter{nodes, parent}
        {}

        auto operator++() -> Enumerator& {
            ++Iter;
            ++Index;
            return *this;
        }

        auto operator++(int) -> Enumerator {
            auto tmp{*this};
            ++(*this);
            return tmp;
        }

        auto begin() const { return *this; } // NOLINT
        auto end() const { return Sentinel{}; } // NOLINT

        inline auto operator==(Enumerator const& other) {
            return Iter == other.Iter && Index == other.Index;
        }

        inline auto operator!=(Enumerator const& other) {
            return !(*this == other);
        }

        inline auto operator<(Enumerator const& other) {
            return std::tie(Index, Iter) < std::tie(other.Index, other.Iter);
        }

        inline auto operator==(Sentinel /*unused*/) const -> bool { return Iter.Done(); }

        inline auto operator*() const -> reference { return { Index, *Iter }; }
        inline auto operator*() -> reference { return { Index, *Iter }; }

        Iterator Iter;
        std::size_t Index{};
    };

    template<bool ReturnIndices = true>
    struct SubtreeIterator {
        using value_type = std::conditional_t<ReturnIndices, std::size_t, T>; // NOLINT
        using reference = std::conditional_t<ReturnIndices, std::size_t, T&>; // NOLINT
        using difference_type = std::ptrdiff_t; // NOLINT
        using iterator_tag = std::forward_iterator_tag; // NOLINT

        SubtreeIterator(Operon::Span<T> nodes, std::size_t parent)
            : Nodes(nodes), Parent(parent), Child(parent-1)
        {
            EXPECT(parent > 0);
        }

        inline auto begin() const { return *this; } // NOLINT
        inline auto end() const { return Sentinel{}; } // NOLINT

        inline auto operator==(SubtreeIterator const& other) const {
            return Parent == other.Parent
                && Child == other.Child
                && Index == other.Index
                && Nodes.data() == other.Nodes.data();
        }

        inline auto operator<(SubtreeIterator const& other) const {
            return Index < other.Index;
        }

        inline auto operator++() -> SubtreeIterator& {
            Child -= Nodes[Child].Length + 1;
            Index += 1;
            return *this;
        }

        inline auto operator++(int) -> SubtreeIterator {
            auto tmp{*this};
            ++(*this);
            return tmp;
        }

        inline auto operator==(Sentinel /*unused*/) const -> bool { return Done(); }

        inline auto operator*() const -> reference {
            if constexpr (ReturnIndices) { return Child; }
            else { return Nodes[Child]; }
        }

        inline auto operator*() -> reference {
            if constexpr (ReturnIndices) { return Child; }
            else { return Nodes[Child]; }
        }

        [[nodiscard]] inline auto Done() const -> bool { return Index >= Nodes[Parent].Arity; }

        Operon::Span<T> Nodes;
        std::size_t Parent;
        std::size_t Child;
        std::size_t Index{};
    };

    using IndexIterator = SubtreeIterator<true>;
    using NodeIterator = SubtreeIterator<false>;

    [[nodiscard]] inline auto Indices() const { return IndexIterator{nodes_, parent_}; }
    [[nodiscard]] inline auto EnumerateIndices() const { return Enumerator<IndexIterator>{nodes_, parent_}; }

    [[nodiscard]] inline auto Nodes() const { return NodeIterator{nodes_, parent_}; }
    [[nodiscard]] inline auto EnumerateNodes() const { return Enumerator<NodeIterator>{nodes_, parent_}; }

private:
    Operon::Span<T> nodes_;
    std::size_t parent_;
};

} // namespace Operon

#endif
