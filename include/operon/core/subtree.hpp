// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_SUBTREE_HPP
#define OPERON_SUBTREE_HPP

#include "contracts.hpp"
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

        auto operator==(Enumerator const& other) {
            return Iter == other.Iter && Index == other.Index;
        }

        auto operator!=(Enumerator const& other) {
            return !(*this == other);
        }

        auto operator<(Enumerator const& other) {
            return std::tie(Index, Iter) < std::tie(other.Index, other.Iter);
        }

        auto operator==(Sentinel /*unused*/) const -> bool { return Iter.Done(); }

        auto operator*() const -> reference { return { Index, *Iter }; }
        auto operator*() -> reference { return { Index, *Iter }; }

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

        auto operator==(SubtreeIterator const& other) const {
            return Parent == other.Parent
                && Child == other.Child
                && Index == other.Index
                && Nodes.data() == other.Nodes.data();
        }

        auto operator<(SubtreeIterator const& other) const {
            return Index < other.Index;
        }

        auto operator++() -> SubtreeIterator& {
            Child -= Nodes[Child].Length + 1;
            Index += 1;
            return *this;
        }

        auto operator++(int) -> SubtreeIterator {
            auto tmp{*this};
            ++(*this);
            return tmp;
        }

        auto operator==(Sentinel /*unused*/) const -> bool { return Done(); }

        auto operator*() const -> reference {
            if constexpr (ReturnIndices) { return Child; }
            else { return Nodes[Child]; }
        }

        auto operator*() -> reference {
            if constexpr (ReturnIndices) { return Child; }
            else { return Nodes[Child]; }
        }

        [[nodiscard]] auto Done() const -> bool { return Index >= Nodes[Parent].Arity; }

        Operon::Span<T> Nodes;
        std::size_t Parent;
        std::size_t Child;
        std::size_t Index{};
    };

    using IndexIterator = SubtreeIterator<true>;
    using NodeIterator = SubtreeIterator<false>;

    [[nodiscard]] auto Indices() const { return IndexIterator{nodes_, parent_}; }
    [[nodiscard]] auto EnumerateIndices() const { return Enumerator<IndexIterator>{nodes_, parent_}; }

    [[nodiscard]] auto Nodes() const { return NodeIterator{nodes_, parent_}; }
    [[nodiscard]] auto EnumerateNodes() const { return Enumerator<NodeIterator>{nodes_, parent_}; }

private:
    Operon::Span<T> nodes_;
    std::size_t parent_;
};

} // namespace Operon

#endif
