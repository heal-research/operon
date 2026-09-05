// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_SUBTREE_HPP
#define OPERON_SUBTREE_HPP

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

#include "contracts.hpp"
#include "node.hpp"

namespace Operon {
// A non-owning view over a subtree that exposes its direct children in postfix order.
template <typename T>
    requires std::is_same_v<Node, T> || std::is_same_v<Node const, T>
struct Subtree {
    struct Sentinel {};
    template <bool ReturnIndices>
    class ChildRange;

    template <bool ReturnIndices>
    class EnumerateRange;

    template <bool ReturnIndices>
    class SubtreeIterator {
    public:
        using value_type = std::conditional_t<ReturnIndices, std::size_t, std::remove_const_t<T>>; // NOLINT
        using reference = std::conditional_t<ReturnIndices, std::size_t, T&>; // NOLINT
        using difference_type = std::ptrdiff_t; // NOLINT
        using iterator_concept = std::forward_iterator_tag; // NOLINT
        using iterator_category = std::forward_iterator_tag; // NOLINT

        SubtreeIterator() = default;

        auto operator++() -> SubtreeIterator&
        {
            ++position_;
            if (position_ < arity_) {
                child_ -= nodes_[child_].Length + 1;
            }
            return *this;
        }

        auto operator++(int) -> SubtreeIterator
        {
            auto copy { *this };
            ++(*this);
            return copy;
        }

        [[nodiscard]] auto operator*() const -> reference
        {
            if constexpr (ReturnIndices) {
                return child_;
            } else {
                return nodes_[child_];
            }
        }

        friend auto operator==(SubtreeIterator const& lhs, SubtreeIterator const& rhs) -> bool
        {
            return lhs.nodes_.data() == rhs.nodes_.data()
                && lhs.nodes_.size() == rhs.nodes_.size()
                && lhs.parent_ == rhs.parent_
                && lhs.child_ == rhs.child_
                && lhs.position_ == rhs.position_;
        }

        friend auto operator==(SubtreeIterator const& iterator, Sentinel /*unused*/) -> bool
        {
            return iterator.position_ >= iterator.arity_;
        }

        friend auto operator==(Sentinel sentinel, SubtreeIterator const& iterator) -> bool { return iterator == sentinel; }

    private:
        friend class ChildRange<ReturnIndices>;

        SubtreeIterator(Operon::Span<T> nodes, std::size_t parent)
            : nodes_(nodes)
            , parent_(parent)
            , child_(nodes[parent].Arity == 0 ? parent : parent - 1)
            , arity_(nodes[parent].Arity)
        {
        }

        Operon::Span<T> nodes_ {};
        std::size_t parent_ {};
        std::size_t child_ {};
        std::size_t position_ {};
        std::size_t arity_ {};
    };

    template <bool ReturnIndices>
    class ChildRange {
    public:
        [[nodiscard]] auto begin() const -> SubtreeIterator<ReturnIndices> { return { nodes_, parent_ }; }
        [[nodiscard]] auto end() const -> Sentinel { return {}; }

    private:
        friend struct Subtree;

        ChildRange(Operon::Span<T> nodes, std::size_t parent)
            : nodes_(nodes)
            , parent_(parent)
        {
        }

        Operon::Span<T> nodes_ {};
        std::size_t parent_ {};
    };

    template <bool ReturnIndices>
    class Enumerator {
    public:
        using Iterator = SubtreeIterator<ReturnIndices>;
        using value_type = std::tuple<std::size_t, typename Iterator::value_type>; // NOLINT
        using reference = std::tuple<std::size_t, typename Iterator::reference>; // NOLINT
        using difference_type = std::ptrdiff_t; // NOLINT
        using iterator_concept = std::forward_iterator_tag; // NOLINT
        using iterator_category = std::forward_iterator_tag; // NOLINT

        Enumerator() = default;

        auto operator++() -> Enumerator&
        {
            ++iterator_;
            ++index_;
            return *this;
        }

        auto operator++(int) -> Enumerator
        {
            auto copy { *this };
            ++(*this);
            return copy;
        }

        [[nodiscard]] auto operator*() const -> reference { return { index_, *iterator_ }; }

        friend auto operator==(Enumerator const& lhs, Enumerator const& rhs) -> bool
        {
            return lhs.iterator_ == rhs.iterator_ && lhs.index_ == rhs.index_;
        }

        friend auto operator==(Enumerator const& iterator, Sentinel sentinel) -> bool { return iterator.iterator_ == sentinel; }
        friend auto operator==(Sentinel sentinel, Enumerator const& iterator) -> bool { return iterator == sentinel; }

    private:
        friend class EnumerateRange<ReturnIndices>;

        explicit Enumerator(Iterator iterator)
            : iterator_(std::move(iterator))
        {
        }

        Iterator iterator_ {};
        std::size_t index_ {};
    };

    template <bool ReturnIndices>
    class EnumerateRange {
    public:
        [[nodiscard]] auto begin() const -> Enumerator<ReturnIndices> { return Enumerator<ReturnIndices> { ChildRange<ReturnIndices> { nodes_, parent_ }.begin() }; }
        [[nodiscard]] auto end() const -> Sentinel { return {}; }

    private:
        friend struct Subtree;

        EnumerateRange(Operon::Span<T> nodes, std::size_t parent)
            : nodes_(nodes)
            , parent_(parent)
        {
        }

        Operon::Span<T> nodes_ {};
        std::size_t parent_ {};
    };

    using IndexIterator = SubtreeIterator<true>;
    using NodeIterator = SubtreeIterator<false>;
    using IndexRange = ChildRange<true>;
    using NodeRange = ChildRange<false>;

    Subtree(Operon::Span<T> nodes, std::size_t parent)
        : nodes_(nodes)
        , parent_(parent)
    {
        EXPECT(parent < nodes_.size());
    }

    [[nodiscard]] auto Indices() const -> IndexRange { return { nodes_, parent_ }; }
    [[nodiscard]] auto EnumerateIndices() const -> EnumerateRange<true> { return { nodes_, parent_ }; }

    [[nodiscard]] auto Nodes() const -> NodeRange { return { nodes_, parent_ }; }
    [[nodiscard]] auto EnumerateNodes() const -> EnumerateRange<false> { return { nodes_, parent_ }; }

private:
    Operon::Span<T> nodes_;
    std::size_t parent_;
};

} // namespace Operon
#endif
