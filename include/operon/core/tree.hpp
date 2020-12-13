/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "common.hpp"
#include "gsl/gsl"
#include "hash/hash.hpp"
#include "node.hpp"

namespace Operon {
class Tree;

namespace detail {
    template <bool IsConst, typename T = std::conditional_t<IsConst, Tree const, Tree>, typename U = std::conditional_t<IsConst, Node const, Node>>
    class ChildIteratorImpl {
    public:
        using value_type = U;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::forward_iterator_tag;

        explicit ChildIteratorImpl(T& tree, size_t i)
            : nodes(tree.Nodes())
            , parentIndex(i)
            , index(i - 1)
            , count(0)
            , arity(nodes[i].Arity)
        {
            EXPECT(i > 0);
        }

        value_type& operator*() { return nodes[index]; }
        value_type const& operator*() const { return nodes[index]; }
        value_type* operator->() { return &**this; }
        value_type const* operator->() const { return &**this; }

        ChildIteratorImpl& operator++() // pre-increment
        {
            index -= nodes[index].Length + 1ul;
            ++count;
            return *this;
        }

        ChildIteratorImpl operator++(int) // post-increment
        {
            auto t = *this;
            ++t;
            return t;
        }

        bool operator==(const ChildIteratorImpl& rhs)
        {
            return &nodes.data() == &rhs.nodes.data() && parentIndex == rhs.parentIndex && index == rhs.index && count == rhs.count;
        }

        bool operator!=(const ChildIteratorImpl& rhs)
        {
            return !(*this == rhs);
        }

        inline bool HasNext() { return count < arity; }
        inline bool IsValid() { return arity == nodes[index].Arity; }

        inline size_t Count() const { return count; } // how many children iterated so far
        inline size_t Index() const { return index; } // index of current child

    private:
        const gsl::span<U> nodes;
        const size_t parentIndex; // index of parent node
        size_t index;
        size_t count;
        const size_t arity;
    };
}

class Tree {
public:
    using ChildIterator = detail::ChildIteratorImpl<false>;
    using ConstChildIterator = detail::ChildIteratorImpl<true>;

    Tree() {}
    Tree(std::initializer_list<Node> list)
        : nodes(list)
    {
    }
    Tree(Operon::Vector<Node> vec)
        : nodes(std::move(vec))
    {
    }
    Tree(const Tree& rhs)
        : nodes(rhs.nodes)
    {
    }
    Tree(Tree&& rhs) noexcept
        : nodes(std::move(rhs.nodes))
    {
    }

    Tree& operator=(Tree rhs)
    {
        swap(rhs);
        return *this;
    }

    void swap(Tree& rhs) noexcept
    {
        std::swap(nodes, rhs.nodes);
    }

    Tree& UpdateNodes();
    Tree& Sort();
    Tree& Reduce();
    Tree& Simplify();

    // convenience method to make it easier to call from the Python side
    Tree& Hash(Operon::HashFunction f, Operon::HashMode m);

    // performs hashing in a manner similar to Merkle trees
    // aggregating hash values from the leafs towards the root node
    template <Operon::HashFunction H>
    Tree& Hash(Operon::HashMode mode) noexcept
    {
        std::vector<size_t> childIndices;
        childIndices.reserve(nodes.size());

        std::vector<Operon::Hash> hashes;
        hashes.reserve(nodes.size());

        Operon::Hasher<H> hasher;

        for (size_t i = 0; i < nodes.size(); ++i) {
            auto& n = nodes[i];

            if (n.IsLeaf()) {
                n.CalculatedHashValue = n.HashValue;
                if (mode == Operon::HashMode::Strict) {
                    const size_t s1 = sizeof(Operon::Hash);
                    const size_t s2 = sizeof(Operon::Scalar);
                    uint8_t key[s1 + s2];
                    std::memcpy(key, &n.HashValue, s1);
                    std::memcpy(key + s1, &n.Value, s2);
                    n.CalculatedHashValue = hasher(key, sizeof(key));
                } else {
                    n.CalculatedHashValue = n.HashValue;
                }
                continue;
            }

            for (auto it = Children(i); it.HasNext(); ++it) {
                childIndices.push_back(it.Index());
            }

            auto begin = childIndices.begin();
            auto end = begin + n.Arity;

            if (n.IsCommutative()) {
                std::sort(begin, end, [&](auto a, auto b) { return nodes[a] < nodes[b]; });
            }
            std::transform(begin, end, std::back_inserter(hashes), [&](auto j) { return nodes[j].CalculatedHashValue; });
            hashes.push_back(n.HashValue);

            n.CalculatedHashValue = hasher(reinterpret_cast<uint8_t*>(hashes.data()), sizeof(Operon::Hash) * hashes.size());
            childIndices.clear();
            hashes.clear();
        }

        return *this;
    }

    std::vector<size_t> ChildIndices(size_t i) const;
    inline void SetEnabled(size_t i, bool enabled)
    {
        for (auto j = i - nodes[i].Length; j <= i; ++j) {
            nodes[j].IsEnabled = enabled;
        }
    }

    Operon::Vector<Node>& Nodes() & { return nodes; }
    Operon::Vector<Node>&& Nodes() && { return std::move(nodes); }
    const Operon::Vector<Node>& Nodes() const& { return nodes; }

    inline auto CoefficientsCount() const
    {
        return std::count_if(nodes.cbegin(), nodes.cend(), [](auto const& s) { return s.IsLeaf(); });
    }

    void SetCoefficients(const gsl::span<const Operon::Scalar> coefficients);
    std::vector<Operon::Scalar> GetCoefficients() const;

    inline Node& operator[](size_t i) noexcept { return nodes[i]; }
    inline const Node& operator[](size_t i) const noexcept { return nodes[i]; }

    size_t Length() const noexcept { return nodes.size(); }
    size_t VisitationLength() const noexcept;
    size_t Depth() const noexcept;
    bool Empty() const noexcept { return nodes.empty(); }

    Operon::Hash HashValue() const { return nodes.empty() ? 0 : nodes.back().CalculatedHashValue; }

    ChildIterator Children(size_t i) { return ChildIterator(*this, i); }
    ConstChildIterator Children(size_t i) const { return ConstChildIterator(*this, i); }

private:
    Operon::Vector<Node> nodes;
};
}
#endif // TREE_H

