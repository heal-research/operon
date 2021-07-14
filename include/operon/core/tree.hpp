// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "hash/hash.hpp"
#include "node.hpp"

#include "pdqsort.h"

namespace Operon {
class Tree;

template<typename T>
class SubtreeIterator {
public:
    using value_type = std::conditional_t<std::is_const_v<T>, Node const, Node>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    explicit SubtreeIterator(T& tree, size_t i)
        : nodes(tree.Nodes())
        , parentIndex(i)
        , index(i - 1)
    {
        EXPECT(i > 0);
    }

    inline value_type& operator*() { return nodes[index]; }
    inline value_type const& operator*() const { return nodes[index]; }
    inline value_type* operator->() { return &**this; }
    inline value_type const* operator->() const { return &**this; }

    SubtreeIterator& operator++() // pre-increment
    {
        index -= nodes[index].Length + 1ul;
        return *this;
    }

    SubtreeIterator operator++(int) // post-increment
    {
        auto t = *this;
        ++t;
        return t;
    }

    bool operator==(SubtreeIterator const& rhs)
    {
        return std::tie(index, parentIndex, nodes.data()) == std::tie(rhs.index, rhs.parentIndex, rhs.nodes.data());
    }

    bool operator!=(SubtreeIterator const& rhs)
    {
        return !(*this == rhs);
    }

    bool operator<(SubtreeIterator const& rhs)
    {
        // this looks a little strange, but correct: we use a postfix representation and trees are iterated from right to left
        // (thus the lower index will be the more advanced iterator)
        return std::tie(parentIndex, nodes.data()) == std::tie(rhs.parentIndex, rhs.nodes.data()) && index > rhs.index;
    }

    inline bool HasNext() { return index < parentIndex && index >= (parentIndex - nodes[parentIndex].Length); }
    inline size_t Index() const { return index; } // index of current child

private:
    Operon::Span<value_type> nodes;
    size_t parentIndex; // index of parent node
    size_t index;       // index of current child node
};

class Tree {
public:
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
                pdqsort(begin, end, [&](auto a, auto b) { return nodes[a] < nodes[b]; });
            }
            std::transform(begin, end, std::back_inserter(hashes), [&](auto j) { return nodes[j].CalculatedHashValue; });
            hashes.push_back(n.HashValue);

            n.CalculatedHashValue = hasher(reinterpret_cast<uint8_t*>(hashes.data()), sizeof(Operon::Hash) * hashes.size());
            childIndices.clear();
            hashes.clear();
        }

        return *this;
    }

    Tree Subtree(size_t i) {
        auto const& n = nodes[i];
        Operon::Vector<Node> subtree;
        subtree.reserve(n.Length);
        std::copy_n(nodes.begin() + i - n.Length, n.Length, std::back_inserter(subtree));
        return Tree(subtree).UpdateNodes();
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
    Operon::Vector<Node> const& Nodes() const& { return nodes; }

    inline auto CoefficientsCount() const
    {
        return std::count_if(nodes.cbegin(), nodes.cend(), [](auto const& s) { return s.IsLeaf(); });
    }

    void SetCoefficients(const Operon::Span<const Operon::Scalar> coefficients);
    std::vector<Operon::Scalar> GetCoefficients() const;

    inline Node& operator[](size_t i) noexcept { return nodes[i]; }
    inline const Node& operator[](size_t i) const noexcept { return nodes[i]; }

    size_t Length() const noexcept { return nodes.size(); }
    size_t VisitationLength() const noexcept;
    size_t Depth() const noexcept;
    bool Empty() const noexcept { return nodes.empty(); }

    Operon::Hash HashValue() const { return nodes.empty() ? 0 : nodes.back().CalculatedHashValue; }

    SubtreeIterator<Tree> Children(size_t i) { return SubtreeIterator(*this, i); }
    SubtreeIterator<Tree const> Children(size_t i) const { return SubtreeIterator(*this, i); }

private:
    Operon::Vector<Node> nodes;
};
}
#endif // TREE_H

