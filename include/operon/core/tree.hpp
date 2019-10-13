/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
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
#include "node.hpp"

namespace Operon {
class Tree;

namespace {
    template <bool IsConst, typename T = std::conditional_t<IsConst, Tree const, Tree>, typename U = std::conditional_t<IsConst, Node const, Node>>
    class ChildIteratorImpl {
    public:
        using value_type = U;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::forward_iterator_tag;

        explicit ChildIteratorImpl(T& tree, gsl::index i)
            : nodes(tree.Nodes())
            , parentIndex(i)
            , index(i - 1)
            , count(0)
            , arity(nodes[i].Arity)
        {
        }

        value_type& operator*() { return nodes[index]; }
        value_type const& operator*() const { return nodes[index]; }
        value_type* operator->() { return &**this; }
        value_type const* operator->() const { return &**this; }

        ChildIteratorImpl& operator++() // pre-increment
        {
            index -= nodes[index].Length + 1;
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
        const gsl::index parentIndex; // index of parent node
        gsl::index index;
        size_t count;
        const size_t arity;
    };
}

class Tree {
public:
    using ChildIterator = ChildIteratorImpl<false>;
    using ConstChildIterator = ChildIteratorImpl<true>;

    Tree() {}
    Tree(std::initializer_list<Node> list)
        : nodes(list)
    {
    }
    Tree(std::vector<Node> vec)
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
    Tree& UpdateNodeDepth();
    Tree& Sort(bool strict = true);
    Tree& Reduce();
    Tree& Simplify();

    std::vector<gsl::index> ChildIndices(gsl::index i) const;
    inline void SetEnabled(gsl::index i, bool enabled)
    {
        for (int j = i - nodes[i].Length; j <= i; ++j) {
            nodes[j].IsEnabled = enabled;
        }
    }

    std::vector<Node>& Nodes() { return nodes; }
    const std::vector<Node>& Nodes() const { return nodes; }
    inline size_t CoefficientsCount() const
    {
        return std::count_if(nodes.begin(), nodes.end(), [](const Node& s) { return s.IsConstant() || s.IsVariable(); });
    }

    void SetCoefficients(const std::vector<double>& coefficients);
    std::vector<double> GetCoefficients() const;

    inline Node& operator[](gsl::index i) noexcept { return nodes[i]; }
    inline const Node& operator[](gsl::index i) const noexcept { return nodes[i]; }

    size_t Length() const noexcept { return nodes.size(); }
    size_t Depth() const noexcept;
    size_t Depth(gsl::index) const noexcept;
    size_t Level(gsl::index) const noexcept;
    bool Empty() const noexcept { return nodes.empty(); }

    operon::hash_t HashValue() const { return nodes.empty() ? 0 : nodes.back().CalculatedHashValue; }

    ChildIterator Children(gsl::index i) { return ChildIterator(*this, i); }
    ConstChildIterator Children(gsl::index i) const { return ConstChildIterator(*this, i); }

private:
    std::vector<Node> nodes;
};

}
#endif // TREE_H
