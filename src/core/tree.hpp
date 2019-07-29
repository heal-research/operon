#ifndef TREE_HPP
#define TREE_HPP

#include <optional>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>

#include "node.hpp"

namespace Operon {
    class Tree;

    namespace
    {
        template<bool IsConst, typename T = std::conditional_t<IsConst, Tree const, Tree>, typename U = std::conditional_t<IsConst, Node const, Node>> class ChildIteratorImpl 
        {
            public:
                using value_type        = U;
                using difference_type   = std::ptrdiff_t;
                using pointer           = value_type*;
                using reference         = value_type&;
                using iterator_category = std::forward_iterator_tag;

                explicit ChildIteratorImpl(T& s, size_t i) : tree(s), parentIndex(i), index(i-1), count(0), arity(s[i].Arity) { }

                value_type& operator*()              { return tree[index]; }
                value_type const& operator*() const  { return tree[index]; }
                value_type* operator->()             { return &**this;     }
                value_type const* operator->() const { return &**this;     }

                ChildIteratorImpl& operator++() // pre-increment 
                {
                    index -= tree[index].Length + 1;
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
                    return &tree == &rhs.tree && 
                        parentIndex == rhs.parentIndex && 
                        index == rhs.index && 
                        count == rhs.count;
                }

                bool operator!=(const ChildIteratorImpl& rhs)
                {
                    return !(*this == rhs);
                }

                inline bool HasNext() { return count < arity; }
                inline bool IsValid() { return arity == tree[index].Arity; }

                inline size_t Count() const { return count; } // how many children iterated so far
                inline size_t Index() const { return index; } // index of current child

            private:
                T& tree;
                const size_t parentIndex; // index of parent node
                size_t index;
                size_t count;
                const size_t arity;
        };
    }

    class Tree
    {
        public:
            using ChildIterator      = ChildIteratorImpl<false>;
            using ConstChildIterator = ChildIteratorImpl<true>;

            Tree() { }
            Tree(std::initializer_list<Node> list) : nodes(list) { }
            Tree(std::vector<Node> vec) : nodes(std::move(vec)) { }
            Tree(const Tree& rhs) : nodes(rhs.nodes) { }
            Tree(Tree&& rhs) noexcept : nodes(std::move(rhs.nodes)) { }

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

            std::vector<int> ChildIndices(int i) const;
            inline void SetEnabled(int i, bool enabled) { for (int j = i - nodes[i].Length; j <= i; ++j) { nodes[j].IsEnabled = enabled; } }

            std::vector<Node>&       Nodes()                   { return nodes; }
            const std::vector<Node>& Nodes()             const { return nodes; }
            inline size_t            CoefficientsCount() const { return std::count_if(nodes.begin(), nodes.end(), [](const Node& s){ return s.IsConstant() || s.IsVariable(); }); }

            void                SetCoefficients(const std::vector<double>& coefficients);
            std::vector<double> GetCoefficients() const;

            inline Node& operator[](int i) noexcept { return nodes[i]; }
            inline const Node& operator[](int i) const noexcept { return nodes[i]; }

            size_t Length() const noexcept { return nodes.size(); }
            size_t Depth() const noexcept;
            size_t Depth(size_t) const noexcept;
            size_t Level(size_t) const noexcept;

            uint64_t HashValue() const { return nodes.empty() ? 0 : nodes.back().CalculatedHashValue; }

            ChildIterator Children(size_t i) { return ChildIterator(*this, i); }
            ConstChildIterator Children(size_t i) const { return ConstChildIterator(*this, i); }

        private:
            std::vector<Node> nodes;
    };

}
#endif // TREE_H
