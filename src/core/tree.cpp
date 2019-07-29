#include <iterator>
#include <utility>
#include <iostream>
#include <algorithm>
#include <optional>
#include <exception>
#include <stack>

#include "tree.hpp"
#include "jsf.hpp"
#include "xxhash/xxhash.hpp"

using namespace std;

namespace Operon {
    Tree& Tree::UpdateNodes()
    {
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto& s = nodes[i];

            if (s.IsLeaf) 
            {
                s.Arity = s.Length = 0;
                continue;
            }
            s.Length = s.Arity;
            for (auto it = Children(i); it.HasNext(); ++it)
            {
                s.Length += it->Length;
                nodes[it.Index()].Parent = i;
            }
        }
        return *this;
    }

    Tree& Tree::Reduce() {
        bool reduced = false;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto& s = nodes[i];
            if (s.IsLeaf || !s.IsCommutative)
            {
                continue;
            }

            for (auto it = Children(i); it.HasNext(); ++it)
            {
                if (s.HashValue == it->HashValue)
                {
                    it->IsEnabled = false;
                    s.Arity += it->Arity - 1;
                    reduced = true;
                }
            }
        }

        // if anything was reduced (nodes were disabled), copy remaining enabled nodes
        if (reduced)
        {
            // erase-remove idiom https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
            nodes.erase(remove_if(nodes.begin(), nodes.end(), [](const Node& s){ return !s.IsEnabled; }), nodes.end());
        }
        // else, nothing to do
        return this->UpdateNodes();
    }

    Tree& Tree::Sort()
    {
        // preallocate memory to reduce fragmentation
        vector<Node> sorted;
        sorted.reserve(nodes.size());

        vector<int> children;
        children.reserve(nodes.size());
        
        vector<uint64_t> hashes;
        hashes.reserve(nodes.size());

        auto start = nodes.begin();
        for (size_t i = 0; i < nodes.size(); ++i) 
        {
            auto& s = nodes[i];
            if (s.IsLeaf)
            {
                continue;
            }

            auto arity  = s.Arity;
            auto size   = s.Length;
            auto sBegin = start + i - size;
            auto sEnd   = start + i;

            if (s.IsCommutative)
            {
                if (arity == size)
                {
                    sort(sBegin, sEnd);
                }
                else
                {
                    for(auto it = Children(i); it.HasNext(); ++it)
                    {
                        children.push_back(it.Index());
                    }
                    sort(children.begin(), children.end(), [&](int a, int b) { return nodes[a] < nodes[b]; }); // sort child indices

                    for (auto j : children) 
                    {
                        auto& c = nodes[j];
                        copy_n(start + j - c.Length, c.Length + 1, back_inserter(sorted));
                    }
                    copy(sorted.begin(), sorted.end(), sBegin);
                    sorted.clear();
                    children.clear();
                }
            }
            transform(sBegin, sEnd, back_inserter(hashes), [](const Node& x) { return x.CalculatedHashValue; }); 
            hashes.push_back(s.HashValue);
            s.CalculatedHashValue = xxh::xxhash<64>(hashes);        
            hashes.clear();
        }
        return *this;
    }

    vector<int> Tree::ChildIndices(int i) const
    {
        if (nodes[i].IsLeaf)
        {
            return std::vector<int>{};
        }
        std::vector<int> indices(nodes[i].Arity);
        for (auto it = Children(i); it.HasNext(); ++it)
        {
            indices[it.Count()] = it.Index();
        }
        return indices;
    }

    std::vector<double> Tree::GetCoefficients() const
    {
        std::vector<double> coefficients;
        for (auto& s : nodes)
        {
            if (s.IsConstant() || s.IsVariable()) 
            {
                coefficients.push_back(s.Value);
            }
        }
        return coefficients;
    }

    void Tree::SetCoefficients(const std::vector<double>& coefficients)
    {
        size_t idx = 0;
        for (auto& s : nodes)
        {
            if (s.IsConstant() || s.IsVariable())
            {
                s.Value = coefficients[idx++];
            }
        }
    }
    
    size_t Tree::Depth() const noexcept 
    {
        return Depth(nodes.size() - 1);
    }

    // calculate the depth of subtree at index i
    size_t Tree::Depth(size_t i) const noexcept 
    {
        size_t depth = 1; 

        std::stack<std::pair<size_t, size_t>> st;
        st.push({ i, 1 });

        while(!st.empty())
        {
            auto t = st.top();
            st.pop();
            auto currDepth = t.second + 1;

            for (auto it = Children(t.first); it.HasNext(); ++it)
            {
                if (it->IsLeaf)
                {
                    continue;
                }
                st.push({ it.Index(), currDepth });

                if (depth < currDepth) 
                {
                    depth = currDepth;
                }
            }
        }

        return depth; // +1 for the level of leafs
    }

    // calculate the level in the tree (distance to tree root) for the subtree at index i
    size_t Tree::Level(size_t i) const noexcept
    {
        // the root node is always the last node with index Length() - 1
        auto root = Length() - 1;

        size_t level = 0;
        while(i < root && nodes[i].Parent != root)
        {
            i = nodes[i].Parent;
            ++level;
        }
        return level;
    }
}

