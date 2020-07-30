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

#include <algorithm>
#include <exception>
#include <execution>
#include <iostream>
#include <iterator>
#include <optional>
#include <stack>
#include <utility>

#include "core/common.hpp"
#include "core/tree.hpp"

namespace Operon {
Tree& Tree::UpdateNodes()
{
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& s = nodes[i];

        s.Depth = 1;
        s.Length = s.Arity;
        if (s.IsLeaf()) {
            s.Arity = s.Length = 0;
            continue;
        }
        for (auto it = Children(i); it.HasNext(); ++it) {
            s.Length += it->Length;
            if (s.Depth < it->Depth) {
                s.Depth = it->Depth;
            }
            nodes[it.Index()].Parent = i;
        }
        ++s.Depth;
    }
    return *this;
}

Tree& Tree::Reduce()
{
    bool reduced = false;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& s = nodes[i];
        if (s.IsLeaf() || !s.IsCommutative()) {
            continue;
        }

        for (auto it = Children(i); it.HasNext(); ++it) {
            if (s.HashValue == it->HashValue) {
                it->IsEnabled = false;
                s.Arity += it->Arity - 1;
                reduced = true;
            }
        }
    }

    // if anything was reduced (nodes were disabled), copy remaining enabled nodes
    if (reduced) {
        // erase-remove idiom https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
        nodes.erase(remove_if(nodes.begin(), nodes.end(), [](const Node& s) { return !s.IsEnabled; }), nodes.end());
    }
    // else, nothing to do
    return this->UpdateNodes();
}

// Sort each function node's children according to node type and hash value
// - note that entire child subtrees / subarrays are reordered inside the nodes array
// - this method assumes node hashes are computed, usually it is preceded by a call to tree.Hash()
Tree& Tree::Sort()
{
    // preallocate memory to reduce fragmentation
    Operon::Vector<Operon::Node> sorted = nodes;

    Operon::Vector<int> children;
    children.reserve(nodes.size());

    auto start = nodes.begin();

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& s = nodes[i];

        if (s.IsLeaf()) {
            continue;
        }

        auto arity = s.Arity;
        auto size = s.Length;

        if (s.IsCommutative()) {
            if (arity == size) {
                std::sort(start + i - size, start + i);
            } else {
                for (auto it = Children(i); it.HasNext(); ++it) {
                    children.push_back(it.Index());
                }
                std::sort(children.begin(), children.end(), [&](int a, int b) { return nodes[a] < nodes[b]; }); // sort child indices

                auto pos = sorted.begin() + i - size;
                for (auto j : children) {
                    auto& c = nodes[j];
                    std::copy_n(start + j - c.Length, c.Length + 1, pos);
                    pos += c.Length + 1;
                }
                children.clear();
            }
        }
    }
    nodes.swap(sorted);
    return this->UpdateNodes();
}

std::vector<gsl::index> Tree::ChildIndices(gsl::index i) const
{
    if (nodes[i].IsLeaf()) {
        return std::vector<gsl::index> {};
    }
    std::vector<gsl::index> indices(nodes[i].Arity);
    for (auto it = Children(i); it.HasNext(); ++it) {
        indices[it.Count()] = it.Index();
    }
    return indices;
}

std::vector<double> Tree::GetCoefficients() const
{
    std::vector<double> coefficients;
    for (auto& s : nodes) {
        if (s.IsConstant() || s.IsVariable()) {
            coefficients.push_back(s.Value);
        }
    }
    return coefficients;
}

void Tree::SetCoefficients(const gsl::span<const double> coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes) {
        if (s.IsLeaf()) {
            s.Value = coefficients[idx++];
        }
    }
}

size_t Tree::Depth() const noexcept
{
    return nodes.back().Depth;
}

size_t Tree::Depth(gsl::index i) const noexcept
{
    return nodes[i].Depth;
}

size_t Tree::VisitationLength() const noexcept
{
    return std::transform_reduce(std::execution::unseq, nodes.begin(), nodes.end(), 0UL, std::plus<> {}, [](const auto& node) { return node.Length + 1; });
}

// calculate the level in the tree (distance to tree root) for the subtree at index i
size_t Tree::Level(gsl::index i) const noexcept
{
    // the root node is always the last node with index Length() - 1
    gsl::index root = Length() - 1;

    size_t level = 0;
    while (i < root) {
        i = nodes[i].Parent;
        ++level;
    }
    return level;
}

Tree& Tree::Hash(Operon::HashFunction f, Operon::HashMode m)
{
    switch (f) {
    case Operon::HashFunction::XXHash: {
        return Hash<Operon::HashFunction::XXHash>(m);
    }
    case Operon::HashFunction::MetroHash: {
        return Hash<Operon::HashFunction::MetroHash>(m);
    }
    case Operon::HashFunction::FNV1Hash: {
        return Hash<Operon::HashFunction::FNV1Hash>(m);
    }
    case Operon::HashFunction::AquaHash: {
        return Hash<Operon::HashFunction::AquaHash>(m);
    }
    }
    return *this;
}

} // namespace Operon
