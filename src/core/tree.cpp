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
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <stack>
#include <utility>

#include "core/common.hpp"
#include "core/tree.hpp"


namespace Operon {
Tree& Tree::UpdateNodes()
{
    for (uint16_t i = 0; i < nodes.size(); ++i) {
        auto& s = nodes[i];

        s.Depth = 1;
        s.Length = s.Arity;
        if (s.IsLeaf()) {
            s.Arity = s.Length = 0;
            continue;
        }
        for (auto it = Children(i); it.HasNext(); ++it) {
            s.Length = static_cast<uint16_t>(s.Length + it->Length);
            s.Depth = std::max(s.Depth, it->Depth);
            nodes[it.Index()].Parent = i;
        }
        ++s.Depth;
    }
    nodes.back().Level = 1;

    for (auto it = nodes.rbegin() + 1; it < nodes.rend(); ++it) {
        it->Level = static_cast<uint16_t>(nodes[it->Parent].Level + 1);
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
                s.Arity = static_cast<uint16_t>(s.Arity + it->Arity - 1);
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

    Operon::Vector<size_t> children;
    children.reserve(nodes.size());

    auto start = nodes.begin();

    for (uint16_t i = 0; i < nodes.size(); ++i) {
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
                std::sort(children.begin(), children.end(), [&](auto a, auto b) { return nodes[a] < nodes[b]; }); // sort child indices

                auto pos = sorted.begin() + i - size;
                for (auto j : children) {
                    auto& c = nodes[j];
                    std::copy_n(start + gsl::narrow_cast<long>(j) - c.Length, c.Length + 1, pos);
                    pos += c.Length + 1;
                }
                children.clear();
            }
        }
    }
    nodes.swap(sorted);
    return this->UpdateNodes();
}

std::vector<size_t> Tree::ChildIndices(size_t i) const
{
    if (nodes[i].IsLeaf()) {
        return std::vector<size_t> {};
    }
    std::vector<size_t> indices(nodes[i].Arity);
    for (auto it = Children(i); it.HasNext(); ++it) {
        indices[it.Count()] = it.Index();
    }
    return indices;
}

std::vector<Operon::Scalar> Tree::GetCoefficients() const
{
    std::vector<Operon::Scalar> coefficients;
    for (auto& s : nodes) {
        if (s.IsConstant() || s.IsVariable()) {
            coefficients.push_back(s.Value);
        }
    }
    return coefficients;
}

void Tree::SetCoefficients(const gsl::span<const Operon::Scalar> coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes) {
        if (s.IsLeaf()) {
            s.Value = gsl::narrow_cast<Operon::Scalar>(coefficients[idx++]);
        }
    }
}

size_t Tree::Depth() const noexcept
{
    return nodes.back().Depth;
}

size_t Tree::VisitationLength() const noexcept
{
    return std::transform_reduce(nodes.begin(), nodes.end(), 0UL, std::plus<> {}, [](const auto& node) { return node.Length + 1; });
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
    }
    return *this;
}

} // namespace Operon
