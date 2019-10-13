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
#include <iostream>
#include <iterator>
#include <optional>
#include <stack>
#include <utility>

#include "core/common.hpp"
#include "core/tree.hpp"

using namespace std;

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

Tree& Tree::Sort(bool strict)
{
    // preallocate memory to reduce fragmentation
    vector<Node> sorted;
    sorted.reserve(nodes.size());

    vector<int> children;
    children.reserve(nodes.size());

    vector<operon::hash_t> hashes;
    hashes.reserve(nodes.size());

    auto start = nodes.begin();
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& s = nodes[i];

        if (s.IsConstant()) {
            continue;
        }

        if (s.IsVariable()) {
            if (strict) {
                auto weightHash = xxh::xxhash<64>({ s.Value });
                s.CalculatedHashValue = xxh::xxhash<64>({ s.HashValue, weightHash });
                continue;
            } else {
                s.CalculatedHashValue = s.HashValue;
            }
            continue;
        }

        auto arity = s.Arity;
        auto size = s.Length;
        auto sBegin = start + i - size;
        auto sEnd = start + i;

        if (s.IsCommutative()) {
            if (arity == size) {
                sort(sBegin, sEnd);
            } else {
                for (auto it = Children(i); it.HasNext(); ++it) {
                    children.push_back(it.Index());
                }
                sort(children.begin(), children.end(), [&](int a, int b) { return nodes[a] < nodes[b]; }); // sort child indices

                for (auto j : children) {
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
    return this->UpdateNodes();
}

vector<gsl::index> Tree::ChildIndices(gsl::index i) const
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

void Tree::SetCoefficients(const std::vector<double>& coefficients)
{
    size_t idx = 0;
    for (auto& s : nodes) {
        if (s.IsConstant() || s.IsVariable()) {
            s.Value = coefficients[idx++];
        }
    }
}

size_t Tree::Depth() const noexcept
{
    return nodes.back().Depth;
}

// calculate the level in the tree (distance to tree root) for the subtree at index i
size_t Tree::Level(gsl::index i) const noexcept
{
    // the root node is always the last node with index Length() - 1
    gsl::index root = Length() - 1;

    size_t level = 0;
    while (i < root && nodes[i].Parent != root) {
        i = nodes[i].Parent;
        ++level;
    }
    return level;
}
}
