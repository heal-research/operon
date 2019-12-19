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

#include "operators/crossover.hpp"

namespace Operon {
static gsl::index SelectRandomBranch(Operon::Random& random, const Tree& tree, double internalProb, size_t maxBranchDepth, size_t maxBranchLength)
{
    auto selectInternalNode = std::bernoulli_distribution(internalProb)(random);
    const auto& nodes = tree.Nodes();
    // create a vector of indices where leafs are in the front and internal nodes in the back
    std::vector<gsl::index> indices(nodes.size());
    size_t head = 0;
    size_t tail = nodes.size() - 1;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        auto idx = node.IsLeaf() ? head++ : tail--;
        indices[idx] = i;
    }
    // if we want an internal node, we shuffle the corresponding part of the indices vector
    // then we walk over it and return the first index that satisfies the constraints
    if (selectInternalNode) {
        std::shuffle(indices.begin() + head, indices.end(), random);
        for (size_t i = head; i < indices.size(); ++i) {
            auto idx = indices[i];
            const auto& node = nodes[idx];

            if (node.Length + 1u > maxBranchLength || node.Depth > maxBranchDepth) {
                continue;
            }

            return idx;
        }
    }
    // if we couldn't find a suitable internal node or just wanted a leaf, fallback here
    std::uniform_int_distribution<size_t> uniformInt(0, head - 1);
    auto idx = uniformInt(random);
    return indices[idx];
}

std::pair<gsl::index, gsl::index> SubtreeCrossover::FindCompatibleSwapLocations(Operon::Random& random, const Tree& lhs, const Tree& rhs) const
{
    auto i = SelectRandomBranch(random, lhs, internalProbability, maxDepth, maxLength);
    size_t maxBranchDepth    = maxDepth - lhs.Level(i);
    size_t partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
    size_t maxBranchLength   = maxLength - partialTreeLength;

    auto j = SelectRandomBranch(random, rhs, internalProbability, maxBranchDepth, maxBranchLength);
    return std::make_pair(i, j);
}

Tree SubtreeCrossover::operator()(Operon::Random& random, const Tree& lhs, const Tree& rhs) const
{
    auto [i, j] = FindCompatibleSwapLocations(random, lhs, rhs);
    return Cross(lhs, rhs, i, j);
}
}
