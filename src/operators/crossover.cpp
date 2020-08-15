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
static gsl::index SelectRandomBranch(Operon::Random& random, const Tree& tree, double internalProb, size_t maxBranchLevel, size_t maxBranchDepth, size_t maxBranchLength)
{
    auto const& nodes = tree.Nodes();

    std::vector<size_t> candidates(nodes.size());

    size_t nLeaf = 0,
           nFunc = 0;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto const& node = nodes[i];

        if (node.Length + 1u > maxBranchLength || node.Depth > maxBranchDepth || tree.Level(i) > maxBranchLevel) {
            continue;
        }

        if (node.IsLeaf()) {
            candidates[nLeaf] = i;
            ++nLeaf;
        } else {
            ++nFunc;
            candidates[nodes.size()-nFunc] = i;
        }
    }
    using dist = std::uniform_int_distribution<size_t>;

    if (nLeaf > 0 && nFunc > 0)
        return std::bernoulli_distribution(internalProb)(random)
            ? candidates[dist(candidates.size()-nFunc, candidates.size()-1)(random)]
            : candidates[dist(0, nLeaf-1)(random)];
    
    if (nLeaf > 0)
        return candidates[dist(0, nLeaf-1)(random)];

    if (nFunc > 0)
        return candidates[dist(candidates.size()-nFunc, candidates.size()-1)(random)];

    throw std::runtime_error(fmt::format("Could not find suitable candidate for: max branch level = {}, max branch depth = {}, max branch length = {}\n", maxBranchLevel, maxBranchDepth, maxBranchLength));
}

std::pair<gsl::index, gsl::index> SubtreeCrossover::FindCompatibleSwapLocations(Operon::Random& random, const Tree& lhs, const Tree& rhs) const
{
    auto i = SelectRandomBranch(random, lhs, internalProbability, maxDepth, std::numeric_limits<size_t>::max(), maxLength);
    size_t partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
    // we have to make some small allowances here due to the fact that the provided trees 
    // might actually be larger than the maxDepth and maxLength limits given here
    size_t maxBranchDepth    = std::max(maxDepth - lhs.Level(i), 1ul);
    size_t maxBranchLength   = std::max(maxLength - partialTreeLength, 1ul);

    auto j = SelectRandomBranch(random, rhs, internalProbability, std::numeric_limits<size_t>::max(), maxBranchDepth, maxBranchLength);
    return std::make_pair(i, j);
}

Tree SubtreeCrossover::operator()(Operon::Random& random, const Tree& lhs, const Tree& rhs) const
{
    auto [i, j] = FindCompatibleSwapLocations(random, lhs, rhs);

    auto child = Cross(lhs, rhs, i, j);
    auto md    = std::max(maxDepth, lhs.Depth());
    auto ml    = std::max(maxLength, lhs.Length());

    ENSURE(child.Depth() <= md);
    ENSURE(child.Length() <= ml);

    return child;
}
}
