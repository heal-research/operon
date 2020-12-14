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
#include <random>

namespace Operon {

namespace {
    using Limits = std::pair<size_t, size_t>;

    bool not_in(Limits t, size_t v) {
        auto [a, b] = t;
        return v < a || b < v;
    }
}

static size_t SelectRandomBranch(Operon::RandomGenerator& random, Tree const& tree, double internalProb, Limits length, Limits level, Limits depth)
{
    if (tree.Length() == 1) {
        return 0;
    }

    auto const& nodes = tree.Nodes();

    std::vector<size_t> candidates(nodes.size());
    auto a = candidates.begin();
    auto b = candidates.rbegin();

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto const& node = nodes[i];

        auto l = node.Length + 1u;
        auto d = node.Depth;
        auto v = node.Level;

        if (not_in(length, l) || not_in(level, v) || not_in(depth, d)) {
            continue;
        }

        if (node.IsLeaf()) {
            *a++ = i;
        } else {
            *b++ = i;
        }
    }

    if (b > candidates.rbegin() && std::bernoulli_distribution(internalProb)(random)) {
        return *Operon::Random::Sample(random, candidates.rbegin(), b);
    } else {
        return *Operon::Random::Sample(random, candidates.begin(), a);
    }
}

std::pair<size_t, size_t> SubtreeCrossover::FindCompatibleSwapLocations(Operon::RandomGenerator& random, Tree const& lhs, Tree const& rhs) const
{
    using signed_t = std::make_signed<size_t>::type;
    signed_t diff = static_cast<signed_t>(lhs.Length() - maxLength + 1); // +1 to account for at least one node that gets swapped in

    auto i = SelectRandomBranch(random, lhs, internalProbability, Limits{std::max(diff, signed_t{1}), lhs.Length()}, Limits{size_t{1}, lhs.Depth()}, Limits{size_t{1}, lhs.Depth()});
    // we have to make some small allowances here due to the fact that the provided trees 
    // might actually be larger than the maxDepth and maxLength limits given here
    signed_t maxBranchDepth = static_cast<signed_t>(maxDepth - lhs[i].Level);
    maxBranchDepth = std::max(maxBranchDepth, signed_t{1});

    size_t partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
    signed_t maxBranchLength = static_cast<signed_t>(maxLength - partialTreeLength);
    maxBranchLength = std::max(maxBranchLength, signed_t{1});

    auto j = SelectRandomBranch(random, rhs, internalProbability, Limits{1ul, maxBranchLength}, Limits{1ul, rhs.Depth()}, Limits{1ul, maxBranchDepth});
    return std::make_pair(i, j);
}

Tree SubtreeCrossover::operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const
{
    auto [i, j] = FindCompatibleSwapLocations(random, lhs, rhs);

    auto child = Cross(lhs, rhs, i, j);

    ENSURE(child.Depth() <= std::max(maxDepth, lhs.Depth()));
    ENSURE(child.Length() <= std::max(maxLength, lhs.Length()));

    return child;
}
}
