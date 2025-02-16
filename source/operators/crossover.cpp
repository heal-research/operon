// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/core.h>
#include <random>

#include "operon/core/contracts.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/random/random.hpp"

namespace Operon {
namespace {
    using Limits = std::pair<std::size_t, std::size_t>;
    auto NotIn(Limits t, size_t v) -> bool {
        auto [a, b] = t;
        return v < a || b < v;
    }
} // namespace

auto CrossoverBase::FindCompatibleSwapLocations(Operon::RandomGenerator& random, Tree const& lhs, Tree const& rhs, size_t maxDepth, size_t maxLength, double internalProbability) -> std::pair<size_t, size_t>
{
    using Signed = std::make_signed_t<size_t>;
    auto diff = static_cast<Signed>(lhs.Length() - maxLength + 1); // +1 to account for at least one node that gets swapped in

    auto i = SelectRandomBranch(random, lhs, internalProbability, Limits{std::max(diff, Signed{1}), lhs.Length()}, Limits{size_t{1}, lhs.Depth()}, Limits{size_t{1}, lhs.Depth()});
    // we have to make some small allowances here due to the fact that the provided trees
    // might actually be larger than the maxDepth and maxLength limits given here
    auto maxBranchDepth = static_cast<Signed>(maxDepth - lhs[i].Level);
    maxBranchDepth = std::max(maxBranchDepth, Signed{1});

    auto partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
    auto maxBranchLength = static_cast<Signed>(maxLength - partialTreeLength);
    maxBranchLength = std::max(maxBranchLength, Signed{1});

    auto j = SelectRandomBranch(random, rhs, internalProbability, Limits{1UL, maxBranchLength}, Limits{1UL, rhs.Depth()}, Limits{1UL, maxBranchDepth});
    return std::make_pair(i, j);
}

auto CrossoverBase::SelectRandomBranch(Operon::RandomGenerator& random, Tree const& tree, double internalProb, Limits length, Limits level, Limits depth) -> size_t
{
    if (tree.Length() == 1) {
        return 0;
    }

    auto const& nodes = tree.Nodes();

    std::vector<size_t> candidates(nodes.size());
    auto head = candidates.begin();
    auto tail = candidates.rbegin();

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto const& node = nodes[i];

        auto l = node.Length + 1U;
        auto d = node.Depth;
        auto v = node.Level;

        if (NotIn(length, l) || NotIn(level, v) || NotIn(depth, d)) {
            continue;
        }

        if (node.IsLeaf()) {
            *head++ = i;
        } else {
            *tail++ = i;
        }
    }

    // check if we have any function node candidates at all and if the bernoulli trial succeeds
    if (tail > candidates.rbegin() && std::bernoulli_distribution(internalProb)(random)) {
        return *Operon::Random::Sample(random, candidates.rbegin(), tail);
    }
    return *Operon::Random::Sample(random, candidates.begin(), head);
}


auto CrossoverBase::Cross(const Tree& lhs, const Tree& rhs, /* index of subtree 1 */ size_t i, /* index of subtree 2 */ size_t j) -> Tree
{
    auto const& left = lhs.Nodes();
    auto const& right = rhs.Nodes();
    Operon::Vector<Node> nodes;
    using Signed = std::make_signed_t<size_t>;
    nodes.reserve(right[j].Length - left[i].Length + left.size());
    std::copy_n(left.begin(), i - left[i].Length, back_inserter(nodes));
    std::copy_n(right.begin() + static_cast<Signed>(j) - right[j].Length, right[j].Length + 1, back_inserter(nodes));
    std::copy_n(left.begin() + static_cast<Signed>(i) + 1, left.size() - (i + 1), back_inserter(nodes));

    auto child = Tree(nodes).UpdateNodes();
    return child;
}

auto SubtreeCrossover::operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> Tree
{
    auto [i, j] = FindCompatibleSwapLocations(random, lhs, rhs, maxDepth_, maxLength_, internalProbability_);
    auto child = Cross(lhs, rhs, i, j);

    auto maxDepth{std::max(maxDepth_, lhs.Depth())};
    auto maxLength{std::max(maxLength_, lhs.Length())};

    ENSURE(child.Depth() <= maxDepth);
    ENSURE(child.Length() <= maxLength);

    return child;
}
} // namespace Operon

