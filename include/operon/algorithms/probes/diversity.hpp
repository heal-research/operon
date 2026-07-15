// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_DIVERSITY_HPP
#define OPERON_ALGORITHMS_PROBES_DIVERSITY_HPP

#include <algorithm>
#include <vector>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/core/constants.hpp"
#include "operon/core/distance.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {

namespace detail {
    // Tree::Hash() is const, so this works directly on ctx.Parents()'s const
    // individuals - no per-generation genotype copy needed.
    inline auto SortedNodeHashes(Tree const& tree, HashMode mode) -> Vector<Hash>
    {
        [[maybe_unused]] auto const& hashed = tree.Hash(mode);
        Vector<Hash> hashes(tree.Length());
        std::transform(tree.Nodes().begin(), tree.Nodes().end(), hashes.begin(),
            [](Node const& n) -> Hash { return n.CalculatedHashValue; });
        std::stable_sort(hashes.begin(), hashes.end());
        return hashes;
    }
    // Distance::Jaccard's CountIntersect dereferences one-before-the-start
    // of either span when it's empty (undefined behavior, not just a 0/0
    // NaN) - see source/core/distance.cpp. An empty hash vector means an
    // individual with an empty/default-constructed Genotype, which
    // shouldn't occur on a live algorithm's Parents() (TreeInitializer
    // always produces non-empty trees before any report callback runs) but
    // is cheap to guard against here rather than depend on that invariant
    // holding at every call site. Convention: two empty sets are identical
    // (distance 0); an empty set vs. a non-empty one is maximally
    // dissimilar (distance 1), matching the usual 0/0 := 1 similarity
    // convention for the Jaccard index.
    inline auto SafeJaccard(Vector<Hash> const& lhs, Vector<Hash> const& rhs) -> double
    {
        if (lhs.empty() && rhs.empty()) { return 0.0; }
        if (lhs.empty() || rhs.empty()) { return 1.0; }
        return Distance::Jaccard(lhs, rhs);
    }
} // namespace detail

// Mean pairwise Jaccard distance over each individual's sorted per-node hash
// set - salvaged from the dead operon/analyzers/diversity.hpp. O(pop^2 *
// length): schedule sparingly via ProbeChain's `every`.
[[nodiscard]] inline auto PopulationDiversity(Span<Individual const> pop, HashMode mode = HashMode::Strict) -> double
{
    if (pop.size() < 2) { return 0.0; }

    std::vector<Vector<Hash>> hashes;
    hashes.reserve(pop.size());
    for (auto const& ind : pop) { hashes.push_back(detail::SortedNodeHashes(ind.Genotype, mode)); }

    double sum{0};
    std::size_t count{0};
    for (std::size_t i = 0; i + 1 < pop.size(); ++i) {
        for (std::size_t j = i + 1; j < pop.size(); ++j) {
            sum += detail::SafeJaccard(hashes[i], hashes[j]);
            ++count;
        }
    }
    return sum / static_cast<double>(count);
}

class StructuralDiversityProbe final : public GenerationProbe {
public:
    explicit StructuralDiversityProbe(HashMode mode = HashMode::Strict) : mode_(mode) { }

    auto operator()(ProbeContext& ctx) -> void override
    {
        ctx.Emit("diversity_jaccard", PopulationDiversity(ctx.Parents(), mode_));
    }

private:
    HashMode mode_;
};

} // namespace Operon

#endif
