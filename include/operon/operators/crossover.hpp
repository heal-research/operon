// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CROSSOVER_HPP
#define OPERON_CROSSOVER_HPP

#include <cstddef>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

#include "operon/operon_export.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

namespace Operon {
// crossover takes two parent trees and returns a child
struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> {
    using Limits = std::pair<std::size_t, std::size_t>;
    static auto FindCompatibleSwapLocations(Operon::RandomGenerator& random, Tree const& lhs, Tree const& rhs, size_t maxDepth, size_t maxLength, double internalProbability = 1.0) -> std::pair<size_t, size_t>;
    static auto SelectRandomBranch(Operon::RandomGenerator& random, Tree const& tree, double internalProb, Limits length, Limits level, Limits depth) -> size_t;
    static auto Cross(const Tree& lhs, const Tree& rhs, /* index of subtree 1 */ size_t i, /* index of subtree 2 */ size_t j) -> Tree;
};

class OPERON_EXPORT SubtreeCrossover : public CrossoverBase {
public:
    SubtreeCrossover(double internalProbability, size_t maxDepth, size_t maxLength)
        : internalProbability_(internalProbability)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    { }

    auto operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> Tree override;

    [[nodiscard]] auto InternalProbability() const -> double { return internalProbability_; }
    [[nodiscard]] auto MaxDepth() const -> size_t { return maxDepth_; }
    [[nodiscard]] auto MaxLength() const -> size_t { return maxLength_; }

private:
    double internalProbability_;
    size_t maxDepth_;
    size_t maxLength_;
};

class OPERON_EXPORT TranspositionAwareCrossover : public CrossoverBase {
public:
    TranspositionAwareCrossover(double internalProbability, size_t maxDepth, size_t maxLength)
        : internalProbability_(internalProbability)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    { }

    auto operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> Tree override;

    [[nodiscard]] auto InternalProbability() const -> double { return internalProbability_; }
    [[nodiscard]] auto MaxDepth() const -> size_t { return maxDepth_; }
    [[nodiscard]] auto MaxLength() const -> size_t { return maxLength_; }

private:
    double internalProbability_;
    size_t maxDepth_;
    size_t maxLength_;
};

} // namespace Operon
#endif
