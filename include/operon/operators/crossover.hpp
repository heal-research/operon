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
};

class OPERON_EXPORT SubtreeCrossover : public CrossoverBase {
public:
    SubtreeCrossover(double p, size_t d, size_t l)
        : internalProbability_(p)
        , maxDepth_(d)
        , maxLength_(l)
    {
    }
    auto operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> Tree override;
    auto FindCompatibleSwapLocations(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> std::pair<size_t, size_t>;

    static inline auto Cross(const Tree& lhs, const Tree& rhs, /* index of subtree 1 */ size_t i, /* index of subtree 2 */ size_t j) -> Tree
    {
        auto const& left = lhs.Nodes();
        auto const& right = rhs.Nodes();
        Operon::Vector<Node> nodes;
        using signed_t = std::make_signed<size_t>::type; // NOLINT
        nodes.reserve(right[j].Length - left[i].Length + left.size());
        std::copy_n(left.begin(), i - left[i].Length, back_inserter(nodes));
        std::copy_n(right.begin() + static_cast<signed_t>(j) - right[j].Length, right[j].Length + 1, back_inserter(nodes));
        std::copy_n(left.begin() + static_cast<signed_t>(i) + 1, left.size() - (i + 1), back_inserter(nodes));

        auto child = Tree(nodes).UpdateNodes();
        return child;
    }

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
