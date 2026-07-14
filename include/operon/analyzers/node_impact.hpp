// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ANALYZERS_NODE_IMPACT_HPP
#define OPERON_ANALYZERS_NODE_IMPACT_HPP

#include <algorithm>
#include <numeric>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/tree.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// Per-node impact, aligned 1:1 with tree.Nodes() (postfix index). For node i:
//   impact[i] = R2(tree) - R2(tree with node i's subtree spliced out and
//   replaced by a Constant equal to that subtree's own mean prediction over
//   `range`)
// This is the same "replace and re-measure" idea used for per-variable
// impact analysis (cf. HeuristicLab's RegressionSolutionVariableImpactsCalculator),
// generalized here from dataset columns to arbitrary tree nodes/subtrees.
// Positive impact means the subtree matters (removing it hurts the fit);
// negative means the subtree actively hurts this quality measure on `range`
// — a candidate for pruning. O(Length^2 * |range|): one interpreter pass per
// node, each over the full range.
//
// Returns an empty vector if the tree contains any Ref node: those are
// backward index references (DAG structural sharing), and splicing part of
// the tree out from under a Ref without also rewriting whichever RefTo
// values point through the spliced range isn't safe to do here.
inline auto NodeImpact(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Hash target, Operon::Range range) -> std::vector<double>
{
    using Interp = Operon::Interpreter<>;

    EXPECT(range.Size() > 0);

    if (std::ranges::any_of(tree.Nodes(), [](auto const& n) -> bool { return n.IsRef(); })) {
        return {};
    }

    // Sliced to `range`, matching what Evaluate() below actually returns -
    // dataset.GetValues(target) alone is the *whole* column, so comparing it
    // directly against a `range`-sized prediction silently misaligns rows
    // whenever range.Start() != 0.
    auto const actual = dataset.GetValues(target).subspan(range.Start(), range.Size());
    auto const predicted = Interp::Evaluate(tree, dataset, range);
    Operon::Span<Operon::Scalar const> const predictedSpan{ predicted.data(), predicted.size() };
    auto const baseline = Operon::R2Score(predictedSpan, actual);

    auto const& nodes = tree.Nodes();
    std::vector<double> impact(nodes.size(), 0.0);

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto const subtree = tree.Splice(i);
        auto const subtreeValues = Interp::Evaluate(subtree, dataset, range);
        EXPECT(!subtreeValues.empty()); // guaranteed by the range.Size() > 0 check above
        auto const mean = std::reduce(subtreeValues.begin(), subtreeValues.end(), Operon::Scalar{0}) / static_cast<Operon::Scalar>(subtreeValues.size());

        auto replacedNodes = nodes;
        auto const first = replacedNodes.begin() + static_cast<std::ptrdiff_t>(i - nodes[i].Length);
        auto const last = replacedNodes.begin() + static_cast<std::ptrdiff_t>(i) + 1;
        replacedNodes.erase(first, last);
        replacedNodes.insert(replacedNodes.begin() + static_cast<std::ptrdiff_t>(i - nodes[i].Length), Operon::Node::Constant(mean));

        auto replacedTree = Operon::Tree(std::move(replacedNodes)).UpdateNodes();
        auto const replacedPredicted = Interp::Evaluate(replacedTree, dataset, range);
        Operon::Span<Operon::Scalar const> const replacedSpan{ replacedPredicted.data(), replacedPredicted.size() };
        auto const replacedR2 = Operon::R2Score(replacedSpan, actual);

        impact[i] = baseline - replacedR2;
    }

    return impact;
}

// Whole-dataset convenience overload. This is post-hoc analysis on an
// already-fixed tree - unlike training, there's no leakage risk in reading
// test rows here, and using every row available gives a less noisy result
// than restricting to a training-only range. Pass an explicit range only if
// there's a specific reason to isolate a subset (e.g. comparing train vs.
// test impact).
inline auto NodeImpact(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Hash target) -> std::vector<double>
{
    return NodeImpact(tree, dataset, target, Operon::Range(0, dataset.Rows<std::size_t>()));
}

} // namespace Operon

#endif
