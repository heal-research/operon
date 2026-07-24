// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/interpreter/range_tightening.hpp"

#include <limits>

#include "operon/core/tree_diff.hpp"

namespace Operon {

namespace {
    constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();
    using Scalar = Operon::Scalar;
    using Interval = IntervalEvaluator::Interval;

    // Mirrors Deriv()'s dispatch in tree_diff.cpp; must stay in sync with it.
    auto IsSymbolicallyDifferentiable(Node const& n) -> bool
    {
        if (n.IsAddition() || n.IsMultiplication() || n.IsSubtraction()
            || n.IsDivision() || n.IsPow()) {
            return true;
        }
        if (n.IsAq() || n.IsPowabs() || n.IsOp<BuiltinOp::Fmin, BuiltinOp::Fmax>()) {
            return false;
        }
        if (n.Arity == 1) { return HasUnarySymbolicDeriv(n.HashValue); }
        if (n.Arity == 2) { return HasBinarySymbolicDeriv(n.HashValue); }
        return true;
    }
} // namespace

auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff
) -> Interval
{
    auto const naive = IntervalEvaluator(&tree, domains).Evaluate(coeff);

    // A variable can occur multiple times with only some occurrences behind
    // an undifferentiated op (e.g. X + abs(X)); the root would then come
    // back nonzero but understate the true partial, so this is checked
    // structurally rather than via BuildVariableGradientDag's root value.
    for (auto const& n : tree.Nodes()) {
        if (n.IsLeaf() || n.IsRef()) { continue; }
        if (!IsSymbolicallyDifferentiable(n)) { return naive; }
    }

    auto const gdag = BuildVariableGradientDag(tree, coeff);
    if (gdag.Variables.empty()) { return naive; } // no input variables: naive is already exact

    // F(m) via a degenerate (lo == hi) domain map, reusing IntervalEvaluator.
    IntervalEvaluator::DomainMap midpoints;
    midpoints.reserve(domains.size());
    for (auto const& [hash, domain] : domains) {
        auto const m = Interval{domain.first, domain.second}.mid();
        midpoints.insert_or_assign(hash, IntervalEvaluator::Domain{m, m});
    }
    auto const fm = IntervalEvaluator(&tree, midpoints).Evaluate(coeff);

    auto meanValue = fm;
    for (std::size_t k = 0; k < gdag.Variables.size(); ++k) {
        auto const root = gdag.Roots[k];
        if (root == NoGrad) { continue; } // genuinely zero, given the pre-check above

        auto const hash = gdag.Variables[k];
        auto const dit  = domains.find(hash);
        if (dit == domains.end()) { return naive; } // defensive: naive would already have thrown
        auto const& [lo, hi] = dit->second;
        auto const m = Interval{lo, hi}.mid();

        Operon::Vector<Node> subnodes(
            gdag.Nodes.cbegin(), gdag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(root) + 1);
        Tree const gradTree{std::move(subnodes)};

        auto const gradInterval = IntervalEvaluator(&gradTree, domains).Evaluate(coeff);
        auto const xkMinusM = pappus::ops::sub<Scalar>(
            pappus::ops::variable<Scalar>(lo, hi), pappus::ops::constant<Scalar>(m));
        meanValue = pappus::ops::add<Scalar>(meanValue, pappus::ops::mul<Scalar>(gradInterval, xkMinusM));
    }

    if (meanValue.is_empty()) { return naive; }
    return naive & meanValue;
}

} // namespace Operon
