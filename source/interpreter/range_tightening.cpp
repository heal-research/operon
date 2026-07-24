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

    auto EvaluateGradientColumn(
        VariableGradientDag const& gdag, std::size_t root,
        IntervalEvaluator::DomainMap const& domains, Operon::Span<Operon::Scalar const> coeff
    ) -> Interval
    {
        Operon::Vector<Node> subnodes(
            gdag.Nodes.cbegin(), gdag.Nodes.cbegin() + static_cast<std::ptrdiff_t>(root) + 1);
        Tree const gradTree{std::move(subnodes)};
        return IntervalEvaluator(&gradTree, domains).Evaluate(coeff);
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

        auto const gradInterval = EvaluateGradientColumn(gdag, root, domains, coeff);
        auto const xkMinusM = pappus::ops::sub<Scalar>(
            pappus::ops::variable<Scalar>(lo, hi), pappus::ops::constant<Scalar>(m));
        meanValue = pappus::ops::add<Scalar>(meanValue, pappus::ops::mul<Scalar>(gradInterval, xkMinusM));
    }

    if (meanValue.is_empty()) { return naive; }
    return naive & meanValue;
}

auto TightenRangeBisected(
    Tree const& tree,
    IntervalEvaluator::DomainMap domains,
    Operon::Span<Operon::Scalar const> coeff,
    int maxDepth
) -> Interval
{
    auto const result = TightenRange(tree, domains, coeff);
    if (maxDepth <= 0 || result.is_empty()) { return result; }

    auto const gdag = BuildVariableGradientDag(tree, coeff);
    if (gdag.Variables.empty()) { return result; }

    // Pick the variable whose gradient interval straddles zero with the
    // largest diameter: it's both sign-ambiguous (mean-value form is
    // loosest there) and contributes the most to that ambiguity.
    Operon::Hash splitVar{};
    Scalar bestDiameter{0};
    bool found = false;
    for (std::size_t k = 0; k < gdag.Variables.size(); ++k) {
        auto const root = gdag.Roots[k];
        if (root == NoGrad) { continue; }
        auto const gradInterval = EvaluateGradientColumn(gdag, root, domains, coeff);
        if (!gradInterval.contains(Scalar{0})) { continue; }
        auto const d = gradInterval.diameter();
        if (d > bestDiameter) { bestDiameter = d; splitVar = gdag.Variables[k]; found = true; }
    }
    if (!found) { return result; } // every gradient is sign-definite already

    auto const dit = domains.find(splitVar);
    if (dit == domains.end()) { return result; }
    auto const [lo, hi] = dit->second;
    auto const mid = Interval{lo, hi}.mid();

    auto leftDomains = domains;
    leftDomains[splitVar] = {lo, mid};
    auto rightDomains = domains;
    rightDomains[splitVar] = {mid, hi};

    auto const left  = TightenRangeBisected(tree, leftDomains, coeff, maxDepth - 1);
    auto const right = TightenRangeBisected(tree, rightDomains, coeff, maxDepth - 1);
    auto const unioned = left | right;

    if (unioned.is_empty()) { return result; }
    return result & unioned;
}

} // namespace Operon
