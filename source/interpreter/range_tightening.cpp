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
} // namespace

auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff
) -> Interval
{
    auto const naive = IntervalEvaluator(&tree, domains).Evaluate(coeff);

    auto const gdag = BuildVariableGradientDag(tree, coeff);
    if (gdag.Variables.empty()) { return naive; } // no input variables: naive is already exact

    // F(m): evaluate the ORIGINAL tree at the domain box's midpoint, reusing
    // IntervalEvaluator with a degenerate (lo == hi) domain map rather than
    // building a separate point-evaluation path - the tiny width introduced
    // by outward-rounded interval ops on a degenerate input is negligible
    // and, being outward-rounded, still conservative.
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
        // NoGrad is Deriv()'s "zero" sentinel, but it's overloaded: it means
        // *either* a genuinely zero partial *or* "couldn't differentiate"
        // (an unregistered/excluded op such as Abs/Sqrtabs/Floor/Ceil, or
        // any binary/user op without a symbolic-diff rule, anywhere along
        // this variable's path). BuildJacobianDag can afford to conflate
        // these (a missing coefficient gradient just degrades optimization
        // quality), but treating "unknown" as "zero" here would silently
        // drop this variable's entire contribution from the mean-value
        // term while still intersecting the result with naive - that can
        // produce a bound *tighter than reality* (e.g. abs(x) on [-1,1]:
        // Abs has no symbolic-diff rule, so the gradient looks like zero,
        // giving mean-value = {F(0)} = {0} and excluding the true [0,1]).
        // Bail out to the naive-only enclosure instead of guessing.
        if (root == NoGrad) { return naive; }

        auto const hash = gdag.Variables[k];
        auto const dit  = domains.find(hash);
        // Every variable in gdag.Variables was already evaluated (with a
        // bound domain) by the naive IntervalEvaluator call above - it
        // walks every original node unconditionally, so a truly unbound
        // variable would have already thrown there. Bail the same way as
        // the NoGrad case above if this invariant is ever violated, rather
        // than silently treating the missing term as zero.
        if (dit == domains.end()) { return naive; }
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
