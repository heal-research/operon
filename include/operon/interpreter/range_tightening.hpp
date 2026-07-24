// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_RANGE_TIGHTENING_HPP
#define OPERON_RANGE_TIGHTENING_HPP

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Mean-value-form (first-order Taylor) enclosure of a tree's output range,
// intersected with IntervalEvaluator's naive enclosure of the same tree.
//
// For a midpoint m of the domain box and the interval-valued gradient
// ∇F([a,b]) (via BuildVariableGradientDag + IntervalEvaluator on each
// gradient column), the mean-value extension gives:
//
//   F([a,b]) ⊆ F(m) + ∇F([a,b]) · ([a,b] − m)
//
// which is generally tighter than the naive enclosure (it doesn't suffer
// naive interval arithmetic's dependency problem for a repeated variable
// the way IntervalEvaluator's direct walk does), but not always - so the
// result actually returned is the intersection of both, which is at least
// as tight as either one alone. `coeff` follows the same convention as
// IntervalEvaluator::Evaluate/Interpreter::Evaluate: one entry per node
// with Node::Optimize == true, consumed in node order.
//
// Domain-error policy: if evaluating the mean-value term hits a domain
// edge (e.g. a gradient column evaluates to IntervalEvaluator::Interval's
// empty()), the mean-value term is discarded and the naive enclosure is
// returned alone, rather than intersecting-to-empty and reporting an
// unsound/overly aggressive empty result. If the naive enclosure itself is
// empty, it stays empty (intersection with anything is still empty) — the
// function is genuinely undefined somewhere in the box, independent of
// this refinement.
OPERON_EXPORT auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff
) -> IntervalEvaluator::Interval;

} // namespace Operon

#endif
