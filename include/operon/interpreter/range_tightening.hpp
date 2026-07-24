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
// intersected with IntervalEvaluator's naive enclosure of the same tree:
//
//   F([a,b]) ⊆ F(m) + ∇F([a,b]) · ([a,b] − m)
//
// `coeff` follows the same convention as IntervalEvaluator::Evaluate: one
// entry per node with Node::Optimize == true, consumed in node order.
//
// Falls back to the naive enclosure alone (no intersection) if the tree
// contains any op Deriv() can't symbolically differentiate, or if a
// gradient column evaluates to IntervalEvaluator::Interval's empty(). If
// naive itself is empty, it stays empty.
OPERON_EXPORT auto TightenRange(
    Tree const& tree,
    IntervalEvaluator::DomainMap const& domains,
    Operon::Span<Operon::Scalar const> coeff
) -> IntervalEvaluator::Interval;

} // namespace Operon

#endif
