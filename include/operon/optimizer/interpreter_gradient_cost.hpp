// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERPRETER_GRADIENT_COST_HPP
#define OPERON_INTERPRETER_GRADIENT_COST_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "operon/core/memory_view.hpp"
#include "operon/core/range.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/gradient_cost.hpp"

namespace Operon::Concepts {

/**
 * What LBFGSOptimizer and SGDOptimizer require of a gradient cost beyond the
 * solver-facing GradientCost contract. The optimizers build the cost
 * themselves from the tree being fitted, so the cost must be:
 *
 *  - constructible from (interpreter, target, range, rng, batchSize, weights):
 *    target is the whole-dataset target column and range/weights are in the
 *    same absolute row coordinates; rng may be null only when batchSize is 0;
 *  - a counter of its own work: FunctionEvaluations() and
 *    JacobianEvaluations() return the number of cost and Jacobian
 *    evaluations performed, which the optimizers scale into FitDiagnostics;
 *  - explicit about dataset weights: a `static constexpr bool
 *    UsesDatasetWeights`. When true the optimizer validates the in-range
 *    slice of Dataset::Weights() and passes the whole column as the
 *    `weights` constructor argument; when false the cost receives an empty
 *    span and the dataset's sample weights are neither validated nor
 *    forwarded. This is how a cost with its own weighting semantics (e.g.
 *    Poisson exposure) stays independent of ordinary WLS sample weights.
 */
template <typename T>
concept InterpreterGradientCost = GradientCost<T>
    && std::constructible_from<T, InterpreterBase<Operon::Scalar> const*, ConstScalarSpan, Range, RandomGenerator*,
        std::size_t, ConstScalarSpan>
    && requires(T const& cost) {
           { cost.FunctionEvaluations() } -> std::same_as<std::size_t>;
           { cost.JacobianEvaluations() } -> std::same_as<std::size_t>;
           requires std::same_as<std::remove_cvref_t<decltype(T::UsesDatasetWeights)>, bool>;
           typename std::bool_constant<T::UsesDatasetWeights>;
       };

} // namespace Operon::Concepts

#endif
