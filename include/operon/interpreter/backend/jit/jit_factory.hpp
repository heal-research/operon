// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
#pragma once

#ifdef HAVE_ASMJIT

#include <functional>
#include <memory>
#include <string_view>

#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/operon_export.hpp"

#include "jit_evaluator.hpp"

namespace Operon::JIT {

// Owns all JIT-related objects for a single run (evaluator, optimizer, Zobrist cache, report callback).
// Evaluator is null in "jac" mode — the caller supplies an interpreter evaluator.
// Optimizer is null for unrecognised modes — the caller falls back to a default.
// Zobrist is always a JitZobrist; assign to GeneticAlgorithmConfig::Cache when a
// transposition table is also needed.
struct JitObjects {
    std::unique_ptr<Operon::EvaluatorBase>  Evaluator;
    std::unique_ptr<Operon::EvaluatorBase>  JitEvalForOptimizer;
    std::unique_ptr<Operon::OptimizerBase>  Optimizer;
    std::unique_ptr<JitZobrist>             Zobrist;
    std::function<void()>                   Report = [](){};
};

// Create JIT-backed evaluator/optimizer for mode "all" or "jac".
// metric and linearScaling are already resolved by the caller (no string parsing here).
// maxLength is the tree size used to size the Zobrist table; jitMaxLength/jitMinVisits
// gate per-tree compilation.
OPERON_EXPORT auto MakeJitObjects(
    std::string_view          mode,
    Operon::Problem&          problem,
    Operon::ScalarDispatch const& dtable,
    Operon::ErrorMetric const& metric,
    bool                      linearScaling,
    int                       maxLength,
    int                       jitMaxLength,
    std::size_t               jitMinVisits,
    std::size_t               seed
) -> JitObjects;

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
