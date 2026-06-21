// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon::CLI {

// CLI-level wrapper around Operon::JIT::JitObjects; adds Error for the no-ASMJIT case.
struct JitObjects {
    std::unique_ptr<Operon::EvaluatorBase> Evaluator;             // null in "jac" mode → caller calls ParseEvaluator
    std::unique_ptr<Operon::EvaluatorBase> OptimizerJacEval;  // non-null in "jac" mode, owns the JitEvaluator
    std::unique_ptr<Operon::OptimizerBase> Optimizer;
    std::unique_ptr<Operon::Zobrist>       Zobrist;        // JitZobrist; suitable for transposition cache too
    std::function<void()>                  Report = [](){};
    bool                                   Error  = false; // true → caller should return EXIT_FAILURE
};

// Resolve objective string → ErrorMetric, then delegate to Operon::JIT::MakeJitObjects.
// Returns .Error=true (with a message to stderr) when built without HAVE_ASMJIT.
// In "jac" mode, .Evaluator is null — caller must create the interpreter evaluator.
// .Zobrist is a JitZobrist; assign to config.Cache if --transposition-cache is set.
auto MakeJitObjects(
    std::string_view jitMode,
    Operon::Problem&            problem,
    Operon::ScalarDispatch const& dtable,
    std::string const&          objective,
    bool                        linearScaling,
    int                         jitMaxLength,
    std::size_t                 jitMinVisits,
    int                         maxLength,
    std::size_t                 seed
) -> JitObjects;

} // namespace Operon::CLI
