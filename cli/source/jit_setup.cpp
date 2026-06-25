// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#include "jit_setup.hpp"


#include "operator_factory.hpp" // ParseErrorMetric

#ifdef HAVE_ASMJIT
#include "operon/interpreter/backend/jit/jit_factory.hpp"
#endif

namespace Operon::CLI {

auto MakeJitObjects(
    std::string_view              jitMode,
    Operon::Problem&              problem,
    Operon::ScalarDispatch const& dtable,
    std::string const&            objective,
    bool                          linearScaling,
    int                           jitMaxLength,
    std::size_t                   jitMinVisits,
    int                           maxLength,
    std::size_t                   seed
) -> JitObjects {
#if !defined(HAVE_ASMJIT)
    fmt::print(stderr, "error: --jit requires a build with JIT support (HAVE_ASMJIT)\n");
    return JitObjects{.Error = true};
#else
    auto [metric, supportsLinearScale] = Operon::ParseErrorMetric(objective);
    auto j = Operon::JIT::MakeJitObjects(
        jitMode, problem, dtable, *metric, linearScaling && supportsLinearScale,
        maxLength, jitMaxLength, jitMinVisits, seed);
    return JitObjects{
        .Evaluator      = std::move(j.Evaluator),
        .OptimizerJacEval = std::move(j.OptimizerJacEval),
        .Optimizer      = std::move(j.Optimizer),
        .Zobrist        = std::move(j.Zobrist),
        .Report         = std::move(j.Report),
        .Error          = false
    };
#endif
}

} // namespace Operon::CLI
