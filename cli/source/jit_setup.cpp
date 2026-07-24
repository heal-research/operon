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
    [[maybe_unused]] std::string_view              jitMode,
    [[maybe_unused]] Operon::Problem&              problem,
    [[maybe_unused]] Operon::ScalarDispatch const& dtable,
    [[maybe_unused]] std::string const&            objective,
    [[maybe_unused]] bool                          linearScaling,
    [[maybe_unused]] int                           jitMaxLength,
    [[maybe_unused]] std::size_t                   jitMinVisits,
    [[maybe_unused]] int                           maxLength,
    [[maybe_unused]] std::size_t                   seed,
    [[maybe_unused]] std::size_t                   cacheMaxAge
) -> JitObjects {
#if !defined(HAVE_ASMJIT)
    fmt::print(stderr, "error: --jit requires a build with JIT support (HAVE_ASMJIT)\n");
    return JitObjects{.Evaluator = nullptr, .OptimizerJacEval = nullptr, .Optimizer = nullptr, .Zobrist = nullptr, .Report = [](){}, .Error = true};
#else
    auto [metric, supportsLinearScale] = Operon::ParseErrorMetric(objective);
    auto j = Operon::JIT::MakeJitObjects(
        jitMode, problem, dtable, *metric, linearScaling && supportsLinearScale,
        maxLength, jitMaxLength, jitMinVisits, seed, cacheMaxAge);
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
