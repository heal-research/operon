// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifdef HAVE_ASMJIT


#include "operon/interpreter/backend/jit/jit_factory.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon::JIT {

auto MakeJitObjects(
    std::string_view              mode,
    Operon::Problem&              problem,
    Operon::ScalarDispatch const& dtable,
    Operon::ErrorMetric const&    metric,
    bool                          linearScaling,
    int                           maxLength,
    int                           jitMaxLength,
    std::size_t                   jitMinVisits,
    std::size_t                   seed,
    std::size_t                   cacheMaxAge
) -> JitObjects {
    JitObjects out;

    Operon::RandomGenerator cacheRng(seed);
    auto jz   = std::make_unique<JitZobrist>(cacheRng, maxLength, problem.GetInputs(), cacheMaxAge);
    auto* jzp = jz.get();
    out.Zobrist = std::move(jz);

    if (mode == "all") {
        out.Evaluator = std::make_unique<JitEvaluator>(&problem, jzp, metric, linearScaling);
        out.Optimizer = std::make_unique<JitLevenbergMarquardtOptimizer<Operon::ScalarDispatch>>(
            &dtable, &problem,
            static_cast<JitEvaluator*>(out.Evaluator.get())); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    } else if (mode == "jac") {
        // Evaluator stays null — caller creates the interpreter evaluator.
        out.OptimizerJacEval = std::make_unique<JitEvaluator>(&problem, jzp, metric, linearScaling);
        out.Optimizer = std::make_unique<JitLevenbergMarquardtOptimizer<Operon::ScalarDispatch,
                                                                         Operon::OptimizerType::Eigen,
                                                                         /*JacobianOnly=*/true>>(
            &dtable, &problem,
            static_cast<JitEvaluator*>(out.OptimizerJacEval.get())); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    } else {
        fmt::print(stderr, "warning: unknown --jit mode '{}', ignoring\n", mode);
        return out; // null evaluator/optimizer — caller falls back to defaults
    }

    auto* jev = static_cast<JitEvaluator*>(out.Evaluator.get()); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    if (jev == nullptr) { jev = static_cast<JitEvaluator*>(out.OptimizerJacEval.get()); } // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    if (jev != nullptr) {
        jev->SetMaxLength(jitMaxLength);
        jev->SetMinVisits(jitMinVisits);
        out.Report = [jev]() -> void {
            auto const hits   = jev->CacheHits();
            auto const misses = jev->CacheMisses();
            auto const total  = hits + misses;
            auto const rate   = total > 0U ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0;
            fmt::print(stderr, "jit | cache {:5} | hits {:6} | misses {:6} | hit% {:5.1f}\n",
                       jev->CacheSize(), hits, misses, rate);
            jev->ResetCounters();
        };
    }

    return out;
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
