// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifdef HAVE_ASMJIT

#include "operon/interpreter/backend/jit/jit_factory.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon::JIT {

auto MakeJitObjects(std::string_view mode, Operon::Problem& problem, Operon::ScalarDispatch const& dtable,
    Operon::ErrorMetric const& metric, int maxLength, int jitMaxLength, std::size_t jitMinVisits, std::size_t seed,
    std::size_t cacheMaxAge, std::optional<Operon::ShapeConstraintSpec> const& shape) -> JitObjects
{
    JitObjects out;

    Operon::RandomGenerator cacheRng(seed);
    auto jz = std::make_unique<JitZobrist>(cacheRng, maxLength, problem.GetInputs(), cacheMaxAge);
    auto* jzp = jz.get();
    out.Zobrist = std::move(jz);

    // The JitEvaluator subobject the optimizer and the report callback use. In mode "all" with a shape constraint it
    // is the base subobject of the constrained adapter, so the optimizer shares the evaluator's own code cache.
    JitEvaluator* jev = nullptr;
    if (mode == "all") {
        if (shape) {
            auto adapter = std::make_unique<ShapeConstrainedValueEvaluator<JitEvaluator>>(
                Operon::ShapeConstraintSpec { *shape }, &dtable, &problem, jzp, metric);
            jev = adapter.get();
            out.Evaluator = std::move(adapter);
        } else {
            auto plain = std::make_unique<JitEvaluator>(&problem, jzp, metric);
            jev = plain.get();
            out.Evaluator = std::move(plain);
        }
        out.Optimizer
            = std::make_unique<JitLevenbergMarquardtOptimizer<Operon::ScalarDispatch>>(&dtable, &problem, jev);
    } else if (mode == "jac") {
        // Evaluator stays null — caller creates the interpreter evaluator.
        auto jacEval = std::make_unique<JitEvaluator>(&problem, jzp, metric);
        jev = jacEval.get();
        out.OptimizerJacEval = std::move(jacEval);
        out.Optimizer = std::make_unique<JitLevenbergMarquardtOptimizer<Operon::ScalarDispatch, /*JacobianOnly=*/true>>(
            &dtable, &problem, jev);
    } else {
        fmt::print(stderr, "warning: unknown --jit mode '{}', ignoring\n", mode);
        return out; // null evaluator/optimizer — caller falls back to defaults
    }

    jev->SetMaxLength(jitMaxLength);
    jev->SetMinVisits(jitMinVisits);
    out.Report = [jev]() -> void {
        auto const hits = jev->CacheHits();
        auto const misses = jev->CacheMisses();
        auto const compiled = jev->CompileSuccesses();
        auto const total = hits + misses;
        auto const rate = total > 0U ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0;
        fmt::print(stderr, "jit | cache {:5} | hits {:6} | misses {:6} | compiled {:6} | hit% {:5.1f}\n",
            jev->CacheSize(), hits, misses, compiled, rate);
        jev->ResetCounters();
    };

    return out;
}

} // namespace Operon::JIT

#endif // HAVE_ASMJIT
