// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_SHAPE_FEASIBILITY_HPP
#define OPERON_ALGORITHMS_PROBES_SHAPE_FEASIBILITY_HPP

#include <cstdint>
#include <cstddef>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"

namespace Operon {

// Emits shape-constraint search-dynamics signals for hard-reject runs:
// feasible fractions in the current parent/offspring buffers, plus the
// ShapeConstrainedEvaluator's cumulative and per-report rejection counts.
class ShapeFeasibilityProbe final : public GenerationProbe {
public:
    auto operator()(ProbeContext& ctx) -> void override
    {
        auto const* eval = dynamic_cast<ShapeConstrainedEvaluator const*>(ctx.Algorithm().GetGenerator()->Evaluator());
        if (eval == nullptr) { return; }

        auto const parent = CountFeasible(ctx.Parents(), *eval);
        auto const offspring = CountFeasible(ctx.Offspring(), *eval);
        auto const total = eval->Violations();
        auto const calls = eval->CallCount.load();
        auto const delta = total - lastViolations_;
        auto const callDelta = calls - lastCalls_;
        lastViolations_ = total;
        lastCalls_ = calls;

        ctx.Emit("shape_parent_feasible", static_cast<std::int64_t>(parent.Feasible));
        ctx.Emit("shape_parent_total", static_cast<std::int64_t>(parent.Total));
        ctx.Emit("shape_parent_feasible_frac", parent.Fraction());
        ctx.Emit("shape_offspring_feasible", static_cast<std::int64_t>(offspring.Feasible));
        ctx.Emit("shape_offspring_total", static_cast<std::int64_t>(offspring.Total));
        ctx.Emit("shape_offspring_feasible_frac", offspring.Fraction());
        ctx.Emit("shape_rejections_total", static_cast<std::int64_t>(total));
        ctx.Emit("shape_rejections_delta", static_cast<std::int64_t>(delta));
        ctx.Emit("shape_eval_calls_total", static_cast<std::int64_t>(calls));
        ctx.Emit("shape_eval_calls_delta", static_cast<std::int64_t>(callDelta));
        ctx.Emit("shape_rejection_rate_total", calls == 0 ? 0.0 : static_cast<double>(total) / static_cast<double>(calls));
        ctx.Emit("shape_rejection_rate_delta", callDelta == 0 ? 0.0 : static_cast<double>(delta) / static_cast<double>(callDelta));
    }

private:
    struct Count {
        std::size_t Feasible{0};
        std::size_t Total{0};

        [[nodiscard]] auto Fraction() const -> double
        {
            return Total == 0 ? 0.0 : static_cast<double>(Feasible) / static_cast<double>(Total);
        }
    };

    static auto CountFeasible(Operon::Span<Operon::Individual const> pop, ShapeConstrainedEvaluator const& eval) -> Count
    {
        Count count{.Total = pop.size()};
        for (auto const& ind : pop) {
            if (!ind.Genotype.Empty() && eval.Feasible(ind.Genotype)) { ++count.Feasible; }
        }
        return count;
    }

    std::size_t lastViolations_{0};
    std::size_t lastCalls_{0};
};

} // namespace Operon

#endif
