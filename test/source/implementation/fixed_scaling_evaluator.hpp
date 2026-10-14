// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_TEST_FIXED_SCALING_EVALUATOR_HPP
#define OPERON_TEST_FIXED_SCALING_EVALUATOR_HPP

#include <atomic>
#include <optional>

#include "operon/core/dispatch.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"

namespace Operon::Test {

// The interpreter value evaluator, instrumented to count its prediction phase. `Predictions` counts the
// prediction transactions (Evaluate invocations) this evaluator ran, independently of any Jacobian work its Score
// may do (MDL) and of Problem-level counters.
class PredictionCountingEvaluator : public Evaluator<ScalarDispatch> {
public:
    using Evaluator<ScalarDispatch>::Evaluator;

    mutable std::atomic_size_t Predictions { 0 };

protected:
    auto Evaluate(Individual const& ind, Span<Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override
    {
        ++Predictions;
        return Evaluator<ScalarDispatch>::Evaluate(ind, buf);
    }
};

// A value evaluator that applies a caller-chosen transform instead of the OLS fit. Its transform deliberately
// differs from the tree fit, so a consumer that refits instead of using the exact transform the evaluator applied
// is observable. It counts its prediction phase like PredictionCountingEvaluator.
class FixedScalingEvaluator : public Evaluator<ScalarDispatch> {
public:
    FixedScalingEvaluator(Problem const* problem, ScalarDispatch const* dtable, std::optional<LinearScaling> scaling)
        : Evaluator<ScalarDispatch>(problem, dtable)
        , scaling_(scaling)
    {
    }

    mutable std::atomic_size_t Predictions { 0 };

protected:
    auto Evaluate(Individual const& ind, Span<Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override
    {
        ++Predictions;
        auto evaluated = Evaluator<ScalarDispatch>::Evaluate(ind, buf);
        if (!evaluated || !evaluated->has_value()) {
            return evaluated;
        }
        auto const values = (*evaluated)->Values(ind, buf);
        return std::optional<EvaluatedBuffer> { MarkEvaluated(ind, values, scaling_) };
    }

    auto Score(RandomGenerator& /*rng*/, Individual const& ind, Span<Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> ReturnType override
    {
        auto const* problem = GetProblem();
        auto const range = problem->TrainingRange();
        ENSURE(evaluated.has_value());
        auto values = evaluated->Values(ind, buf);
        if (scaling_) {
            scaling_->ApplyInPlace(values);
        }
        auto const fit = MSE {}(values, problem->TargetValues(range));
        return ReturnType { static_cast<Operon::Scalar>(fit) };
    }

private:
    std::optional<LinearScaling> scaling_;
};

// Phase hooks and the restoration hook are not reachable from outside the evaluator hierarchy and the
// non-Lamarckian lifecycle helper.
template <typename T>
concept CanCallEvaluatePhase = requires(T const& t, Individual const& ind, Span<Scalar> buf) { t.Evaluate(ind, buf); };

template <typename T>
concept CanCallScorePhase = requires(T const& t, RandomGenerator& rng, Individual const& ind, Span<Scalar> buf,
    std::optional<EvaluatedBuffer> e) { t.Score(rng, ind, buf, std::move(e)); };

template <typename T>
concept CanCallRestorationHook = requires(T const& t, Individual& ind, Span<Scalar const> stored, Span<Scalar> buf) {
    t.FinalizeRestoredCoefficients(ind, stored, buf);
};

} // namespace Operon::Test

#endif
