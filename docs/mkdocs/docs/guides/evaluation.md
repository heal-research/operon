# Evaluation

![Evaluation pipeline](../assets/diagrams/evaluation.svg){ .diagram }

## Two-phase evaluator contract

`EvaluatorBase::operator()` is final and always composes `Evaluate(individual, scratch)` followed by `Score(context, evaluated)`. Value-based evaluators override `Evaluate` to fill a caller-supplied scratch span and return an `EvaluatedBuffer`; objective-only evaluators may return `nullopt` and compute directly in `Score`.

`EvaluatedBuffer` is move-only evidence that a particular scratch span contains output for a particular `Individual`. A `Score` implementation must retrieve values through `evaluated->Values(ctx.Ind, ctx.Scratch)`, not read the scratch span directly. This catches a common composite-evaluator bug: scoring stale output from a different genotype.

| Event | Result |
| --- | --- |
| `Evaluate` succeeds | `Score` produces one or more minimization objectives |
| `Evaluate` returns an interpreter error | final `operator()` increments `CallCount` and returns `ErrMax` |
| no sampled values are required | `Evaluate` returns `nullopt`; `Score` owns the objective |

## Built-in objective building blocks

`Evaluator<DTable>` performs sampled interpretation and scores an `ErrorMetric` such as `SSE`, `MSE`, `NMSE`, `RMSE`, `MAE`, `R2`, or `C2`. `FitLeastSquares` supplies linear scaling coefficients. Information criteria and weighted complexity evaluators can be composed into multi-objective fitness vectors.

Every standard error metric follows the algorithm's minimization convention. If an external score has “larger is better” semantics, transform it before exposing it as an objective or supply a comparison consistent with it.

## Budgets and preparation

`ObjectiveCount()` tells population algorithms the required fitness-vector width. `Prepare(population)` is the per-generation hook for cached or population-level state; shape-constrained evaluation uses it to prewarm feasibility. `ResidualEvaluations`, `JacobianEvaluations`, and `CallCount` are atomic counters. A custom evaluator must account for its own `Score` call exactly once and should preserve the base failure behavior rather than translating execution errors into ordinary scores.
