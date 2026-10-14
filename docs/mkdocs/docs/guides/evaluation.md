# Evaluation

![Evaluation pipeline](../assets/diagrams/evaluation.svg){ .diagram }

## Two-phase evaluator contract

`EvaluatorBase` is an `OperatorBase`: its only public execution path is
`operator()(RandomGenerator&, Individual const&, Span<Scalar>) -> Vector<Scalar>`, which is `final`. It performs, in
order, the protected `Evaluate(individual, scratch)` phase, failure handling, and the protected
`Score(rng, individual, scratch, evaluated)` phase, and it owns the single `CallCount` increment per call. Value-based
evaluators override `Evaluate` to fill a caller-supplied scratch span and return an `EvaluatedBuffer`; objective-only
evaluators leave it returning `nullopt` and compute directly in `Score`. Composition (`MultiEvaluator`, the structural
objectives of `ShapeConstrainedValueEvaluator`) invokes children only through their `operator()`.

`EvaluatedBuffer` is move-only evidence that a particular scratch span contains output for a particular `Individual`. A
`Score` implementation must retrieve values through `evaluated->Values(ind, buf)`, not read the scratch span directly.
This catches a common composite-evaluator bug: scoring stale output from a different genotype. The buffer is an
invocation-local artifact passed from `Evaluate` to `Score` and never escapes the operator call.

| Event | Result |
| --- | --- |
| `Evaluate` succeeds | `Score` produces one or more minimization objectives |
| `Evaluate` returns an interpreter error | `operator()` increments `CallCount` once and returns `ErrMax` repeated `ObjectiveCount()` times |
| no sampled values are required | `Evaluate` returns `nullopt`; `Score` owns the objective |

## Linear scaling policy

A value evaluator fits its linear scaling during `Evaluate` and stores the exact transform it will apply in the
`EvaluatedBuffer` (`std::optional<LinearScaling>`; `nullopt` means scaling is disabled). Its own `Score` consumes
that transform without refitting. The fit uses one declared policy,
`Problem::LinearScalingOmitsNonFinite()`, shared by the interpreter evaluator, the JIT evaluator, shape certification,
the Pareto-front exporter and reporting. An evaluator's `skipNonFinite` option is scoring-only; it is rejected at
construction unless linear scaling is disabled or the problem's fitting policy already omits non-finite rows.

`JitEvaluator` is an `Evaluator<ScalarDispatch>` that overrides only the prediction step (`Predict`); evaluation
counting, the scaling fit and scoring are the interpreter evaluator's own, so the two backends cannot drift.

## Shape constraints in one prediction pass

`ShapeConstrainedValueEvaluator<BaseEvaluator>` is a `final` evaluator derived from a concrete value evaluator
(`Concepts::ValueEvaluator`) and owns the complete constrained objective vector. Inside a single `operator()`
invocation the base evaluator's prediction phase runs once, the `ShapeCertifier` measures the model under the exact
transform that phase fitted (before the base evaluator's score rewrites the buffer), and the adapter enforces the
policy. See [Shape constraints](shape-constraints.md) and the [architecture review](evaluator-architecture-review.md).

The adapter must be the evaluator handed to the algorithm, so that the non-Lamarckian restoration step reaches it:
`MultiEvaluator::Add` rejects it with `std::invalid_argument`. It already includes its structural objectives
(`ShapeConstraintSpec::Structural`), so it needs no composition.

## Built-in objective building blocks

`Evaluator<DTable>` performs sampled interpretation and scores an `ErrorMetric` such as `SSE`, `MSE`, `NMSE`, `RMSE`, `MAE`, `R2`, or `C2`. `FitLeastSquares` supplies linear scaling coefficients. Information criteria and weighted complexity evaluators can be composed into multi-objective fitness vectors.

Every standard error metric follows the algorithm's minimization convention. If an external score has “larger is better” semantics, transform it before exposing it as an objective or supply a comparison consistent with it.

`Evaluator` can omit non-finite prediction/target pairs during scoring. Every built-in error metric, including `R2`
and `C2`, is computed over that common finite subset; omitted rows add a configurable penalty. The `R2` and `C2`
penalty is dimensionless, while residual metrics scale it to their error units.

`ObjectiveCount()` tells population algorithms the required fitness-vector width. `Prepare(population)` is the per-generation hook for cached or population-level state; the shape-constrained adapter keeps the published completions of the live population. `ResidualEvaluations`, `JacobianEvaluations`, and `CallCount` are atomic counters; `operator()` increments `CallCount` exactly once per call, including a failed one, so a custom `Score` must not increment it again. A custom evaluator should preserve the base failure behavior rather than translating execution errors into ordinary scores.
