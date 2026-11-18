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

The adapter must be the evaluator handed to the algorithm: `MultiEvaluator::Add` rejects it with
`std::invalid_argument`. It already includes its structural objectives (`ShapeConstraintSpec::Structural`), so it needs
no composition. Non-Lamarckian restoration only restores the stored coefficients; it does not re-certify them.

## Built-in objective building blocks

`Evaluator<DTable>` performs sampled interpretation and scores an `ErrorMetric` such as `SSE`, `MSE`, `NMSE`, `RMSE`, `MAE`, `R2`, or `C2`. `FitLeastSquares` supplies linear scaling coefficients. Information criteria and weighted complexity evaluators can be composed into multi-objective fitness vectors.

Every standard error metric follows the algorithm's minimization convention. If an external score has “larger is better” semantics, transform it before exposing it as an objective or supply a comparison consistent with it.

`Evaluator` can omit non-finite prediction/target pairs during scoring. Every built-in error metric, including `R2`
and `C2`, is computed over that common finite subset; omitted rows add a configurable penalty. The `R2` and `C2`
penalty is dimensionless, while residual metrics scale it to their error units.

`ObjectiveCount()` tells population algorithms the required fitness-vector width. `Prepare(population)` is the per-generation hook for cached or population-level state; the shape-constrained adapter prepares its base and structural evaluators and keeps no certification cache. `ResidualEvaluations`, `JacobianEvaluations`, and `CallCount` are atomic counters; `operator()` increments `CallCount` exactly once per call, including a failed one, so a custom `Score` must not increment it again. A custom evaluator should preserve the base failure behavior rather than translating execution errors into ordinary scores.

## Evaluation accounting

Evaluation work has two owners, and the two are never mixed.

- **Evaluators own scoring work only.** `Stats()` returns `(residual evaluations, Jacobian evaluations, calls)` of the
  evaluator's own scoring: the prediction pass of `Evaluate`, a Jacobian taken by a likelihood scorer such as the MDL
  evaluator, and so on. `MultiEvaluator` and the shape-constrained adapter sum their children's scoring counters slot
  by slot (a child registered in two slots counts twice, as before); `Calls` is the composite's own. No evaluator
  holds optimizer counters or optimizer time: coefficient optimization is not something an evaluator performs.
- **The coefficient optimizer owns per-fit diagnostics.** The `FitOutcome` of `CoefficientOptimizer` (success,
  non-improving fit, or a typed evaluation/configuration failure) carries `FunctionEvaluations` and
  `JacobianEvaluations` for that one fit; `Diagnostics(outcome)` reads them whatever the outcome.
- **The run owns the reduction of optimizer work.** `LocalSearch` no longer takes an evaluator and touches no counter;
  it returns a `LocalSearchResult` (the outcome, the wall time of the optimizer call, and the coefficients to restore
  after a non-Lamarckian update). `OptimizerAccounting` (`local_search.hpp`) is the run-level accumulator: saturating
  residual, Jacobian and time counters, lock-free from any worker thread. `GeneticAlgorithmBase` owns one
  (`Accounting()`), resets it at the start of every `Run()`, and passes it explicitly to `ScoreIndividual`, to the
  offspring generators, and to the direct `LocalSearch` calls of cold initialization in GP and NSGA2. The accounting
  argument is required everywhere: there is no default, no nullable pointer and no thread-local "current run". Each
  invoked optimizer is recorded exactly once, before the candidate is scored, so a fit that did work stays charged
  when the candidate is later rejected by offspring selection, loses a brood or polygenic selection, fails scoring, or
  is discarded with an incomplete generation. A skipped search (`pLocal` trial), an absent optimizer, a zero-iteration
  optimizer and an error raised before anything was evaluated contribute zero; a transposition-cache hit never reaches
  local search and so costs no optimizer work. A custom `OffspringGeneratorBase` must forward the accounting it is
  given to `Generate`/`ScoreIndividual` (or to the generator it delegates to). An optimizer used directly, outside a
  run, is not attributed to any run.

`GeneticAlgorithmBase::DetailedStats()` is the combined run report, an `EvaluationStats` with disjoint fields:
`ResidualScoring`/`JacobianScoring`/`Calls` come from the evaluator's `Stats()`,
`ResidualOptimizer`/`JacobianOptimizer`/`OptimizerTimeUs` from the run's accounting, and `CacheHits`/`CacheLookups`
from the generator's active cache (`OffspringGeneratorBase::Cache()`: initially `config.Cache`, replaceable with
`SetCache`; `nullopt` when the generator has no active cache). Nothing is derived by subtracting one source from the
other. `Consumed()` is the evaluation budget unit, `ResidualTotal() + JacobianTotal()` (saturating); calls, optimizer
time and cache counters never consume the budget. A report read while workers run is a non-atomic snapshot across the
counters; one read after `Run()` returned is exact. Optimizer time is a wall-clock sum over threads: it is neither
deterministic nor thread-count independent.

The unit of the evaluation budget is scoring plus optimizer work, but what ends a run is
`GeneticAlgorithmConfig::Evaluations`, applied by the algorithm to committed work (see
[Evaluation budget and commit rule](../reference/search.md#evaluation-budget-and-commit-rule)). The algorithm sums the
cost of a generation from the evaluator tree's `Stats()` and the run's optimizer accounting after its workers finish.
The evaluator's own `Budget()`/`BudgetExhausted()` and `TotalEvaluations()` see scoring work only and are telemetry,
as is the deprecated `OffspringGeneratorBase::Terminate(OptimizerAccounting const&)`: none of them stops a run. What a
slot costs must be a function of the slot alone: a custom evaluator has to charge the base counters
deterministically (not depending on other slots or on arrival order), and an evaluator or cache whose cost depends on
what another slot did in the same generation (a transposition cache with `CacheVisibility::Immediate`, JIT compilation
gated by `--jit-min-visits` above 1) voids the thread-count independence of the commit decision. An evaluator that
charges nothing is simply not budgeted; a genotype-only objective that costs nothing to score still consumes budget
through the local search it triggers. Work done with the evaluator outside a generation (for example in a report
callback) is not committed.

The same cache has its generation clock advanced by `Run()`, so entry stamping, entry visibility (`CacheVisibility`)
and age-based expiry, the counters and the reset all refer to it. `EvaluatorBase::Reset()` is not virtual: it clears
the base counters and then calls the protected virtual `ResetChildren()`, which composite evaluators override to reset
their children (`MultiEvaluator`, the shape-constrained adapter). `GeneticAlgorithmBase::Reset()` (called at the start
of every `Run()`) calls it, zeroes the run's optimizer accounting and the budget report, and zeroes the active cache's
counters. Copying a quiescent algorithm copies its optimizer accounting and budget report with it.
