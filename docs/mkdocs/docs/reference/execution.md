# Execution and evaluation API

## `Interpreter`

Header: [`operon/interpreter/interpreter.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/interpreter/interpreter.hpp)

```cpp
Interpreter<Scalar, ScalarDispatch> interpreter { &dispatch, &dataset, &tree };
auto values = interpreter.Evaluate(tree.GetCoefficients(), range);
Operon::Vector<Scalar> jacobian(range.Size() * tree.CoefficientsCount()); // column-major
auto filled = interpreter.JacRev(tree.GetCoefficients(), range, jacobian);
```

| Method family | Result |
| --- | --- |
| `Evaluate(coeff, range[, output])` | model values or `InterpreterError` |
| `JacRev` / `JacFwd` `(coeff, range, jacobian)` | fills the column-major coefficient Jacobian (`range.Size() * CoefficientsCount()` elements) or returns `InterpreterError`; rows are observations, columns are optimizable constants |
| `JacRevVariable` / `JacFwdVariable` | derivative of output with respect to one input variable hash, or `InterpreterError` |

All fallible `Interpreter` operations return `tl::expected`; inspect the result before using its value.

Empty `coeff` means the tree's own coefficients. The interpreter interface exposes no Eigen types: the Jacobian and `EvaluateRoots` results are written into caller-provided spans, so the Eigen-free guarantee of the signatures does not depend on how the interpreter computes them internally.

This API change is a deliberate breaking boundary for the next Operon API epoch: `InterpreterBase` virtual return types use `tl::expected`, and the former virtual `Try*` methods are removed. Downstream bindings and consumers must rebuild and migrate together; binaries built against the previous virtual interface are not ABI-compatible.

The interpreter interface in this release also drops the following. `Interpreter::Primal()` and `Trace()`, and public access to its scratch state and trace helpers, are removed. `DispatchTable::Get<T>`, `GetFunction`, `GetDerivative` and `TryGetFunction`/`TryGetDerivative` are removed; use `FindFunction<T>`/`FindDerivative<T>`, which return a pointer to a usable callable, or null if the hash is unregistered or its slot for `T` is empty (check before dereferencing; the pointer is valid until the table is modified, and `Contains(hash)` tells the two cases apart). `Interpreter::BindTree` takes no range and is the way to refresh a binding after the tree, dataset storage or dispatch table changed. The allocating `JacRev(coeff, range)` and `JacFwd(coeff, range)` overloads are removed from `InterpreterBase` and `Interpreter`, and `EvaluateRoots` writes into a caller-provided span. Downstream code must migrate to the span-filling overloads.

`Interpreter` methods taking a `Range` return `InterpreterError::Code::InvalidRange` (`ExpectedSize` = dataset rows, `ActualSize` = range end) if the range ends after the last dataset row; nothing is written to the output in that case.

## `EvaluatorBase`

Header: [`operon/operators/evaluator.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/operators/evaluator.hpp)

```cpp
class Objective final : public EvaluatorBase {
public:
  using EvaluatorBase::EvaluatorBase;

protected:
  auto Evaluate(Individual const&, Span<Scalar>) const
      -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;
  auto Score(RandomGenerator&, Individual const&, Span<Scalar>, std::optional<EvaluatedBuffer>) const
      -> ReturnType override;
};
```

The three-argument `operator()` is the sole public execution path and is `final`. It increments `CallCount` once, runs
`Evaluate`, replaces an evaluation error with `ErrMax` repeated `ObjectiveCount()` times, and otherwise runs `Score`.
`Evaluate` and `Score` are protected hooks. A value-based evaluator MUST fill the provided scratch span and return
`MarkEvaluated(individual, values, scaling)`, where `scaling` is the exact `std::optional<LinearScaling>` it will apply
(`nullopt`: scaling disabled). `Score` MUST acquire values through `EvaluatedBuffer::Values(ind, buf)` and consume the
buffer's transform rather than refitting. It is invalid to assume the scratch buffer contains the right individual
without that proof. Whether an evaluator is a value evaluator is a compile-time fact: `Concepts::ValueEvaluator` is satisfied by an `Evaluator<DTable>` that is not itself a shape-constrained adapter.

`ObjectiveCount()` defaults to one; override it for vector objectives. `Prepare(population)` is a pre-evaluation hook. `SetBudget`, `BudgetExhausted`, and atomic counter accessors expose the evaluation budget and accounting. Built-in `Evaluator<DTable>` accepts an `ErrorMetric`; `SSE`, `MSE`, `NMSE`, `RMSE`, `MAE`, `R2`, and `C2` are available metric types. Its linear scaling is fitted under `Problem::LinearScalingOmitsNonFinite()`; `skipNonFinite` only changes scoring. `JIT::JitEvaluator` derives from `Evaluator<ScalarDispatch>` and overrides only the `Predict` step.


Evaluator failures are converted to the evaluator's worst-score sentinel rather than escaping through the optimization loop. In particular, `MinimumDescriptionLengthEvaluator::Score` returns `EvaluatorBase::ErrMax` when Jacobian evaluation fails, preserving the evaluator's fixed fitness arity and allowing population comparisons to continue safely.

## Shape constraints

Header: [`operon/operators/shape_constrained_evaluator.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/operators/shape_constrained_evaluator.hpp)

```cpp
TreePropertyEvaluator length { &problem, [](Tree const& t) { return Scalar(t.Length()); } };
ShapeConstrainedValueEvaluator<Evaluator<ScalarDispatch>> evaluator {
    ShapeConstraintSpec { .Constraints = constraints,
        .Policy = { .Enforcement = ShapeConstraintEnforcement::ExtraObjective },
        .Structural = { &length },
        .Bound = { .Mode = ShapeBoundMode::Interval } },
    &dispatch, &problem, &dispatch, NMSE {} };                     // trailing arguments construct the base evaluator
auto fitness = evaluator(rng, individual, scratch);               // [error, length, shape violation]
auto summary = evaluator.Certifier().MeasureCanonical(tree);       // standalone/reporting only
```

`ShapeCertifier` (not an evaluator) owns the constraint set, domain boxes, bound backend and options.
`Measure(tree, transform)` takes the exact fitted transform;
it never fits a fallback. `MeasureCanonical(tree)` fits the canonical transform under
`Problem::LinearScalingOmitsNonFinite()` and is for reporting and standalone use.

`ShapeConstrainedValueEvaluator<BaseEvaluator>` is a `final` evaluator derived from a concrete
`Concepts::ValueEvaluator` (`Evaluator<DTable>` with any metric, the MDL, FBF and likelihood evaluators, or
`JIT::JitEvaluator`; BIC and AIC are sealed). It *is* the value evaluator, so its only public execution method is the
inherited `operator()` and it calls `BaseEvaluator`'s own protected phases on itself. Its structural post-objectives
are restricted to `TreePropertyEvaluator`, whose prediction phase is sealed, so no post-objective can add a prediction
pass. The output is `[primary (+ PenaltyWeight * violation under Penalty), post-objectives..., violation under
ExtraObjective]`. `HardReject` returns `WorstValue()` for the whole vector without evaluating post-objectives; a failed
prediction or score returns `ErrMax` for the whole vector. The verdict's `Feasible` flag is authoritative: an
uncertified constraint is infeasible even when `UnknownViolation` is zero. Constrained adapters cannot nest: an
adapter is not a `ValueEvaluator`, so
`ShapeConstrainedValueEvaluator<ShapeConstrainedValueEvaluator<X>>` does not compile.

The evaluator-independent half of the adapter is a non-template implementation detail,
`Operon::detail::ShapeConstraintEnforcer`, a second (non-evaluator) base. It is not part of the public API surface:
the shape configuration (`ShapeConstraintSpec`, `ShapeConstraintPolicy`, `ShapeCertifier`), the adapter template and the
`MakeValueEvaluator` factory are. Its public members (`Certifier()`, `Policy()`, `WorstValue()`, `SetWorstValue()`,
`Violations()`) are available on every adapter and, for type-erased evaluators
inside the library's own CLI and probes, through `dynamic_cast<detail::ShapeConstraintEnforcer const*>(evaluator)`.
Factories construct the typed adapter before erasing it: `MakeValueEvaluator<Leaf>(spec, dtable, leafArgs...)` returns
`std::unique_ptr<EvaluatorBase>` holding either the plain leaf or `ShapeConstrainedValueEvaluator<Leaf>`, and
`JIT::MakeJitObjects(..., spec)` builds `ShapeConstrainedValueEvaluator<JitEvaluator>` for mode `all`, so the JIT
optimizer keeps using the same `JitEvaluator` subobject.

A call evaluates the base evaluator's prediction exactly once. The fitness describes the evaluated model, i.e. the
coefficients present at the call (under non-Lamarckian local search, the optimized coefficients that were scored);
the adapter does not re-certify the restored genotype. To certify a model for export or reporting, use
`ShapeCertifier::MeasureCanonical` (or `Measure` with an exact transform). The adapter reports
`StructuralFitnessReusable() == false`, so offspring generators bypass the structural (Zobrist) fitness cache for it.

`MultiEvaluator::Add` throws `std::invalid_argument` for a `ShapeConstrainedValueEvaluator` child, whatever its base
type and however it is type-erased; the adapter already owns the complete constrained objective vector, including its
structural objectives (pass them in `ShapeConstraintSpec::Structural`). Ordinary children keep generic `operator()`
dispatch.

`Interval` is the default bound mode. `Combined` intersects interval and affine bounds; `Affine` uses affine arithmetic alone. With any mode, a `BisectionDepth` of `N > 0` bounds the model over the hull of the `2^N` sub-boxes of the domain box (`0`, the default, is the whole box); `Affine` and `Combined` intersect that hull with the whole-box bound. The routes that evaluate one sub-box at a time (derivative constraints of `Interval`, all constraints of `Affine` and `Combined`) perform at most 12 halvings however large `N` is. The mode and `BisectionDepth` (in `[0, 20]`, validated by the `ShapeCertifier` constructor) are set through `ShapeBoundOptions`; `ParseShapeBoundOptions` reads the `mode[:N]` form used by `--shape-bound-mode`. A failure to certify is not proof that the mathematical constraint fails—enclosures are conservative.

## Coefficient optimization

Header: [`operon/optimizer/optimizer.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/optimizer/optimizer.hpp)

`OptimizerBase::Optimize(rng, tree)` returns `FitOutcome`, an expected `FitResult` or one of `FitFailure`, `FitEvaluationError`, and `FitConfigurationError`. `Diagnostics(outcome)` always returns initial/final parameters, cost, iteration, and function/Jacobian counts. `EvaluationError(outcome)` and `ConfigurationError(outcome)` selectively expose typed failures.

Dataset sample weights are validated with `ValidateWeights(weights, rows)` (`operon/optimizer/least_squares.hpp`), which accepts an empty span, one broadcast value, or one value per row, each finite and nonnegative. Violations are typed: `FitConfigurationError::Error` is a `WeightError` (`SizeMismatch`, `NegativeValue`, `NotANumber`, `Infinite`, plus the expected/actual size and the first offending `Row`) for the Levenberg–Marquardt, L-BFGS and SGD optimizers alike. `WeightError::Row` is an index into the span that was validated, so its frame depends on the caller: for the optimizers' `FitConfigurationError` it is relative to the start of the training range (0 = first training row); for `FitLeastSquares` it is the index into `options.Weights`; for a gradient cost's `Evaluate` (`GradientErrorCode::InvalidWeights`) it is the absolute row of the whole-dataset weight column. `FitEvaluationError::Error` is a `GradientError`; its `Row`/`Column` locate a non-finite residual or Jacobian entry, or (for `InvalidWeights`) the first invalid weight in the frame above, and `ToGradientError(LeastSquaresError)` maps every least-squares error code to the distinct `GradientErrorCode` of the same name.

`GaussianGradientCostFunction` and `PoissonGradientCostFunction<LogInput>` are written in `Operon::Scalar`; they take no scalar template parameter.

`LBFGSOptimizer<DTable, Cost>` and `SGDOptimizer<DTable, Cost>` accept any `Cost` satisfying `Concepts::InterpreterGradientCost` (`operon/optimizer/interpreter_gradient_cost.hpp`): the solver-facing `Concepts::GradientCost` (`NumParameters()`, `Evaluate(ConstScalarSpan, ScalarSpan) -> tl::expected<Scalar, GradientError>`) plus a constructor from `(interpreter, target, range, rng, batchSize, weights)` (`target`, `range`, and `weights` in whole-dataset absolute row coordinates), `FunctionEvaluations()` and `JacobianEvaluations()` returning `std::size_t`, and a `static constexpr bool UsesDatasetWeights`. When it is `true` (Gaussian) the optimizer validates the in-range slice of the dataset weights and passes the whole weight column to the constructor; when `false` (Poisson) the constructor receives an empty span and the dataset weights are neither validated nor forwarded.

Solver failure semantics: `SGDOptimizer` stops at the first failed or non-finite cost evaluation and returns the last finite iterate, so no non-finite update reaches the tree. Its `SetIterations` budget is a maximum number of parameter updates, each one gradient step on one batch (not a pass over the data), and `Iterations` reports the number of updates actually applied; a step that converges (update below tolerance) or hits a failed evaluation applies no update and is not counted. `LBFGSOptimizer` lets `lbfgs` reject non-finite line-search trials and revert to the last accepted iterate; a solver-facing evaluation error is returned as `FitEvaluationError` only if the final endpoint evaluation fails or the solve itself fails. `lbfgs` exposes no iteration count, so `LBFGSOptimizer` always reports `Iterations == 0`. Dataset weights are validated only for costs that declare `UsesDatasetWeights` (the Gaussian cost); `PoissonGradientCostFunction` never reads them (exposure is passed explicitly to its constructor). Iteration budgets larger than a solver's integer type saturate at its maximum instead of wrapping.

A tree with no optimizable coefficients is not solved by any optimizer: the endpoint cost is evaluated once and reported as both `InitialCost` and `FinalCost` with `Iterations == 0`, `FunctionEvaluations == 1`, `JacobianEvaluations == 0`, and an empty parameter vector (outcome `FitFailure`, or `FitEvaluationError` if that evaluation fails), identically for the Tiny and Eigen backends.

`LevenbergMarquardtOptimizer<DTable, OptimizerType>` is the standard local optimizer; the shipped non-JIT CLI and enumeration paths select `OptimizerType::Eigen` because it is more numerically robust on GP-scale fits. `OptimizerType::Tiny` remains available as an explicit backend and remains the template default for library callers. `SetIterations` sets its accepted-step budget (see the `FitLeastSquares` iteration semantics below); `SetBatchSize(0)` means the complete data range. It optimizes only the tree's marked coefficients and leaves topology unchanged. `Optimize()` builds all per-call state locally, so concurrent calls on a shared optimizer are safe provided nothing calls `SetIterations` concurrently and the `Problem` and `Dataset` are read-only. `JitLevenbergMarquardtOptimizer<DTable, JacobianOnly>` (with `HAVE_ASMJIT`) always solves with the Eigen backend; it has no backend parameter.

This optimizer migration is a deliberate breaking API change. `Concepts::HasFisherMatrix`, the likelihood `ComputeFisherMatrix` methods and their Eigen `Matrix`/`Vector` aliases, `StatisticalDiagnosticsCapability`, `lm_weights.hpp` (`TryValidateLMWeights`, `LMWeightError`, `ApplyLM*`), the `OptimizerType` template parameter of `JitLevenbergMarquardtOptimizer` (now `<DTable, bool JacobianOnly>`), and scalar template parameters on `GaussianGradientCostFunction` and `PoissonGradientCostFunction` are removed. Custom `LBFGSOptimizer` and `SGDOptimizer` costs must now satisfy `Concepts::InterpreterGradientCost`, including `static constexpr bool UsesDatasetWeights`; their configuration errors use `WeightError` rather than `LMWeightError`. Downstream consumers must migrate to the Fisher-diagonal and backend-neutral APIs and rebuild.

`JitLeastSquaresCostFunction` (`operon/optimizer/jit_least_squares.hpp`, `HAVE_ASMJIT`; float-only, `Operon::Scalar == float`) never throws on user data. The constructor validates once and stores the first violation; every `Evaluate()` then returns it unchanged as a `LeastSquaresError` before any buffer, kernel, or interpreter call, and `FitLeastSquares`/`JitLevenbergMarquardtOptimizer` report it as a `FitEvaluationError` with the same `Code`/`Expected`/`Actual`/`Row`/`Column` (`NaN` costs, `Iterations == 0`), except that ranges whose residual count exceeds `INT_MAX` are reported as `InvalidShape` before the cost is evaluated.

| Violation | `Code` | `Expected` | `Actual` | `Row` / `Column` |
|---|---|---|---|---|
| range overruns the target (`Range.Start() + Range.Size() > target.size()`) | `InvalidShape` | the required target size, `Start + Size` (saturating at `SIZE_MAX`) | `target.size()` | 0 / 0 |
| range has more than `INT32_MAX - 7` rows | `InvalidShape` | `INT32_MAX - 7` (the row limit) | `Range.Size()` | 0 / 0 |
| non-negative `nConsts` different from `NumParameters()` | `InvalidShape` | `nConsts` | `NumParameters()` | 0 / 0 |
| column array size differs from `nVars` (or from `JIT::VarOrder(tree).size()` when `nVars < 0`) | `InvalidShape` | the variable count | the array size | `Row` is 0 for the residual kernel's array and 1 for the Jacobian kernel's; `Column` 0 |
| null column pointer | `InvalidView` | the variable count | the array size | `Row` as above; `Column` is the index of the first null pointer |

An absent kernel selects the interpreter fallback and its column array and metadata are ignored. That a kernel was compiled for this tree and that each column addresses enough padded rows remain caller preconditions (the `JitEvaluator` metadata supplies them). `LeastSquaresError::Row` and `Column` are not a fixed coordinate system: their meaning depends on `Code` (for example a residual or Jacobian index for `NonFiniteEvaluation`, a kernel/array selector and index for the JIT shape errors above), so interpret them only together with the code and the producing cost. The object is single-threaded: `Evaluate()` is `const` but writes mutable residual, Jacobian, and interpreter scratch, so concurrent `Evaluate()` calls on one instance are not safe and there is no locking; use one instance per thread.

### `FitLeastSquares`

Header: [`operon/optimizer/least_squares_fit.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/optimizer/least_squares_fit.hpp)

`Operon::FitLeastSquares(cost, initialParameters, options)` minimizes `0.5 * sum(w_i * r_i^2)` for any `LeastSquaresCostFunction` through the same adapter and solver driver that `LevenbergMarquardtOptimizer` and `JitLevenbergMarquardtOptimizer` use, so out-of-tree costs (for example a grouped cost over tree coefficients) need no solver loop and no `detail::` type. It returns the same `FitOutcome` as `OptimizerBase::Optimize`.

```cpp
Operon::LeastSquaresFitOptions options {
    .Backend = Operon::OptimizerType::Tiny, // default; Eigen is selectable
    .Iterations = 100,
    .Weights = weights,                     // empty, one value, or NumResiduals() values
    .RecoverNonFinite = false,              // true: non-finite trial outputs are rejected steps
};
auto outcome = Operon::FitLeastSquares(cost, tree.GetCoefficients(), options);
if (outcome) { tree.SetCoefficients(outcome->FinalParameters); }
```

`cost`, `initialParameters`, and `options.Weights` are borrowed for the call only; the outcome owns its vectors. The function keeps no static state, so concurrent calls on distinct cost objects are safe; sharing one cost object across threads additionally requires that cost to document a thread-safe `Evaluate`.

| Situation | Outcome |
|---|---|
| `FinalCost < InitialCost` | `FitResult` |
| valid fit that does not improve (including non-finite final cost) | `FitFailure` |
| weights violate `ValidateWeights` | `FitConfigurationError` carrying a `WeightError`; the cost is not evaluated and the counters are zero |
| `initialParameters.size() != cost.NumParameters()` | `FitEvaluationError` with `GradientErrorCode::InvalidShape` (`Expected`/`Actual` hold the two counts); the cost is not evaluated; `InitialCost` and `FinalCost` are NaN and the counters are zero |
| Eigen backend with `cost.NumResiduals() < cost.NumParameters()` | `FitEvaluationError` with `GradientErrorCode::InvalidShape` (`Expected == NumParameters()`, `Actual == NumResiduals()`); no solver step and no cost evaluation; NaN costs, zero counters, `FinalParameters == InitialParameters`. Tiny has no such restriction |
| `cost.NumResiduals() > INT_MAX` (either backend, any parameter count) | `FitEvaluationError` with `GradientErrorCode::InvalidShape` (`Expected == INT_MAX`, `Actual == NumResiduals()`); no solver, buffer allocation, or cost evaluation; NaN costs, zero counters, `FinalParameters == InitialParameters` |
| cost failure, or non-finite output with `RecoverNonFinite == false` | `FitEvaluationError` with the cost's `Code`/`Row`/`Column`/`Cause` preserved; if the very first evaluation fails, `InitialCost` and `FinalCost` are NaN and `Iterations == 0` |
| zero parameters | no solver runs; one residual evaluation; `InitialCost == FinalCost`, `Iterations == 0`, `FunctionEvaluations == 1`, `JacobianEvaluations == 0`, empty parameters; `FitFailure` (or `FitEvaluationError` if the evaluation fails) |

Iteration semantics: `Iterations` is an accepted-LM-step budget on both backends, and `Diagnostics().Iterations` is the number of accepted steps taken. Tiny: accepted steps are capped by `Iterations`, with total attempts (accepted or rejected) bounded by `Iterations * (n + 1)`. Eigen: accepted steps are capped by `Iterations` (`lm.iterations() - 1`, since Eigen counts from 1), and `maxfev`, which bounds function evaluations including rejected trial steps, is `max(Iterations * (n + 1), 1)`. `Iterations == 0` performs no step on either backend: the initial point is evaluated and the outcome is a `FitFailure` with unchanged parameters. Budgets larger than the backend's integer type saturate at its maximum instead of wrapping. `FunctionEvaluations` and `JacobianEvaluations` count the cost evaluations that requested residuals and a Jacobian.

## Search-driver layout

`GeneticAlgorithmBase` gains the data member that backs `IncompleteGeneration()`, which changes the object layout of `GeneticProgrammingAlgorithm` and `NSGA2`. Consumers compiled against earlier headers are not ABI-compatible and must be rebuilt.
