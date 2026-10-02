# Execution and evaluation API

## `Interpreter`

Header: [`operon/interpreter/interpreter.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/interpreter/interpreter.hpp)

```cpp
Interpreter<Scalar, ScalarDispatch> interpreter { &dispatch, &dataset, &tree };
auto values = interpreter.Evaluate(tree.GetCoefficients(), range);
auto jacobian = interpreter.JacRev(tree.GetCoefficients(), range);
```

| Method family | Result |
| --- | --- |
| `Evaluate(coeff, range[, output])` | model values or `InterpreterError` |
| `JacRev` / `JacFwd` | coefficient Jacobian or `InterpreterError`; rows are observations, columns are optimizable constants |
| `JacRevVariable` / `JacFwdVariable` | derivative of output with respect to one input variable hash, or `InterpreterError` |

All fallible `Interpreter` operations return `tl::expected`; inspect the result before using its value.

This API change is a deliberate breaking boundary for the next Operon API epoch: `InterpreterBase` virtual return types use `tl::expected`, and the former virtual `Try*` methods are removed. Downstream bindings and consumers must rebuild and migrate together; binaries built against the previous virtual interface are not ABI-compatible.

## `EvaluatorBase`

Header: [`operon/operators/evaluator.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/operators/evaluator.hpp)

```cpp
class Objective final : public EvaluatorBase {
public:
  using EvaluatorBase::EvaluatorBase;
  auto Evaluate(Individual const&, Span<Scalar>) const
      -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;
  auto Score(ScoreContext, std::optional<EvaluatedBuffer>) const
      -> ReturnType override;
};
```

The three-argument `operator()` is final. It calls `Evaluate`, then `Score`, and replaces an evaluation error with `{ErrMax}`. A value-based evaluator MUST fill the provided scratch span and return `MarkEvaluated(individual, values)`. `Score` MUST acquire values through `EvaluatedBuffer::Values(ctx.Ind, ctx.Scratch)`; it is invalid to assume the scratch buffer contains the right individual without that proof.

`ObjectiveCount()` defaults to one; override it for vector objectives. `Prepare(population)` is a pre-evaluation hook. `SetBudget`, `BudgetExhausted`, and atomic counter accessors expose the evaluation budget and accounting. Built-in `Evaluator<DTable>` accepts an `ErrorMetric`; `SSE`, `MSE`, `NMSE`, `RMSE`, `MAE`, `R2`, and `C2` are available metric types.


Evaluator failures are converted to the evaluator's worst-score sentinel rather than escaping through the optimization loop. In particular, `MinimumDescriptionLengthEvaluator::Score` returns `EvaluatorBase::ErrMax` when Jacobian evaluation fails, preserving the evaluator's fixed fitness arity and allowing population comparisons to continue safely.
## Shape constraints

Header: [`operon/operators/shape_constrained_evaluator.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/operators/shape_constrained_evaluator.hpp)

```cpp
ShapeConstrainedEvaluator constrained { &baseEvaluator, &dispatch, constraints };
constrained.SetBoundMode(ShapeBoundMode::Interval);
auto summary = constrained.Measure(tree);
bool feasible = constrained.Feasible(tree);
```

`ShapeConstrainedEvaluator` wraps another evaluator. It bounds the tree and requested input derivatives on declared domain boxes before delegating. `HardReject` returns the configured worst value; `Penalty`, `ExtraObjective`, and `FeasibilityFirst` express other policy choices. `ShapeViolationEvaluator` exposes the measurement as an objective without wrapping another evaluator.

`Interval` is the default bound mode. `Combined` intersects interval and affine bounds; `Bisected` requires `Interval`. `SetBoundMode` and `SetBoundOptions` validate these combinations. A failure to certify is not proof that the mathematical constraint fails—enclosures are conservative.

## Coefficient optimization

Header: [`operon/optimizer/optimizer.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/optimizer/optimizer.hpp)

`OptimizerBase::Optimize(rng, tree)` returns `FitOutcome`, an expected `FitResult` or one of `FitFailure`, `FitEvaluationError`, and `FitConfigurationError`. `Diagnostics(outcome)` always returns initial/final parameters, cost, iteration, and function/Jacobian counts. `EvaluationError(outcome)` and `ConfigurationError(outcome)` selectively expose typed failures.

Dataset sample weights are validated with `ValidateWeights(weights, rows)` (`operon/optimizer/least_squares.hpp`), which accepts an empty span, one broadcast value, or one value per row, each finite and nonnegative. Violations are typed: `FitConfigurationError::Error` is a `WeightError` (`SizeMismatch`, `NegativeValue`, `NotANumber`, `Infinite`, plus the expected/actual size and the first offending `Row`) for the Levenberg–Marquardt, L-BFGS and SGD optimizers alike. `WeightError::Row` is an index into the span that was validated, so its frame depends on the caller: for the optimizers' `FitConfigurationError` it is relative to the start of the training range (0 = first training row); for `FitLeastSquares` it is the index into `options.Weights`; for a gradient cost's `Evaluate` (`GradientErrorCode::InvalidWeights`) it is the absolute row of the whole-dataset weight column. `FitEvaluationError::Error` is a `GradientError`; its `Row`/`Column` locate a non-finite residual or Jacobian entry, or (for `InvalidWeights`) the first invalid weight in the frame above, and `ToGradientError(LeastSquaresError)` maps every least-squares error code to the distinct `GradientErrorCode` of the same name.

`GaussianGradientCostFunction` and `PoissonGradientCostFunction<LogInput>` are written in `Operon::Scalar`; they take no scalar template parameter.

`LBFGSOptimizer<DTable, Cost>` and `SGDOptimizer<DTable, Cost>` accept any `Cost` satisfying `Concepts::InterpreterGradientCost` (`operon/optimizer/interpreter_gradient_cost.hpp`): the solver-facing `Concepts::GradientCost` (`NumParameters()`, `Evaluate(ConstScalarSpan, ScalarSpan) -> tl::expected<Scalar, GradientError>`) plus a constructor from `(interpreter, target, range, rng, batchSize, weights)` (`target`, `range`, and `weights` in whole-dataset absolute row coordinates), `FunctionEvaluations()` and `JacobianEvaluations()` returning `std::size_t`, and a `static constexpr bool UsesDatasetWeights`. When it is `true` (Gaussian) the optimizer validates the in-range slice of the dataset weights and passes the whole weight column to the constructor; when `false` (Poisson) the constructor receives an empty span and the dataset weights are neither validated nor forwarded.

Solver failure semantics: `SGDOptimizer` stops at the first failed or non-finite cost evaluation and returns the last finite iterate, so no non-finite update reaches the tree. Its `SetIterations` budget is a maximum number of parameter updates, each one gradient step on one batch (not a pass over the data), and `Iterations` reports the number of updates actually applied; a step that converges (update below tolerance) or hits a failed evaluation applies no update and is not counted. `LBFGSOptimizer` lets `lbfgs` reject non-finite line-search trials and revert to the last accepted iterate; a solver-facing evaluation error is returned as `FitEvaluationError` only if the final endpoint evaluation fails or the solve itself fails. `lbfgs` exposes no iteration count, so `LBFGSOptimizer` always reports `Iterations == 0`. Dataset weights are validated only for costs that declare `UsesDatasetWeights` (the Gaussian cost); `PoissonGradientCostFunction` never reads them (exposure is passed explicitly to its constructor). Iteration budgets larger than a solver's integer type saturate at its maximum instead of wrapping.

A tree with no optimizable coefficients is not solved by any optimizer: the endpoint cost is evaluated once and reported as both `InitialCost` and `FinalCost` with `Iterations == 0`, `FunctionEvaluations == 1`, `JacobianEvaluations == 0`, and an empty parameter vector (outcome `FitFailure`, or `FitEvaluationError` if that evaluation fails), identically for the Tiny and Eigen backends.

`LevenbergMarquardtOptimizer<DTable, OptimizerType>` is the standard local optimizer; `OptimizerType::Tiny` is the default and `OptimizerType::Eigen` stays selectable. `SetIterations` sets its accepted-step budget (see the `FitLeastSquares` iteration semantics below); `SetBatchSize(0)` means the complete data range. It optimizes only the tree's marked coefficients and leaves topology unchanged. `Optimize()` builds all per-call state locally, so concurrent calls on a shared optimizer are safe provided nothing calls `SetIterations` concurrently and the `Problem` and `Dataset` are read-only. `JitLevenbergMarquardtOptimizer<DTable, JacobianOnly>` (with `HAVE_ASMJIT`) always solves with the Eigen backend; it has no backend parameter.

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
| cost failure, or non-finite output with `RecoverNonFinite == false` | `FitEvaluationError` with the cost's `Code`/`Row`/`Column`/`Cause` preserved; if the very first evaluation fails, `InitialCost` and `FinalCost` are NaN and `Iterations == 0` |
| zero parameters | no solver runs; one residual evaluation; `InitialCost == FinalCost`, `Iterations == 0`, `FunctionEvaluations == 1`, `JacobianEvaluations == 0`, empty parameters; `FitFailure` (or `FitEvaluationError` if the evaluation fails) |

Iteration semantics: `Iterations` is an accepted-LM-step budget on both backends, and `Diagnostics().Iterations` is the number of accepted steps taken. Tiny: accepted steps are capped by `Iterations`, with total attempts (accepted or rejected) bounded by `Iterations * (n + 1)`. Eigen: accepted steps are capped by `Iterations` (`lm.iterations() - 1`, since Eigen counts from 1), and `maxfev`, which bounds function evaluations including rejected trial steps, is `max(Iterations * (n + 1), 1)`. `Iterations == 0` performs no step on either backend: the initial point is evaluated and the outcome is a `FitFailure` with unchanged parameters. Budgets larger than the backend's integer type saturate at its maximum instead of wrapping. `FunctionEvaluations` and `JacobianEvaluations` count the cost evaluations that requested residuals and a Jacobian.
