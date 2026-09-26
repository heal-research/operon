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

`LevenbergMarquardtOptimizer<DTable, OptimizerType>` is the standard local optimizer. `SetIterations` controls its iteration budget; `SetBatchSize(0)` means the complete data range. It optimizes only the tree's marked coefficients and leaves topology unchanged.
