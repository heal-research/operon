# Execution

![Expression execution](../assets/diagrams/execution.svg){ .diagram }

## Execution paths

`Interpreter<T, DTable>` binds a tree, dataset, dispatch table, and row range. Its fallible evaluation and differentiation operations return `tl::expected` with `InterpreterError` for a missing variable/function, derivative, or incorrectly sized output span.

| Need | API |
| --- | --- |
| sampled model values | `Evaluate` |
| coefficient Jacobian for local fitting | `JacRev` / `JacFwd` |
| derivative with respect to an input column | `JacRevVariable` / `JacFwdVariable` |
| enclosure over a domain box | `IntervalEvaluator` or `AffineEvaluator` |

The Jacobian has one row per requested data row and one column per optimizable constant. It is not a derivative with respect to inputs; use the variable-Jacobian APIs for that.

## Dispatch and custom functions

The dispatch table maps node hashes to scalar callables and, where needed, derivatives. Register a custom callable in `ScalarDispatch`, then expose its hash and arity to generation with `PrimitiveSet::AddFunction`; registering only one side produces a tree that cannot be generated or executed. The [custom primitives example](https://github.com/heal-research/operon/blob/main/example/custom_primitives.cpp) shows the complete pairing.

## Sampling is not certification

Sampled evaluation measures rows selected by a `Range`; it says nothing about points between rows. Interval and affine evaluators instead propagate enclosures through the same tree over a domain box. They can prove a bound, but dependency over-approximation can make an actually valid expression uncertifiable. Use sampled execution for loss and bounds for whole-domain claims; do not infer one from the other.
