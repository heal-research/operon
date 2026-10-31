# Evaluator architecture review

This record documents the evaluator, fitted linear scaling and shape-constraint redesign: the reviews that motivated it and the design that was implemented. The authoritative description of the shipped API is the implemented design below.

## Decision status

**Replacement implemented.** The earlier non-template internal `ShapeGate` was replaced by the derived `template <Concepts::ValueEvaluator BaseEvaluator> ShapeConstrainedValueEvaluator final : public BaseEvaluator`. `EvaluatorBase::operator()` remains the only public execution path, and the replacement removed the gate's privileged access to a foreign evaluator. The reviews below are kept as the record of the decision.

The invariant that drives the design: an `EvaluatorBase` is an `OperatorBase`, so its public execution contract is exclusively

```cpp
operator()(RandomGenerator&, Individual const&, Span<Scalar>) -> Vector<Scalar>
```

A generic `MultiEvaluator` invokes arbitrary children only through that contract, and shape semantics never depend on the preceding sibling in insertion order.

### Post-implementation gate-boundary review

Claude Opus, GLM, and GPT-6 Astra reviewed the implemented gate and the typed alternative:

| Review | Recommendation | Evidence |
|---|---|---|
| Opus | Replace `ShapeGate` with a derived `template <ValueEvaluator BaseEvaluator> ShapeConstrainedValueEvaluator final : public BaseEvaluator`. | Removes privileged child-phase friendship and makes the wrapped value evaluator explicit. It found no correctness failure in a derived template. It accepts deliberate removal of `final` from value leaves and typed construction before factory erasure as the cost. |
| GLM | Keep non-template `ShapeGate`. | Current value leaves and JIT are `final`; the CLI and JIT factories erase evaluator types; inherited prediction capability would make nested gates harder to reject. It judged constructor-time capability validation adequate and found no defect in the implemented gate. |
| GPT-6 Astra | Replace `ShapeGate` with the derived template. | The `ShapeGate` friendship is unnecessary in a derived adapter. Removing `final` only from the closed set of wrapped value leaves and constructing the adapter in bounded factory branches does not reopen the final `operator()` execution contract. |

The selected migration, now implemented, was:

1. retain `ShapeCertifier` and move non-type-specific enforcement into a non-template core (`detail::ShapeConstraintEnforcer`);
2. introduce the final derived `ShapeConstrainedValueEvaluator<BaseEvaluator>` constrained to `Concepts::ValueEvaluator` and reject nested constrained types at compile time;
3. remove `final` only from the concrete value leaves that must be adapter bases, including JIT;
4. construct each supported typed adapter before the GP, NSGP, and JIT factories erase it;
5. remove the non-template `ShapeGate`, its evaluator friendship, and the runtime `ProducesPredictions()` check.

## Reviewed implementation (before the redesign)

| Area | Files |
|---|---|
| Generic operator contract | `include/operon/core/operator.hpp` |
| Evaluator base, value buffer, scaling claim, composition context | `include/operon/operators/evaluator.hpp` |
| Scalar evaluator and multi-objective composition | `source/operators/evaluator.cpp` |
| Shape-constrained and shape-violation evaluators | `include/operon/operators/shape_constrained_evaluator.hpp`, `source/operators/shape_constrained_evaluator.cpp` |
| JIT evaluator | `include/operon/interpreter/backend/jit/jit_evaluator.hpp`, `source/interpreter/backend/jit/jit_evaluator.cpp` |

The previous design ran `Evaluate` and `Score` through a second public protocol (`Run` with a stack-local `ScoreContext`) whose `LastAppliedScaling()` published the latest value evaluator's transform to later `MultiEvaluator` siblings. The shape evaluators consumed that claim or fell back to a tree-based measurement.

## Review findings

| Finding | Assessment |
|---|---|
| `MultiEvaluator::Score` called `ev->Run(ctx)`; `Run` was a second evaluator invocation protocol | Confirmed |
| Shape score depended on the latest preceding sibling claim, so reordering or nesting changed the transform used for measurement | Confirmed |
| A shape-first evaluation could take a second raw prediction pass | Confirmed |
| A wrapped shape gate could use a scaling policy other than the wrapped evaluator's (`Problem` fallback) | Confirmed |
| `Evaluate`/`Score` were public virtual hooks that bypassed the `operator()` failure boundary | Confirmed |
| JIT hard-coded `omitNonFinite = false` and duplicated the scalar `Score` | Confirmed |
| ErrMax fallback returned one element regardless of `ObjectiveCount()` | Confirmed |
| Non-finite scaling policy was split between `skipNonFinite_`, `Problem` and JIT | Confirmed |

Scratch mutation was also raised: scaling is applied in place, so the scratch span is not a reusable prediction artifact. Each scalar evaluator's `Evaluate` overwrites the training-range view before fitting, so the narrow claim that a later evaluator fits against earlier scaled bytes was not generally correct; the design nevertheless keeps the span strictly invocation-local.

## Settled requirements

1. Shape constraints apply to the genotype and coefficients present at the moment of an evaluation call.
2. NSGP keeps `[error, tree length, shape violation]`, including Penalty with ExtraObjective.
3. Two prediction passes are unacceptable for shape enforcement. Normal constrained evaluation (GP and NSGP, feasible and infeasible, HardReject included) performs exactly one.
4. Fitness describes the evaluated model: under non-Lamarckian local search the optimized coefficients that were scored are the ones certified, and the restored genotype is not re-certified. Exported models are certified explicitly with `ShapeCertifier::MeasureCanonical`.
5. An opaque generic wrapper that can add a pass is not acceptable.

## Implemented design

### Evaluator contract

- `EvaluatorBase::operator()` is the sole public execution path. It is `final`, increments `CallCount` exactly once per call (failure included), runs the protected `Evaluate` phase, and on failure returns `ErrMax` repeated `ObjectiveCount()` times; otherwise it runs the protected `Score(rng, ind, buf, evaluated)` phase. `Run`, `ScoreContext`, `LastAppliedScaling` and `AppliedScaling` no longer exist, and phases no longer increment counters.
- `Evaluate`, `Score` and the restoration hook are protected. The access boundary is narrow: a derived evaluator (`ShapeConstrainedValueEvaluator`) calls its base evaluator's phases on itself, and the lifecycle helper `RestoreNonLamarckianCoefficients` is the one friend (it reaches the restoration hook). `MultiEvaluator` has no such access and calls every child through `operator()`. Ordinary lifecycle APIs (`Prepare`, `ObjectiveCount`, `Stats`, `Budget`, ...) stay public.
- `EvaluatedBuffer` is the one invocation-local prediction artifact handed from `Evaluate` to `Score`. It proves the scratch view's individual identity and carries the exact `std::optional<LinearScaling>` the producing evaluator will apply (`nullopt`: scaling disabled). It never escapes an operator call.
- The value-evaluator capability is a compile-time fact, `Concepts::ValueEvaluator` (an `Evaluator<DTable>` that is not itself a shape-constrained adapter); there is no runtime capability query. `StructuralFitnessReusable()` states whether a fitness may be reused across structurally identical genotypes.

### Scaling policy and JIT

The scaling fit uses one declared policy, `Problem::LinearScalingOmitsNonFinite()`, for the interpreter and JIT evaluators, shape certification, the Pareto-front exporter and reporting. `skipNonFinite` is scoring-only and is rejected by the `Evaluator` constructor unless linear scaling is disabled or the problem policy already omits non-finite rows. `JitEvaluator` derives from `Evaluator<ScalarDispatch>` and overrides only the protected `Predict` step, so counting, scaling fit and scoring are shared with the interpreter.

### ShapeCertifier and the typed adapter

`ShapeCertifier` is a non-evaluator service. It owns the constraint set, domain boxes, bound backend and options:

- `Measure(tree, transform)` receives the exact transform; it never fits a fallback. `MeasureCanonical(tree)` is the entry point for reporting and for certifying exported models; it fits the canonical transform itself.
- The measurement summary's `Feasible` and `Violation` are the per-measurement result. `Feasible` is the only feasibility truth, so `UnknownViolation == 0` still yields an infeasible result when a bound cannot be certified.

`ShapeConstrainedValueEvaluator<BaseEvaluator>` is a `final` class derived from the concrete value evaluator it constrains; it owns the complete constrained output:

```text
[ primary (+ PenaltyWeight * violation), structural post-objectives..., violation ]
```

- The enforcement and bookkeeping that do not depend on `BaseEvaluator` (certifier, policy, structural objectives, worst value, rejection counter, objective width, assembly of the vector) live in the non-template `detail::ShapeConstraintEnforcer`, a second base of the adapter that is not an evaluator. It is an implementation detail in `Operon::detail`, present in the header only because the adapter template needs its complete type; the public shape surface is `ShapeConstraintSpec`, `ShapeConstraintPolicy`, `ShapeCertifier`, the adapter template and `MakeValueEvaluator`. Its public members (`Certifier`, `Policy`, `WorstValue`, `Violations`) are available on every adapter and, for erased `EvaluatorBase` pointers inside the CLI and probes, through `dynamic_cast<detail::ShapeConstraintEnforcer const*>`; its enforcement steps are protected, so the adapter template is a thin shell over `BaseEvaluator`'s own phases. Everything the adapter needs beyond the base evaluator is a `ShapeConstraintSpec` (constraints, policy, structural objectives, worst value, bound mode and options).
- Nesting is excluded at compile time: the adapter is `final` and sets `IsShapeConstrained`, which `Concepts::ValueEvaluator` rejects, so `ShapeConstrainedValueEvaluator<ShapeConstrainedValueEvaluator<X>>` does not name a type. Post-objectives are restricted to `TreePropertyEvaluator`, whose prediction phase is sealed (`final`, returns `nullopt`), so a value-producing post-objective cannot add a pass; construction rejects a post-objective with more than one objective and a policy valid for neither GP nor NSGA2.
- In one invocation the adapter's `Evaluate` runs `BaseEvaluator::Evaluate` once; its `Score` has the enforcer measure under the transform that pass fitted, before `BaseEvaluator::Score` rewrites the buffer, then enforce. HardReject returns `WorstValue` for the whole vector and skips post-objectives; a failed prediction or failed score returns `ErrMax` for the whole vector; Penalty adds the weighted violation to the primary score; ExtraObjective appends the violation last.
- The adapter reports `StructuralFitnessReusable() == false`, so offspring generators bypass the Zobrist fitness cache: a structural hit carries no coefficient identity and cannot certify the recipient genotype.
- `Stats()` adds the structural objectives' scoring counters to the base evaluator's own (scoring only: optimizer work is recorded once per fit by the run, never per objective); `CallCount` and the residual counters are the adapter's own because it is the value evaluator.
- Only the concrete value leaves that factories construct are unsealed: `Evaluator<DTable>`, the MDL, FBF and likelihood evaluators, and `JitEvaluator`. BIC and AIC stay `final`. Removing `final` from a leaf does not reopen `operator()`, which stays `final` in `EvaluatorBase`.

### CLI and factory wiring

The factories construct the typed adapter over the concrete leaf before erasing it to `std::unique_ptr<EvaluatorBase>`: `ParseEvaluator` (CLI) and `JIT::MakeJitObjects` take an optional `ShapeConstraintSpec`, and `MakeValueEvaluator<Leaf>(spec, dtable, args...)` builds either the plain leaf or `ShapeConstrainedValueEvaluator<Leaf>`. In JIT mode `all` the adapter is a `JitEvaluator`, so the JIT optimizer is given the JIT base subobject of the same object and shares its code cache; in mode `jac` the Jacobian-only JIT evaluator stays unconstrained and the constrained interpreter evaluator comes from `ParseEvaluator`.

| Algorithm / mode | Evaluator | Objective vector |
|---|---|---|
| GP, no constraints | value evaluator | `[error]` |
| GP, hard-reject / penalty | `ShapeConstrainedValueEvaluator<value>` | `[error]` (infeasible: `WorstValue`; penalty: `error + w * violation`) |
| NSGP, no constraints | `MultiEvaluator(error, length)` | `[error, length]` |
| NSGP, hard-reject | `ShapeConstrainedValueEvaluator<error>` with `{length}` structural | `[error, length]` (infeasible: all `WorstValue`) |
| NSGP, penalty | same | `[error + w * violation, length]` |
| NSGP, extra-objective | same | `[error, length, violation]` |
| NSGP, penalty + extra-objective | same | `[error + w * violation, length, violation]` |

NSGP keeps its crowded comparison.

## Verification

| Scenario | Observation (tests) |
|---|---|
| GP HardReject, Penalty and NSGP ExtraObjective | exactly one child prediction evaluation per call (counting value evaluator) |
| NSGP objective order | `[error, length, violation]`, including Penalty with ExtraObjective |
| JIT vs interpreter | identical scores and shape results for identical inputs and policy |
| Generic `MultiEvaluator` composition | children invoked only through `operator()`; order and nesting do not change any result |
| Scaling disabled / non-finite omission | adapter, score and JIT follow the declared policy |
| Failed prediction | result size equals `ObjectiveCount()`; `CallCount` incremented once |
| Cache | bound mode, transform bits and options identify measurements; the adapter bypasses the structural cache |
| API boundary | `Evaluate`, `Score` and the restoration hook are inaccessible outside the evaluator hierarchy and the lifecycle helper; nested adapters do not compile |

## Known limits

- A shape-constrained adapter owns the complete constrained objective vector, so `MultiEvaluator::Add` rejects an adapter child with `std::invalid_argument` (detected through the internal `detail::ShapeConstraintEnforcer`, with no new public virtual API); the adapter already owns its structural objectives.
- The fitness of a shape-constrained evaluator describes the evaluated (optimized-coefficient) model; under non-Lamarckian local search the restored genotype is not re-certified. Certify an exported model with `ShapeCertifier::MeasureCanonical`.
- Enclosures are conservative: failure to certify is not proof that a constraint fails.
