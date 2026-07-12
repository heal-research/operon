# Operon Architecture Overview & Remediation Plan

**Status:** architecture review / planning note, not a committed spec.

This note maps how operon's layers are wired together today, names the
invariants that hold the design together, and lays out a phased, PR-sized
remediation plan aimed at C++23 idioms (deducing-this, `std::expected`,
`std::move_only_function`, concepts) and de-duplication. It is written to be
handed to implementing agents one item at a time.

---

## 1. The shape of the system

Operon is layered as a **data model → evaluation → optimization → search**
stack. Dependencies point strictly downward. CLI drivers
(`operon_gp`/`operon_nsgp`/`operon_enum`/`operon_parse_model`, and separately
pyoperon) own every lifetime and inject collaborators as non-owning
`gsl::not_null<... const*>` pointers — the core never reaches back up.

```
        CLI drivers  (operon_gp / operon_nsgp / operon_enum / operon_parse_model)
              |  constructs & wires everything, owns lifetimes
              v
   +----------------- SEARCH -----------------------+
   | GeneticAlgorithmBase -> GP, NSGA2               |   GrammarEnumerationAlgorithm
   |   (population/generation model)                |   (bottom-up DP, no population)
   +----------------+---------------------------------+
                    | uses operators
   +----------------v-- OPERATORS -------------------------------------+
   | creator . crossover . mutation . selector . reinserter            |
   | OffspringGenerator (Basic/Brood/Poly/OS) . CoefficientOptimizer   |
   | EvaluatorBase (Evaluator, Multi, Aggregate, MDL, FBF, AIC/BIC...) |
   +----------------+---------------------------------------------------+
                    |
   +----------------v-- OPTIMIZATION -----------+   ErrorMetric (SSE/MSE/R2/...)
   | OptimizerBase -> LM(Tiny/Eigen) . SGD . LBFGS|  Likelihood (Gaussian/Poisson)
   +----------------+------------------------------+  InformationCriteria (MDL/FBF)
                    |
   +----------------v-- EVALUATION ---------------+
   | Interpreter<T,DTable>  (forward + rev/fwd AD) |
   | Dispatch table  ->  Backend (Eve/MadEve/Eigen/Stl/JIT)
   +----------------+------------------------------+
                    |
   +----------------v-- DATA MODEL ---------------+
   | Node (POD) . Tree (flat postorder vector)    |
   | Dataset . Problem . PrimitiveSet . Grammar   |
   | Range . Individual . Hashes (Metro/Zobrist)  |
   +------------------------------------------------+
```

The single most important structural decision: **a `Tree` is a flat
`std::vector<Node>` in postorder**, not a pointer-linked tree. Everything
downstream (interpreter, hashing, autodiff, simplification) is a linear sweep
over that array with arity-driven index arithmetic (`nextArg`,
`Tree::Indices`). This is what makes the interpreter cache-friendly and the
whole thing fast. `Node::Ref` adds opt-in structural sharing (a backward
index reference) on top of the flat layout.

## 2. Load-bearing invariants

Three ideas hold the whole design together. Most of the remediation plan
either protects or generalizes one of them.

### I. A tree is a flat `vector<Node>` in postorder

Every downstream pass — interpretation, hashing, autodiff, `Reduce()`/
`Simplify()` — is a linear sweep with arity-driven index arithmetic.
`Node::Ref` adds opt-in structural sharing as a backward index. This is the
source of the engine's speed, and of the enumeration engine's budget-
accounting subtlety (`WorkingBudgetMargin` in `algorithms/enumeration.cpp`).

### II. Coefficient fitting and fitness scoring are separate concerns

`OptimizerBase` (LM-Tiny, LM-Eigen, SGD, LBFGS) minimizes an *internal* loss
to fit weights; its `OptimizerSummary::Success`/cost fields are an
implementation detail of fitting. `EvaluatorBase` + `ErrorMetric` produce the
*ranking* fitness (R2/NMSE/MSE/MAE/..., user-selectable via `--objective`).
`BasicOffspringGenerator::Generate` is the canonical wiring: optimize
coefficients, then score with the evaluator, then optionally discard the
fitted coefficients (Lamarckian toggle). `GrammarEnumerationAlgorithm::Run`
mirrors this exactly.

### III. The interpreter is mutable-through-`const`, so it has thread affinity

`Interpreter<T, DTable>` caches `primal_`/`trace_`/`context_` as `mutable`:
bind once per `(tree, range)`, re-evaluate cheaply via
`UpdateCoefficients`. Consequence: one interpreter (or scratch slot) *per
worker thread* — enforced today only by `gp.cpp`'s `slots[worker_id]`
convention, not expressed in the type.

## 3. Read of the codebase

### What's working well

- **Non-owning `gsl::not_null` injection** everywhere — lifetimes are the
  caller's job, and it's uniform throughout the codebase.
- **Flat-array `Tree` + linear-sweep everything** is the right core design
  and is exploited consistently by every subsystem.
- **`std::mdspan`** (C++23) is already the buffer abstraction for the
  interpreter (`Backend::Buffer`/`Backend::View`).
- The **generic autodiff trace sweep** (`ForwardTraceGeneric`/
  `ReverseTraceGeneric` in `interpreter/interpreter.hpp`) is a clean recent
  refactor: all four Jacobian entry points (`JacRev`, `JacFwd`,
  `JacRevVariable`, `JacFwdVariable`) collapse onto one
  `<bool Accumulate, LocalFactor>` templated sweep, differing only by a
  lambda and a column-mapping predicate.
- The **optimizer/evaluator separation** (invariant II above) is clean and
  consistently applied.
- `std::ranges` is used pervasively and correctly.

### Where the seams are

- **Two contract mechanisms for one thing.** `core/concepts.hpp` defines
  `Creator`, `Mutator`, `Crossover`, `Selector`, `Reinserter`,
  `EvaluatorCallable` concepts — but the live runtime path is entirely
  virtual dispatch through `OperatorBase<Ret, Args...>`. The concepts are not
  used to constrain any template today; only the vtable contract is load-
  bearing.
- **Order-encoded semantics in `NodeType`.** Arity and category
  (`IsNary`/`IsBinary`/`IsUnary`) are inferred from enumerator *position*
  (`Type < NodeType::Abs ⇒ arity 2`, `< Dynamic ⇒ arity 1`). Compact, but a
  reorder of the enum silently breaks arity inference.
- **Hand-rolled CRTP and getter triplets.**
  `EvaluatorBase::Evaluate(Derived const* self, ...)` is a static-helper CRTP
  pattern repeated in ~6 subclasses (`UserDefinedEvaluator`,
  `MultiEvaluator`, `MinimumDescriptionLengthEvaluator`,
  `FractionalBayesFactorEvaluator`, ...). `const`/non-`const`/`&&` getter
  triplets recur across `core/tree.hpp` (`Nodes()`) and
  `algorithms/ga_base.hpp` (`Parents()`, `Offspring()`, `Individuals()`,
  `Timings()`, `WorkerRngs()`, ...).
- **Duplicated stop/report idiom.** `GrammarEnumerationAlgorithm` cannot
  inherit `GeneticAlgorithmBase` (no population/generation model) but
  reimplements its `stopRequested_`/`StopRequested()`/`RequestStop()`/
  `ReportCallback` idiom verbatim.
- **Duplicated σ-profiling.** `MinimumDescriptionLengthEvaluator` and
  `FractionalBayesFactorEvaluator` both contain a near-identical
  "profile MLE σ̂ = √(SSR/n) from residuals" block.

---

## 4. Remediation plan

Ordered by payoff and safety, not by dependency — most items are independent
and can be handed to separate agents in parallel. Each item below is a
mini-spec: scope, rationale, files, the target C++23 idiom, things to watch,
and acceptance criteria.

### Conventions for the implementing agent

- Work in the `operon` repo directly (not the umbrella `operon-workspace`
  symlink folder, though either path resolves the same files).
- Build and test **only** inside operon's own devShell:
  `nix develop --command bash -c "cmake --build build ... && ./build/test/operon_test ..."`.
  Never use a different repo's devShell.
- Run `clang-tidy` before committing. Naming convention: PascalCase
  constants, no `k`-prefix.
- One item ≈ one PR. Keep commit messages to at most 3 lines (one-line why,
  no bullet list of every change).
- Cross-repo guardrail: any item marked **pyoperon-facing** below changes a
  signature the Python bindings consume. Before merging, check pyoperon's
  binding code and its pinned operon rev. Co-develop via a local flake
  `path:` override (temporarily point pyoperon's `operon.url` at
  `path:../operon`) rather than waiting on a published rev; revert the
  override before/at merge time unless the intent is to re-pin.

---

### Phase 1 — Mechanical dedup (safe, high-leverage, start here)

#### A1. Collapse getter pairs & manual CRTP with deducing-this — DONE

- **Effort:** medium · **pyoperon-facing:** yes
- **What:** Replace `const`/non-`const`/`&&` accessor overload sets with a
  single explicit-object-parameter member, and fold
  `EvaluatorBase::Evaluate(Derived const* self, ...)` into a deducing-this
  member so the ~6 subclasses stop forwarding `this` by hand.
- **Why:** Removes the codebase's most repeated boilerplate and its one
  hand-rolled CRTP pattern; fewer places for a `const`/non-`const` pair to
  drift apart.
- **Files:** `core/tree.hpp` (the `Nodes()` triplet), `algorithms/ga_base.hpp`
  (`Parents`/`Offspring`/`Individuals`/`Timings`/`WorkerRngs`),
  `operators/evaluator.hpp` (the static `Evaluate` helper).
- **Idiom:** C++23 explicit object parameter —
  `auto Nodes(this Self&& self)`; `auto operator()(this auto const& self, ...)`.
- **Watch:** Deducing-this members cannot be `virtual`. Keep the virtual
  `operator()` overrides as-is; only migrate the non-virtual helper and the
  plain getters. Verify pybind11 signatures still resolve after the change.
- **Acceptance:** Full suite green (`'~[performance]'` filter for fast
  iteration, then a full run); pyoperon builds against a local path override;
  no new clang-tidy findings.

#### A2. Factor a shared `ProfileSigma` helper — DONE

- **Effort:** small · **pyoperon-facing:** no
- **What:** Extract the duplicated "profile MLE σ̂ = √(SSR/n) from
  residuals" block shared by the MDL and FBF evaluators into one
  `static auto ProfileSigma(estimated, target) -> Scalar`.
- **Why:** Two copies of a numeric routine invite silent drift; one home
  makes the epsilon-clamp and the `Lik::UsesSigma` gating auditable in a
  single place.
- **Files:** `operators/evaluator.hpp` —
  `MinimumDescriptionLengthEvaluator`, `FractionalBayesFactorEvaluator`.
- **Watch:** FBF also derives an NLL from the profiled σ; keep that
  derivation in the caller — the helper should return σ only. Preserve the
  `epsilon()` clamp exactly.
- **Acceptance:** MDL/FBF numeric outputs are bit-identical on an existing
  regression dataset before/after the change.

#### A3. Extract a `StoppableAlgorithm` mixin — DONE

- **Effort:** small · **pyoperon-facing:** no
- **What:** Lift the `std::atomic<bool> stopRequested_` +
  `StopRequested()`/`RequestStop()` + `ReportCallback` idiom into one small
  base shared by `GeneticAlgorithmBase` and `GrammarEnumerationAlgorithm`.
- **Why:** The enumeration algorithm copied this idiom because it can't
  inherit the population-based base. A tiny shared mixin removes the copy
  and gives the callback-timing contract a single documented home.
- **Files:** new `algorithms/stoppable.hpp`; `algorithms/ga_base.hpp`,
  `algorithms/enumeration.hpp`/`.cpp`.
- **Watch:** `GeneticAlgorithmBase` hand-writes copy/assign around the
  atomic (`std::atomic<bool>` isn't copyable) — the mixin must reproduce
  that. Preserve acquire/release memory ordering.
- **Acceptance:** GP, NSGA2, and enumeration early-stop tests all still
  pass; no change to report-callback timing.

### Phase 2 — Safety rails (cheap insurance)

#### B1. Pin `NodeType` ordering with `static_assert` — DONE

- **Effort:** small · **pyoperon-facing:** no
- **What:** Add a block of `static_assert`s next to the `NodeType` enum
  asserting the category boundaries the code relies on (e.g.
  `Powabs < Abs`, `Square < Dynamic`), so a future reorder is a compile
  error rather than a silent arity miscompute.
- **Why:** Arity inference and `IsNary`/`IsBinary`/`IsUnary` depend entirely
  on enumerator position — the one place order carries semantics. Free
  insurance against a categorically silent bug class.
- **Files:** `core/node.hpp`.
- **Idiom:** `static_assert` on the `constexpr` category boundary values.
- **Acceptance:** Builds unchanged; a deliberate local reorder makes the
  build fail (verify once locally, then revert the reorder).

#### B2. Name the interpreter's thread-affinity contract in the type — DONE (documentation-only variant)

- **Effort:** medium · **pyoperon-facing:** no
- **What:** Make invariant III self-documenting. Minimum: a class-level
  contract comment plus a note on the mutable caches
  (`primal_`/`trace_`/`context_`). Stronger: hand out a per-thread
  `ScratchContext` handle the caller must hold, so "one instance per worker"
  is expressed structurally rather than by convention.
- **Why:** Today the rule lives only in `gp.cpp`'s `slots[worker_id]` usage.
  Anyone reusing an `Interpreter` across threads hits a silent data race.
- **Files:** `interpreter/interpreter.hpp`; call sites in `algorithms/gp.cpp`,
  the `Evaluator<DTable>` family.
- **Watch:** Start with the documentation-only variant to de-risk; the
  handle refactor is a follow-up only if the team wants structural
  enforcement. Don't regress the bind-once/re-evaluate fast path
  (`InitContext`/`UpdateCoefficients`).
- **Acceptance:** No perf regression on the interpreter benchmark
  (`test/source/performance`); contract is visible at the class
  declaration.

### Phase 3 — Contract clarity (pick a lane)

#### C1. Resolve the concepts-vs-vtable duplication

- **Effort:** medium · **needs a direction call first**
- **What:** Decide the role of `core/concepts.hpp`. Recommended: keep
  runtime `OperatorBase` dispatch (pyoperon and the CLI factories need
  dynamic wiring) and *use* the concepts to constrain the templates that
  build/adapt operators — better compile errors, real documentation, zero
  runtime change. Alternative: if they're going to stay unused, delete
  them so there's exactly one contract, not two.
- **Why:** Two parallel descriptions of the same contracts, with only one
  load-bearing, is a standing source of confusion for new contributors.
- **Files:** `core/concepts.hpp` and the operator construction/adapter
  sites that would gain constraints.
- **Idiom:** Constrain template parameters with the existing concepts;
  `static_assert(Concepts::Mutator<T>)` at adapter boundaries.
- **Decide:** This needs a human call (constrain vs. delete) before an
  agent proceeds — surface it as a question rather than guessing.
- **Acceptance:** Exactly one contract mechanism remains authoritative;
  concept names either constrain real templates or are removed entirely.

#### C2. `std::move_only_function` for callbacks

- **Effort:** small · **pyoperon-facing:** yes
- **What:** Swap `std::function` for `std::move_only_function` on
  invoke-and-move-only callback members: `ReportCallback` and the
  enumeration engine's `onNovelExpression_`/`SetOnNovelExpression` hook.
- **Why:** These are only ever called and moved; dropping the copyability
  requirement permits move-only captures (e.g. a captured `unique_ptr`
  progress sink) and is marginally cheaper to construct.
- **Files:** `algorithms/ga_base.hpp` (the `ReportCallback` alias),
  `algorithms/enumeration.hpp`.
- **Watch:** `ReportCallback` is passed by value into several `Run()`
  signatures and used from Python — confirm pybind11 can still bind it, or
  keep a `std::function`-typed shim at the binding boundary if not.
- **Acceptance:** GP/NSGA2/enumeration report callbacks work from both C++
  and pyoperon; a move-only-captured callback compiles and runs.

### Phase 4 — Error-handling modernization (largest blast radius, do last)

#### D1. `std::expected` at the CLI / lookup boundaries

- **Effort:** medium · **pyoperon-facing:** maybe
- **What:** Return `std::expected<T, Error>` from fallible lookups that
  currently return a bare `optional` (losing the *why*) or throw:
  `Dataset::GetVariable`, CLI option parsing, target-hash resolution in the
  CLI entry points.
- **Why:** Carries the failure reason to the call site without exceptions
  and reads cleanly at the CLI boundary, where the errors are user-facing.
- **Files:** `core/dataset.hpp`, CLI option-parsing utilities
  (`cli/source/util.*`, `cli/source/operator_factory.*`),
  `cli/source/*.cpp`.
- **Idiom:** C++23 `std::expected`; `.transform`/`.or_else` monadic
  chaining at call sites instead of nested `if`s.
- **Watch:** Do `Dataset::GetVariable` in isolation first — it has the
  widest fan-out across CLIs. pyoperon may depend on the current `optional`
  return; provide a shim if so.
- **Acceptance:** CLI error messages unchanged or improved; callers handle
  the error arm explicitly; no new throws introduced on the hot path.

#### D2. Richer optimizer result type

- **Effort:** large · **pyoperon-facing:** yes
- **What:** Reshape the `OptimizerSummary::Success`-bool-plus-costs return
  into an explicit outcome type (or
  `std::expected<FitResult, FitFailure>`), so "did fitting help?" isn't
  reconstructed from comparing cost fields at each call site.
- **Why:** The current shape is exactly what a result type is for; today
  the meaning of failure is spread across `Success`, `InitialCost`, and
  `FinalCost`, and every caller re-derives it independently
  (see `detail::CheckSuccess` in `optimizer/optimizer.hpp`).
- **Files:** `optimizer/optimizer.hpp`, all `OptimizerBase` subclasses,
  `operators/local_search.cpp`, `operators/generator/*.cpp`,
  `algorithms/enumeration.cpp`.
- **Watch:** Highest blast radius in this plan — `OptimizerSummary` is
  threaded through every offspring generator and the pyoperon bindings.
  Sequence it last; consider keeping `OptimizerSummary` as a compatibility
  view over the new type during migration rather than a hard cutover.
- **Acceptance:** Every generator and pyoperon compile against the new
  type; Lamarckian coefficient-restore behavior is unchanged; full suite
  green.

---

## 5. Sequencing summary

A dependency-free reading of the plan: agents can pull any Phase 1–2 item in
parallel; Phase 3–4 items want a decision or careful staging first.

| ID  | Item                                  | Phase | Effort | pyoperon | Status | Blocker        |
|-----|----------------------------------------|:-----:|:------:|:--------:|:------:|----------------|
| A1  | Deducing-this: getters & CRTP           |   1   | Medium |   Yes    | Done   | —              |
| A2  | Shared `ProfileSigma`                   |   1   | Small  |    No    | Done   | —              |
| A3  | `StoppableAlgorithm` mixin              |   1   | Small  |    No    | Done   | —              |
| B1  | `NodeType` order asserts                |   2   | Small  |    No    | Done   | —              |
| B2  | Interpreter thread-affinity contract    |   2   | Medium |    No    | Done*  | —              |
| C1  | Concepts vs vtable                      |   3   | Medium |    No    |        | Direction call |
| C2  | `move_only_function` callbacks          |   3   | Small  |   Yes    |        | —              |
| D1  | `std::expected` lookups                 |   4   | Medium |  Maybe   |        | —              |
| D2  | Optimizer result type                   |   4   | Large  |   Yes    |        | Stage last     |

\* B2 landed the documentation-only variant (class-level contract comment).
The structural `ScratchContext` handle described as a stronger option remains
open as a follow-up if the team wants compile-time enforcement.

## 6. Scope note

Every item is incremental — none requires a rewrite. Build and test only
inside operon's own `nix develop` devShell. Items marked **pyoperon-facing**
touch the binding surface: co-develop with a local flake `path:` override
and re-check the pinned rev before merging.
