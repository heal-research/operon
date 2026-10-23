# Search and optimization

![Population search loop](../assets/diagrams/search-optimization.svg){ .diagram }

## Population transition

The GP and NSGA-II drivers share the same operator boundary. Initializers create valid trees, an offspring generator selects parents and applies crossover/mutation, optional local search changes coefficients, then the evaluator assigns fitness before reinsertion. The algorithm—not the operators—owns the population and stopping configuration.

| Concern | Main types |
| --- | --- |
| tree construction | `BalancedTreeCreator`, `TreeInitializerBase`, coefficient initializers |
| variation | `SubtreeCrossover`, `MultiMutation`, mutation operators |
| parent choice | `TournamentSelector` and comparison callback |
| local search | `CoefficientOptimizer`, `OptimizerBase` |
| survival | reinserters and comparison callback |
| drivers | `GeneticProgrammingAlgorithm`, `NSGA2` |

`GeneticAlgorithmConfig` bounds generations, evaluations, iterations, population size, pool size, and seed. Its evaluation budget is consumed by evaluator scoring work plus coefficient-optimizer work (residual and Jacobian evaluations of both)—not by offspring attempts, calls, optimizer time or cache lookups—so operators should not embed untracked objective calculations. Optimizer work is recorded by the run (`GeneticAlgorithmBase::Accounting()`) for every local search it invokes: cold initialization, and every offspring candidate, including candidates later rejected, losing a brood or polygenic selection, or discarded with an incomplete generation. A cache hit performs no local search and spends nothing.

The evaluation budget limits committed work. Initialization always commits (a zero population or pool size is a configuration error: `Run()` throws `std::invalid_argument`, `operon_gp`/`operon_nsgp` exit with status 1; a structural evaluator may cost zero work, while a larger initialization cost, e.g. with local search, is allowed and recorded as the overshoot). A generation commits only if its offspring pool was completely filled and everything it spent still fits in what the budget has left; offspring selection (`os`) additionally needs at most `MaxSelectionPressure * parents` attempts, one attempt being one candidate generated for a pool slot, whether it was evaluated or found in the transposition cache (see below for what this changes with a cache). Otherwise both drivers discard that generation and end the run: nothing is reinserted, `Generation()` is not incremented, the report callback is not invoked, `Parents()` holds the last committed generation and the worker RNG states are restored, so the committed result is the same at every thread count. A run can therefore finish with less than the budget used; what the discarded attempts spent is `EvaluationsDiscarded` in `BudgetReport()` (informational: it can depend on the thread count, and it does not include work done outside the generations, e.g. in a report callback). A time limit or a stop request that cuts a generation discards it the same way, but where that happens depends on timing. `IncompleteGeneration()` returns the attempted/completed/pool counts of the discarded generation (informational: they describe the work executed before the workers stopped and can differ between thread counts) and why it was discarded (`ResourceLimit`, `TimeLimit` or `StopRequested`, plus flags for the budget and the attempt cap seen exceeded: deterministic only when just one of the two limits is finite), or `nullopt` if the last generation was committed. `BudgetReport().End` says why the run ended, also when a committed generation crossed the attempt cap and nothing was discarded.

// keep both
An incomplete generation is not itself a stop reason and does not set `StopRequested()`: the run ends only because the stop condition that cut the generation short (evaluation budget, generator termination, time limit, stop request or a `true` report) holds. A generator-defined termination that clears again before the loop checks it, such as the selection-pressure limit, still ends the run. `StopRequested()` is true only after `RequestStop()` or a `true` report callback.

## Stop reasons

A run records *why* it stopped as a `StopMask` (`stop_reason.hpp`): `GenerationLimit`, `EvaluationLimit`,
`PressureLimit`, `TimeLimit`, `Callback`, `Signal`, `EngineError`. The drivers poll the four limits (generation count,
evaluator budget, `OffspringGeneratorBase::PressureExhausted()`, time) together with the external requests, record
every condition that holds with an atomic `fetch_or`, and so tolerate concurrent polls from worker threads.
`StopReasons()` is what was recorded, `TerminationReasons()` adds the conditions that hold in the final state, and
`PrimaryStopReason()` picks the one to report first (`engine_error`, `signal`, `callback`, `evaluation_limit`,
`pressure_limit`, `generation_limit`, `time_limit`). `RequestStop(source)` takes `Callback` (the default, and what a
report callback returning `true` records), `Signal` or `EngineError`; it is async-signal-safe. `StopRequested()` is
true only for these external requests. `Progress()` is the offspring accounting of the generation body started last
and `GenerationsCompleted()` the index of the last generation whose pool was filled.
`OffspringGeneratorBase::Terminate()` is the union of `BudgetExhausted()` and `PressureExhausted()`.

`operon_gp` and `operon_nsgp` write these, with the final counters read from the evaluator after `Run()` returns and
the final model's `ModelRef` (SHA-256 of `ModelSerialization`), as an `operon.final` JSON document with
`--final-json PATH --run-id ID --attempt-id UUID`. The file is published atomically, also when the run fails
(`exit_status` `error`, reason `engine_error`). With `--final-json`, SIGINT and SIGTERM stop the run gracefully
(reason `signal`) and the process exits with 128 plus the signal number; a second signal terminates it immediately.
The document's schema is `docs/schemas/final.schema.json`.

## Fitness and ordering

Fitness is a minimization vector. Single-objective GP commonly uses `SingleObjectiveComparison`; NSGA-II computes `Rank` and `Distance` and uses crowded comparison. `ParetoComparison` assumes minimization in every dimension. A custom comparator must be consistent for selection and reinsertion or the search transition no longer has a coherent ordering.

The `operon_nsgp --pareto-front` export is computed from retained parents, not `NSGA2::Best()` (which, after a completed generation, is the front of parents plus offspring). Fitness vectors are sorted lexicographically and adjacent epsilon-equivalent individuals are collapsed onto the first representative. The CLI sorters (`rs` and `ms`) then rank the representatives by exact dominance; `--epsilon` affects grouping, not their dominance comparisons. Stored `Rank` values are not used because they may describe the previously sorted parents and offspring rather than the retained parents. Unfilled offspring are not front members.

## Coefficient optimization

`CoefficientOptimizer` delegates to an `OptimizerBase`, usually `LevenbergMarquardtOptimizer`. It changes only `Optimize` constants in the tree. `FitOutcome` distinguishes an improving fit, a valid non-improving fit, an interpreter evaluation error, and a configuration error; `Diagnostics(outcome)` is available in every case. Treat an unsuccessful local fit as a candidate-level result, not as a reason to discard the structural tree before normal evaluation.
