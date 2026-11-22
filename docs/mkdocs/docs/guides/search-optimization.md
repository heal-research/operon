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

A run records why it stopped as a `StopMask` (`stop_reason.hpp`): `GenerationLimit`, `EvaluationLimit`,
`PressureLimit`, `TimeLimit`, `GeneratorTermination`, `Callback`, `Signal`, `EngineError`. The drivers poll the four
limits (generation count, evaluator budget, `OffspringGeneratorBase::PressureExhausted()`, time) together with
external requests, record every condition that holds with an atomic `fetch_or`, and so tolerate concurrent worker
polls. `StopReasons()` is what was recorded, `TerminationReasons()` adds final-state conditions, and
`PrimaryStopReason()` uses `engine_error`, `signal`, `callback`, `generator_termination`, `evaluation_limit`,
`pressure_limit`, `generation_limit`, `time_limit`. `RequestStop(source)` takes `Callback`, `Signal` or
`EngineError`; `StopRequested()` is true only for external requests.
`OffspringGeneratorBase::Terminate()` remains virtual for custom generators. Its default combines `BudgetExhausted()`
and `PressureExhausted()`; an override without a more specific reason is recorded as `generator_termination`.

`operon_gp` and `operon_nsgp` write these, with the final counters read from the evaluator after `Run()` returns and
the final model's `ModelRef` (SHA-256 of `ModelSerialization`), as an `operon.final` JSON document with
`--final-json PATH --run-id ID --attempt-id UUID`. The file is published atomically (a flushed temporary sibling
replaces any existing file in one step; POSIX `rename` plus directory `fsync`, Windows `MoveFileExW` with
`MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH`), also when the run fails
(`exit_status` `error`, reason `engine_error`). Final records require a fresh run: `--final-json` rejects `--resume`
until checkpoints persist the prior evaluation counters. With `--final-json` or `--generation-log`, SIGINT and SIGTERM
stop the run gracefully (reason `signal`) and the process exits with 128 plus the signal number; a second signal
terminates it immediately. The final document's schema is `docs/schemas/final.schema.json`.

## Lossless generation records

Both CLIs accept `--generation-log DIR --run-id HEX64 --attempt-id UUID`. The directory must not contain an earlier
stream. Generation 0 is `initialized`, followed by one `generation_complete` per fully filled offspring pool.
`--segment-generations 1000` and `--segment-seconds 5` seal at the first reached threshold, checked at each completed
record; `--sync segment` is the default. Every line is written directly to the OS in `generation-000000.jsonl.part`.
Sealing syncs the model sidecar and segment, closes the segment and renames it to `.jsonl`. On POSIX it then syncs the
directory, which makes the rename durable; on Windows the file contents are flushed (`_commit`) but the directory
entry is not, so the rename itself is not guaranteed to survive power loss.
One persistent seal worker overlaps that durability sequence with the next generation's computation. Before writing
another record the callback waits for the pending seal, so there is never more than one `.part`; worker errors are
re-thrown on the callback thread or by `Close()`. `Drain()` waits for an already-requested count/time seal.
`--sync record` additionally syncs after every record. SIGKILL can leave one `.part`, whose trailing incomplete line
must be ignored; on POSIX only sealed segments are power-loss durable, on Windows a sealed segment's contents are
durable but its name may revert to `.part`. A failed `Close()` is terminal: later calls rethrow the same error.
Open/write/serialization/sync errors fail the run.
With `--final-json`, the final document includes sealed segment ranges, byte-identical SHA-256 hashes, record count
and instrumentation error count. A stream can also be requested without a final document.
Each line follows `docs/schemas/generation.schema.json`.

The library `GenerationRecorder` captures objective/length/depth/rank reductions in one population pass before the
table reporter. The reporter then hands it the exact selected, linearly scaled tree and NMSE values, avoiding another
model-selection or prediction pass. The existing table and `--report-json` v1 remain unchanged. Objective means use
double accumulation in population order, excluding non-finite values, `ErrMax`, and `Individual::ShapeRejected`.
Means are rounded once to native build precision for JSON, consistently with best/min/max; accumulation does not
overflow at float precision, and a uniform population's emitted mean equals its emitted extrema.
The shape adapter records that flag when hard rejection occurs; it is not inferred from the configurable worst
fitness value, and describes evaluated fitness even under non-Lamarckian search. Resume with a generation log is
rejected: this stream is restart-only.

`models.jsonl.txt` is the per-attempt model sidecar (the `.txt` suffix distinguishes opaque objects from contract
documents). It contains a `model_ref` and lossless `ModelSerialization` string only when the selected model changes;
every generation references its SHA-256. Equality checks all serialized node attributes and coefficient bits.
NSGA-II uses `--model-selection` exactly as the table does. `--pareto-front PATH` supplies the final front hash;
with a stream and no explicit path, the front is stored in `DIR/pareto-front.json.txt`.
Shape feasibility is certified for the exported model on change, not inferred from its possibly non-Lamarckian score.
Validation NMSE is unavailable and declared as such. Scalar objectives use native-precision shortest round-trip
numbers; non-finite metrics are JSON `null` with counts/status.

`time.report_s` includes reductions, model selection, prediction metrics, changed-model serialization/certification,
and JSON construction up to the timing field. The final write/fsync cannot be included in its own immutable bytes;
measure that cost end-to-end. Generation wall time is the interval between callback entries; process elapsed time
starts at `main()`. OS pressure counts evaluator calls since `Prepare()` (cache hits do not increment it).

## Fitness and ordering

Fitness is a minimization vector. Single-objective GP commonly uses `SingleObjectiveComparison`; NSGA-II computes `Rank` and `Distance` and uses crowded comparison. `ParetoComparison` assumes minimization in every dimension. A custom comparator must be consistent for selection and reinsertion or the search transition no longer has a coherent ordering.

The `operon_nsgp --pareto-front` export is computed from retained parents, not `NSGA2::Best()` (which, after a completed generation, is the front of parents plus offspring). Fitness vectors are sorted lexicographically and adjacent epsilon-equivalent individuals are collapsed onto the first representative. The CLI sorters (`rs` and `ms`) then rank the representatives by exact dominance; `--epsilon` affects grouping, not their dominance comparisons. Stored `Rank` values are not used because they may describe the previously sorted parents and offspring rather than the retained parents. Unfilled offspring are not front members.

## Coefficient optimization

`CoefficientOptimizer` delegates to an `OptimizerBase`, usually `LevenbergMarquardtOptimizer`. It changes only `Optimize` constants in the tree. `FitOutcome` distinguishes an improving fit, a valid non-improving fit, an interpreter evaluation error, and a configuration error; `Diagnostics(outcome)` is available in every case. Treat an unsuccessful local fit as a candidate-level result, not as a reason to discard the structural tree before normal evaluation.
