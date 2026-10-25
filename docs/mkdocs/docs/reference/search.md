# Search and enumeration API

## Genetic programming

Header: [`operon/algorithms/gp.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/algorithms/gp.hpp)

```cpp
GeneticProgrammingAlgorithm gp {
    config, &problem, &treeInitializer, &coefficientInitializer,
    &offspringGenerator, &reinserter
};
gp.Run(executor, rng, report);
```

`GeneticProgrammingAlgorithm` is assembled from borrowed components. `Run(tf::Executor&, RandomGenerator&, ReportCallback, warmStart)` uses the supplied executor; the convenience overload creates its own execution context and accepts a thread count. The callback is the observation/stop hook. The supplied `Problem`, initializers, generator, and reinserter MUST outlive the algorithm.

`GeneticAlgorithmConfig` supplies `Generations`, `Evaluations`, `Iterations`, `PopulationSize`, `PoolSize`, and `Seed`. Population algorithms consume scoring work (evaluator `Stats()`) plus coefficient-optimizer work (the run's `Accounting()`) under the evaluation budget; `DetailedStats()` reports both with disjoint fields. A warm start retains the existing population; otherwise initialization happens before the first transition. Cold initialization runs `LocalSearch` directly and records its fits; a warm start re-evaluates without local search and spends no optimizer work.

## Transposition cache

Header: [`operon/hash/zobrist.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/hash/zobrist.hpp)

```cpp
Zobrist cache { rng, maxLength, problem.GetInputs(),
    { .MaxAge = 0, .Visibility = CacheVisibility::Committed } };
config.Cache = &cache; // borrowed; must outlive the run
```

`Zobrist` caches the fitness vector of a tree by its structural hash (topology, variables, `Optimize` flags; not coefficient values), so a structurally identical offspring skips scoring and local search. It is used by the offspring generators when `GeneticAlgorithmConfig::Cache` is set and the evaluator's `StructuralFitnessReusable()` is true. `JitZobrist` takes the same `ZobristOptions`.

| `ZobristOptions` field | Meaning |
| --- | --- |
| `MaxAge` | `0` keeps entries forever. Otherwise an entry older than `MaxAge` generations is expired. Freshness/evolvability knob, not a memory bound. |
| `Visibility` | `Committed` (default) or `Immediate`, below. |

**`CacheVisibility::Committed`.** An entry is stamped with the generation clock when it is inserted (the algorithm sets the clock to `Generation() + 1` before it generates the offspring of a generation). Lookups made inside a slot ignore shared-table entries stamped with the current clock, so no slot can observe what another slot inserted in the same generation; those entries become visible when the clock advances. Each offspring slot additionally sees what it inserted itself (`Zobrist::SlotScope`, opened by `GeneticProgrammingAlgorithm` and `NSGA2` around every slot): a slot's own earlier attempts are reused, nobody else's. When several slots insert the same structure in one generation, the lexicographically smallest fitness vector is kept (`std::strong_order` per component, so NaN and signed zeros are ordered), and an entry of an earlier generation is never overwritten. With `MaxAge > 0`, expiry runs in `SetGeneration` at the generation boundary instead of inside lookups; entries stamped later than the clock (it moved back, e.g. a second `Run()` on the same cache) are expired too, as in `Immediate`. Hit/miss decisions then do not depend on the order in which slots run. The initial population is scored without the cache; a cache that was never given a clock value stamps entries with `0`, which are visible to slots once the clock moves past `0`.

**`CacheVisibility::Immediate`.** Entries are visible to every slot at once, the first writer keeps the entry, and expired entries are removed lazily inside the lookup that finds them. This is the behaviour before `CacheVisibility` existed. With more than one thread, hits depend on arrival order, so results can differ between thread counts and between runs.

Cost of `Committed`: lookups that used to hit an entry inserted by another slot of the same generation become evaluations. The search trajectory then changes, so the number of evaluations a run consumes changes too, in either direction; it is not a fixed surcharge. Measured at population = pool = 500, `GeneticProgrammingAlgorithm`, one thread, the extra evaluations in the first generation alone (both modes start it from identical parents) are about 17 for `basic`, 39 for `os`, 380 for `brood` and 483 for `poly`. Whole-run totals for `os` differed by +5% to +66% across four seeds. Measure with your own configuration. A slot's own reuse is kept, which matters mainly for `brood` and `poly`.

`Committed` makes the cache hit/miss decisions independent of slot arrival order. Independence of the whole run from the thread count additionally requires that no stop condition cuts a generation: the evaluation budget, offspring-selection pressure (`os` generator), the time limit and a stop request. A generation that is in flight when one of them fires can still complete with more threads, which this option does not change. JIT evaluation is not covered: results with `--jit` differ between runs even at one thread, independently of the cache, and `--jit-min-visits` above 1 makes compilation depend on arrival order.

A generation that is cut short is discarded (see `IncompleteGeneration()`). In `Committed` mode the drivers then call `Zobrist::AbortGeneration()`, which erases every entry stamped with the attempted generation, so the discarded work never becomes visible and a warm resume does not see it. `Immediate` keeps today's behaviour and does not roll back. Expiry already applied, the hit and lookup counters and the consumed worker RNG state are not undone. The same rollback runs when an exception leaves a generation that was opened but not yet committed (a task of the generation threw): the drivers call `AbortGeneration()` after the run has been joined, and not for an exception thrown by the report callback, which runs after the generation was committed. Nothing else is rolled back: after a task exception the parents, ranks and random states may be half-way through a generation, so the algorithm object is **not resumable** (a later `Run(..., warmStart=true)` throws `std::logic_error`; create a new object). What holds is that the cache has no entry of the failed generation, `Generation()` is the last committed generation, and a new algorithm object can reuse the cache.

Slot state is held by the `SlotScope` objects, tied to the cache instance and the thread that opened them, so two caches, or two runs sharing one cache, do not see each other's slots. A scope must be destroyed on its own thread, innermost first. Calling `TryGet`/`Insert` outside a scope is allowed: nothing is then slot-local and, in `Committed` mode, a lookup outside any scope sees every entry, including those stamped with the current clock value, so a default-constructed cache used by code outside the GA loops hits what it inserted.

The command-line programs expose both options as `--cache-max-age` and `--cache-visibility=immediate|committed` (default `committed`), effective only with `--transposition-cache`.

**Checkpoints.** `Zobrist::Export()` returns the cache as a `ZobristState` (every entry with its insertion generation, the generation clock, the maximum age, the visibility and a fingerprint of the random table the keys were computed with; entries are ordered by key, so the encoding is deterministic) and `Zobrist::Import(state, objectiveCount)` restores it into a cache of the same configuration. The checkpoint of the command-line programs (`Serialization::Checkpoint`, BEVE, format version 3) carries it next to the parent population, the master and worker random states, the generation and a run fingerprint, in compact parallel arrays (about 8 + 4 + 4 + 4 bytes per entry plus 4 bytes per fitness value). A run resumed with `--resume` continues exactly as the uninterrupted run would have, with the cache on as well (the same population, worker random states and cache entries at every later generation), unless `--evaluations` (or another limit that cuts a generation) binds: the evaluation budget applies per invocation, so a resumed run gets a fresh `--evaluations` budget and is charged one evaluation per individual for the re-evaluation of the parents at the warm start.

*Run fingerprint.* The table fingerprint only says that the keys mean the same thing; it says nothing about what the cached fitness values mean. The command-line programs therefore store a fingerprint of everything that determines what the run computes: the build identity (`build.identity`: the Operon revision, the scalar precision, the architecture and the x86-64 ISA level), the dataset contents (after `--standardize`) and the resolved training and test ranges, target and inputs, the parsed content of `--shape-constraints-config` (`shape.constraints`; the path of the file is not part of it, and neither is the order of the keys or the white space in it), the resolved algorithm configuration and objective count, and the effective value of every other option except the excluded ones below. `--resume` compares it before it imports the cache or restores any fitness, and rejects a mismatch naming the differing entries (`dataset.content`, `dataset.train`, `shape.constraints`, `--objective`, `--offspring-generator`, ...). A mismatch of the build identity is reported as "written by another build" and nothing else. The identity is not strict: it is the revision string, the scalar precision, the architecture and the x86-64 ISA level; the compiler and its flags are not part of it, and builds without a git revision (`unknown`) cannot be told apart. The ISA level is part of the identity because the levels that enable FMA and wider vectors change the rounding of floating point results; the compiler and its flags are not part of it, and a build without a git revision records the revision `unknown`, which then tells builds apart by the other components only.

What may differ between the checkpointed run and the resumed one is exactly this: `--threads`; `--generations` (a finite value must be at least the generation of the checkpoint, `unlimited` always fits); `--evaluations`; `--timelimit`; what is only reported or written (`--checkpoint-interval`, `--checkpoint-file`, `--pareto-front`, `--model-selection`, `--mdl-likelihood`, `--probes-config`, `--report-json`, `--debug`); the options that name the data (`--dataset`, `--train`, `--test`, `--target`, `--inputs`, `--shape-constraints-config`: the data content, the resolved ranges, target and inputs and the constraints are compared, not the paths or spellings); and `--seed`, only when the checkpoint has no cache. `--transposition-cache`, `--cache-visibility` and `--cache-max-age` must agree with the cache stored in the checkpoint (a seed matters through the cache's random table, so with a cache it must be the same too). Every other option is part of the fingerprint. The effective value is compared (an explicitly given default is the same as omitting it); resume with the options of the checkpointed run.

*Failures.* `--resume` with a file that does not exist starts a fresh run with a warning (the "resume if present" use of a restartable job). Every other problem is an error and exits with status 1, before any state of the algorithm is replaced: a file that cannot be read, is truncated or corrupt, was written by another format version, carries inconsistent arrays, has another population size, worker count, build or fingerprint, a generation above a finite `--generations`, or a cache that this run's cache cannot continue: a checkpoint with a cache resumed without `--transposition-cache`, a checkpoint without one resumed with it (neither direction is silently repaired), another visibility, maximum age, random table, or an entry whose fitness length is not the evaluator's objective count. The parents are also checked against the run: every `Variable` node must be an input of the problem, every `Function` node a registered and enabled primitive of its primitive set, and no tree may be longer than `--maxlength`. `Zobrist::Import` throws `std::invalid_argument` and leaves the cache unchanged on any failure: it validates the state (including an entry inserted after the state's generation clock), builds the new content aside and exchanges it in only when it is complete. A checkpoint of format version 2 (without the cache) is rejected with a message naming the version.

*Decoding is bounded.* A checkpoint file may not exceed `Serialization::MaxCheckpointFileBytes` (1 GiB), and `SaveCheckpoint` refuses to write a file the reader would refuse (an error is printed and the run continues without that checkpoint). The declared element counts are limited per field: at most 2^20 individuals and 2^20 worker random states (one per population or pool slot, not per thread), 65535 nodes per tree, 64 objectives per individual, 2^26 cache entries and 4096 fingerprint entries; a count can never exceed the bytes that are left, so a small or hostile file cannot make the reader allocate gigabytes. `ToZobristState` and the checkpoint reader validate the parallel arrays with checked arithmetic (equal lengths, non-zero fitness lengths that add up to the number of values, ascending keys) and every tree (node types, arities, backward `Ref` targets, postfix structure, `Tree::Validate()`, no empty parent) before anything is used, and `--resume` then checks the trees against the dataset and primitive set; malformed input gives an error, not undefined behavior.

*Integrity and writing.* A serialized checkpoint is an envelope (magic, format version and a 64-bit checksum of the payload) followed by the payload; the reader verifies the checksum before it decodes anything, so any damaged byte is reported as a corrupt checkpoint. `SaveCheckpoint` writes `<path>.tmp` and renames it over the destination: on POSIX the replacement is atomic; on Windows an existing destination is removed and the rename retried, which is not failure-atomic. Concurrent writers to the same path are not supported (the temporary name is fixed; this scheme is the one `main` already had). The run keeps going when a checkpoint cannot be written.

*Warm start.* A warm start re-evaluates the parents (to run `Prepare()` of the evaluator) but keeps the fitness the checkpoint stored, and for `NSGA2` its `Rank` and `Distance`, as long as the evaluator has the same objective count: the fitness of an individual that came from a cache hit or from local search that did not write its coefficients back cannot be reproduced by evaluating the genotype, and selection reads all three. With another objective count the re-evaluated fitness and the freshly sorted rank and distance are kept. The re-evaluation does not detect an evaluator that computes something else with the same number of objectives; the caller has to establish that (the command-line programs use the run fingerprint). Hit and lookup counters, evaluation counters and timings restart at zero when a run is resumed, and the state of the JIT backend is not part of a checkpoint. The generation limit is `Generation() >= config.Generations`, so a population restored at a generation above the limit runs no further generation.

*Resumability.* An algorithm object is resumable (`IsFitted()`) only if its last `Run()` completed, or if a checkpoint restore replaced its population and marked it fitted (`ResumeFromCheckpoint` of the command-line programs does; code that restores a checkpoint itself sets `IsFitted() = true`). `Run()` captures `IsFitted()` before it resets anything, clears it before the first task starts and sets it again only in its final task, so an exception anywhere in a run (population creation, a generation, the report callback, also at generation 0) leaves the object not resumable, whatever the run before it did. A warm `Run(..., warmStart = true)` on an object that is not resumable throws `std::logic_error`; this includes an object that was never run (a warm start has nothing to continue). A new cold `Run()` is always allowed.

*Portability.* The hashed inputs of the random table, its fingerprint and the structural keys are encoded as little-endian integers, but the run fingerprint is not: the dataset content hash and the hashes of the configuration use the native byte representation of the scalars and integers, and so does the hash function over them. Run fingerprints, and with them cache-bearing (and all other) checkpoints of the command-line programs, are therefore only valid on the architecture (byte order, scalar representation) that wrote them; a checkpoint moved to another architecture is rejected as written by another build or with different settings. A cache-bearing checkpoint also depends on the representation the keys are computed from (the node hash values, the scalar precision and the hash function): a mismatch is reported as a different random table or incompatible representation.

!!! note "Binding change"
    The `std::size_t maxAge` constructor parameter of `Zobrist` and `JitZobrist` (and of `JIT::MakeJitObjects`) is replaced by `ZobristOptions`, and the default visibility is `Committed`. Code that passed a max age must pass `{ .MaxAge = n }` (or `.Visibility = CacheVisibility::Immediate` to keep the old behaviour). `Zobrist` gains a data member, so the layout of `Zobrist`, `JitZobrist` (the JIT cache) and of the classes that contain one, `EnumerationEngine` (member `zobrist_`) and `GrammarEnumerationAlgorithm` (member `engine_`), changes too: downstream bindings and enumeration consumers must be rebuilt against this header. See also [Execution](execution.md#search-driver-layout).

## Operator assembly

Headers: `operon/operators/{creator,initializer,generator,selector,crossover,mutation,reinserter,local_search}.hpp`

| Role | Typical implementation | Responsibility |
| --- | --- | --- |
| creator | `BalancedTreeCreator` | create a valid topology for inputs and primitive set |
| tree initializer | `UniformTreeInitializer` | choose initial lengths/depths and invoke creator |
| coefficient initializer | `NormalCoefficientInitializer` | initialize `Optimize` constants |
| selectors | `TournamentSelector` | choose parents under a comparison |
| variation | `SubtreeCrossover`, `MultiMutation` | produce valid bounded offspring |
| local search | `CoefficientOptimizer`, `LocalSearch` | apply an `OptimizerBase` result to coefficients; `LocalSearch` returns the fit outcome, which the run records in its `OptimizerAccounting` |
| offspring generator | `BasicOffspringGenerator` | compose selection, variation, local search, evaluation; forwards the run's `OptimizerAccounting&` to every scored candidate |
| reinserter | `ReplaceWorstReinserter` | combine offspring and parent population |

The comparison supplied to selectors and reinserters defines the meaning of “better.” Use the same minimization convention across both. For NSGA-II, use rank/crowding-aware ordering after non-dominated sorting rather than an arbitrary scalar comparison.

## Grammar enumeration

Header: [`operon/algorithms/enumeration.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/algorithms/enumeration.hpp)

```cpp
EnumerationConfig config { .MaxComplexity = 12, .TopK = 20, .Ranking = EnumerationRanking::MinimumDescriptionLength,
                            .EvaluationBufferSize = problem.TrainingRange().Size() };
auto scorer = MakeMdlScorer<DTable, GaussianLikelihood<Operon::Scalar>>(&problem, &dtable);
GrammarEnumerationAlgorithm algorithm { config, grammar, &optimizer, std::move(scorer), setupRng };
algorithm.Run(fitRng, report);
for (auto const& result : algorithm.BestTrees()) { /* result.Score, result.Tree, result.CanonicalKey, ... */ }
```

`EnumerationEngine(grammar, maxComplexity, rng)` builds canonical candidate buckets bottom-up. `Bucket(nonterminal, budget)` returns a read-only view after `Build()`. The engine has no per-candidate callback; it only discovers and deduplicates distinct trees per `(nonterminal, budget)`. The engine is move-only and `Build()` is not a concurrent candidate-generation API.

`GrammarEnumerationAlgorithm` wraps the engine with a group/fit/rank pass: every stored candidate is canonicalized (`CanonicalizeEnumerationTree`) and grouped into canonical classes, one representative per class is fit via `CoefficientOptimizer`, then scored with the caller-supplied `EnumerationScorer` (`MakeMdlScorer` for MDL ranking, `MakeObjectiveScorer` for the pre-existing evaluator-based ranking) and the `TopK` ascending-`Score` results are retained as `EnumerationResult`s, with the canonical key as the strict tie-break on equal score. It is single-shot: construct a fresh object for another run. `optimizer->Iterations()` MUST be greater than zero, because a zero-iteration fit makes ranking tied placeholder coefficients meaningless.

## Stop control

`GrammarEnumerationAlgorithm` derives from `StoppableAlgorithm`. `RequestStop()` sets its persistent stop request; `Run` also stops when the report callback returns true. Enumeration observes both only after completing a complexity level, so the current level's work remains internally consistent.
