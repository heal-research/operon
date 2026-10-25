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

A generation counts only if its offspring pool was completely filled. If a stop condition (evaluation budget, generator termination such as `OffspringGeneratorBase::Terminate(OptimizerAccounting const&)` and the offspring-selection pressure limit, time limit, stop request) fires while the pool is being filled, both drivers discard that generation: nothing is reinserted, `Generation()` is not incremented and the report callback is not invoked, so `Parents()` holds the last completed generation. The work performed for the discarded generation stays in the evaluator and optimizer counters and in `DetailedStats()` (executed work is never rolled back) and counts against the budget. `IncompleteGeneration()` returns the attempted/completed/pool counts of such a cut-short generation, or `nullopt` if the last generation completed.

## Transposition cache

With `GeneticAlgorithmConfig::Cache` set, offspring that are structurally identical to an earlier one reuse its fitness instead of being scored again. By default (`CacheVisibility::Committed`) an entry becomes visible to other offspring slots only in the next generation, which makes the cache hit/miss decisions independent of slot arrival order. The whole run is independent of the thread count only if no stop condition cuts a generation (evaluation budget, offspring-selection pressure, time limit, stop request); JIT evaluation is not covered. Lookups that used to hit an entry of the same generation become evaluations, which changes the search trajectory and therefore the number of evaluations a run consumes, in either direction. `CacheVisibility::Immediate` shares entries at once and depends on thread timing. See the [search reference](../reference/search.md#transposition-cache). Checkpoints written with `--checkpoint-interval` carry the cache, so `--resume` continues a run with the cache on exactly like the uninterrupted run, unless `--evaluations` (or another limit that cuts a generation) binds (a resumed run gets a fresh evaluation budget and is charged one evaluation per parent for the re-evaluation at the warm start); resume with the same `--seed`, `--cache-visibility` and `--cache-max-age` (a mismatch is rejected).

## Fitness and ordering

Fitness is a minimization vector. Single-objective GP commonly uses `SingleObjectiveComparison`; NSGA-II computes `Rank` and `Distance` and uses crowded comparison. `ParetoComparison` assumes minimization in every dimension. A custom comparator must be consistent for selection and reinsertion or the search transition no longer has a coherent ordering.

The `operon_nsgp --pareto-front` export contains the non-dominated members of the retained parent population, computed from their fitness vectors with the same sorter and epsilon deduplication as the NSGA-II sort, applied to the retained parents (the sorter chosen with `--sorter`; the export is not `NSGA2::Best()`, which after a completed generation is the front of parents plus offspring), including the `--epsilon` tolerance: objective differences up to `--epsilon` do not establish dominance, and epsilon-equivalent individuals are listed once (the lexicographically smallest). Stored `Rank` values are not used because they may refer to a previously sorted population (typically parents plus offspring after a completed generation, or the initial parents when a budget ends generation 0) rather than to the retained parents. Unfilled offspring from an evaluation-limited generation are not front members.

## Coefficient optimization

`CoefficientOptimizer` delegates to an `OptimizerBase`, usually `LevenbergMarquardtOptimizer`. It changes only `Optimize` constants in the tree. `FitOutcome` distinguishes an improving fit, a valid non-improving fit, an interpreter evaluation error, and a configuration error; `Diagnostics(outcome)` is available in every case. Treat an unsuccessful local fit as a candidate-level result, not as a reason to discard the structural tree before normal evaluation.
