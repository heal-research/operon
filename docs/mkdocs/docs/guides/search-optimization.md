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

`GeneticAlgorithmConfig` bounds generations, evaluations, iterations, population size, pool size, and seed. Its evaluation budget is consumed by evaluator work—not by offspring attempts—so operators should not embed untracked objective calculations.

A generation counts only if its offspring pool was completely filled. If a stop condition (evaluation budget, generator termination such as `OffspringGeneratorBase::Terminate()` and the offspring-selection pressure limit, time limit, stop request) fires while the pool is being filled, both drivers discard that generation: nothing is reinserted, `Generation()` is not incremented and the report callback is not invoked, so `Parents()` holds the last completed generation. `IncompleteGeneration()` returns the attempted/completed/pool counts of such a cut-short generation, or `nullopt` if the last generation completed.

An incomplete generation is not itself a stop reason and does not set `StopRequested()`: the run ends only because the stop condition that cut the generation short (evaluation budget, generator termination, time limit, stop request or a `true` report) holds. A generator-defined termination that clears again before the loop checks it, such as the selection-pressure limit, still ends the run. `StopRequested()` is true only after `RequestStop()` or a `true` report callback.

## Fitness and ordering

Fitness is a minimization vector. Single-objective GP commonly uses `SingleObjectiveComparison`; NSGA-II computes `Rank` and `Distance` and uses crowded comparison. `ParetoComparison` assumes minimization in every dimension. A custom comparator must be consistent for selection and reinsertion or the search transition no longer has a coherent ordering.

The `operon_nsgp --pareto-front` export is computed from retained parents, not `NSGA2::Best()` (which, after a completed generation, is the front of parents plus offspring). Fitness vectors are sorted lexicographically and adjacent epsilon-equivalent individuals are collapsed onto the first representative. The CLI sorters (`rs` and `ms`) then rank the representatives by exact dominance; `--epsilon` affects grouping, not their dominance comparisons. Stored `Rank` values are not used because they may describe the previously sorted parents and offspring rather than the retained parents. Unfilled offspring are not front members.

## Coefficient optimization

`CoefficientOptimizer` delegates to an `OptimizerBase`, usually `LevenbergMarquardtOptimizer`. It changes only `Optimize` constants in the tree. `FitOutcome` distinguishes an improving fit, a valid non-improving fit, an interpreter evaluation error, and a configuration error; `Diagnostics(outcome)` is available in every case. Treat an unsuccessful local fit as a candidate-level result, not as a reason to discard the structural tree before normal evaluation.
