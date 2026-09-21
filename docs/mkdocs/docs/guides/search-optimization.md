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

## Fitness and ordering

Fitness is a minimization vector. Single-objective GP commonly uses `SingleObjectiveComparison`; NSGA-II computes `Rank` and `Distance` and uses crowded comparison. `ParetoComparison` assumes minimization in every dimension. A custom comparator must be consistent for selection and reinsertion or the search transition no longer has a coherent ordering.

`FeasibilityFirstComparison` can order valid trees before invalid ones independently of their score. It is complementary to shape evaluator worst-value substitution; choose it when feasibility should remain explicit instead of being encoded in a penalty.

## Coefficient optimization

`CoefficientOptimizer` delegates to an `OptimizerBase`, usually `LevenbergMarquardtOptimizer`. It changes only `Optimize` constants in the tree. `FitOutcome` distinguishes an improving fit, a valid non-improving fit, an interpreter evaluation error, and a configuration error; `Diagnostics(outcome)` is available in every case. Treat an unsuccessful local fit as a candidate-level result, not as a reason to discard the structural tree before normal evaluation.
