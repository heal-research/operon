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

`GeneticAlgorithmConfig` supplies `Generations`, `Evaluations`, `Iterations`, `PopulationSize`, `PoolSize`, and `Seed`. Population algorithms consume evaluator work under the evaluation budget. A warm start retains the existing population; otherwise initialization happens before the first transition.

## Operator assembly

Headers: `operon/operators/{creator,initializer,generator,selector,crossover,mutation,reinserter,local_search}.hpp`

| Role | Typical implementation | Responsibility |
| --- | --- | --- |
| creator | `BalancedTreeCreator` | create a valid topology for inputs and primitive set |
| tree initializer | `UniformTreeInitializer` | choose initial lengths/depths and invoke creator |
| coefficient initializer | `NormalCoefficientInitializer` | initialize `Optimize` constants |
| selectors | `TournamentSelector` | choose parents under a comparison |
| variation | `SubtreeCrossover`, `MultiMutation` | produce valid bounded offspring |
| local search | `CoefficientOptimizer` | apply an `OptimizerBase` result to coefficients |
| offspring generator | `BasicOffspringGenerator` | compose selection, variation, local search, evaluation |
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
