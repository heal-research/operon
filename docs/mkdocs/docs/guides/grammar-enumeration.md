# Grammar enumeration

![Grammar enumeration flow](../assets/diagrams/grammar-enumeration.svg){ .diagram }

## What is enumerated

`EnumerationEngine` performs bottom-up dynamic programming over an `Operon::Grammar`. For each nonterminal and symbolic-complexity budget it constructs candidate postfix trees, then calls `Reduce()`, `Simplify()`, and content-hash deduplication before storing a bucket. Complexity counts every non-constant node; free constants are fitted parameters, not structural complexity.

The engine visits budgets in increasing order. `SetOnNovelExpression()` receives only newly inserted complete `Expression` trees, not grammar intermediates. Use that hook for work that must happen exactly once per unique candidate.

## Top-level algorithm

`GrammarEnumerationAlgorithm` connects the engine to an `OptimizerBase` and an `EvaluatorBase`. It fits coefficients for each novel expression, scores it with the evaluator, and keeps the ascending-fitness `TopK` entries returned by `BestTrees()`. This separates fitting loss from ranking objective: a least-squares optimizer can be paired with a different evaluator metric.

| Configuration | Effect |
| --- | --- |
| `EnumerationConfig::MaxComplexity` | maximum caller-visible structural complexity |
| `EnumerationConfig::TopK` | number of best scored trees retained |
| `Run(rng, report)` | single-shot build, fit, score, and retain |
| `RequestStop()` or `report == true` | stops after the current completed complexity level |

`Run()` is intentionally single-shot: buckets and deduplication state are retained, and the engine is not resettable. Construct a new algorithm for another enumeration. Enumeration can grow combinatorially; use it when the grammar and complexity limit are tractable, not as a drop-in replacement for a large population search.
