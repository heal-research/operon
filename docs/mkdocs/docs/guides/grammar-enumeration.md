# Grammar enumeration

![Grammar enumeration flow](../assets/diagrams/grammar-enumeration.svg){ .diagram }

## What is enumerated

`EnumerationEngine` performs bottom-up dynamic programming over an `Operon::Grammar`. For each nonterminal and symbolic-complexity budget it constructs candidate postfix trees, then calls `Reduce()`, `Simplify()`, and content-hash deduplication before storing a bucket. Complexity counts every non-constant node; free constants are fitted parameters, not structural complexity.

The engine visits budgets in increasing order and has no per-candidate callback hook: `EnumerationEngine::Build()` only discovers and deduplicates distinct trees per `(nonterminal, budget)`. Work that must happen exactly once per unique candidate (fitting, scoring, ranking) happens afterward, in `GrammarEnumerationAlgorithm::Run`'s group/fit/rank pass below - not during `Build()`.

`Operon::Grammar` is configured from either a `PrimitiveSetConfig` (a narrow, behavior-preserving-for-legacy-callers compatibility shim - see `Grammar::Configure(PrimitiveSetConfig)`'s doc comment) or, for the full ESR function vocabulary, an `EnumerationFunctionSet` built from a named preset (`ParseEnumerationPreset`/`PresetFunctions` - `keep_duplicates`, `core_maths`, `ext_maths`, `osc_maths`, `base10_maths`, `base_e_maths`) or assembled by hand (`custom`).

## Canonical grouping

Distinct-looking candidate trees can represent the same algebraic family (e.g. `x+y` and `y+x`, or `Square(x)` and `x*x`). `CanonicalizeEnumerationTree` (`algorithms/enumeration_canonicalizer.hpp`) computes a deterministic sum-of-monomials `Key` for each tree, sound (never merges two genuinely different families) but not necessarily complete (a bounded distribution cap can leave some equivalent expressions with distinct keys). `GrammarEnumerationAlgorithm::Run` groups every stored candidate by this `Key` and fits/scores exactly one representative per class - see its own doc comment for the exact 3-level representative tie-break.

## Top-level algorithm

`GrammarEnumerationAlgorithm` connects the engine to an `OptimizerBase` and an `EnumerationScorer` (built via `MakeMdlScorer` for the ESR-parity `EnumerationRanking::MinimumDescriptionLength` ranking, or `MakeObjectiveScorer` for the pre-existing evaluator-based `EnumerationRanking::Objective` ranking). It fits coefficients for each canonical-class representative, scores it, and keeps the ascending-`Score` `TopK` entries returned by `BestTrees()` as `EnumerationResult`s. This separates fitting loss from ranking objective: a least-squares optimizer can be paired with a different scoring metric.

| Configuration | Effect |
| --- | --- |
| `EnumerationConfig::MaxComplexity` | maximum caller-visible structural complexity |
| `EnumerationConfig::TopK` | number of best scored trees retained |
| `EnumerationConfig::Ranking` | which ranking the CLI/caller intends (informational - the scorer passed to the constructor is what actually determines ranking behavior) |
| `EnumerationConfig::EvaluationBufferSize` | per-worker scratch buffer size `Run()` allocates for `scorer`; must be >= the scorer's training range size |
| `Run(rng, report)` | single-shot build, canonical-group, fit, score, and retain |
| `RequestStop()` or `report == true` | stops after the current completed build level or fitting batch |

`Run()` is intentionally single-shot: buckets and deduplication state are retained, and the engine is not resettable. Construct a new algorithm for another enumeration. Enumeration can grow combinatorially; use it when the grammar and complexity limit are tractable, not as a drop-in replacement for a large population search.
