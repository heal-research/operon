# Evaluation

![Evaluation pipeline](../assets/diagrams/evaluation.svg){ .diagram }

## Evaluate, then score

Evaluators have two phases. `Evaluate` produces sampled model values when an objective needs them; `Score` turns those values into one or more objective values. `EvaluatedBuffer` records that a scratch buffer belongs to the current individual, preventing a scorer from accidentally consuming values from a different tree.

Execution failures become the evaluator's maximum error rather than entering selection as an ordinary fitness value.
