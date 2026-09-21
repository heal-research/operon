# Core model

![Core model relationships](../assets/diagrams/core-model.svg){ .diagram }

## Model objects

| Type | Role | Key contract |
| --- | --- | --- |
| `Dataset` | named numeric columns and optional row weights | owning datasets pad columns for SIMD; `Wrap` is zero-copy and requires caller-owned, padded column-major storage |
| `Problem` | target, input hashes, data splits, primitive set, linear-scaling policy | it borrows or owns the dataset; changing the target resets default inputs |
| `PrimitiveSet` | enabled terminals/functions, frequencies, arity bounds | configure before concurrent creation; mutation during cache reads is unsupported |
| `Tree` | postfix node sequence and derived metadata | call `UpdateNodes()` after structural edits; `Validate()` checks representation invariants without evaluating |
| `Individual` | genotype plus minimization objectives | `Rank` and `Distance` are NSGA-II state, not intrinsic tree properties |

## Tree representation

Operon stores a complete expression in postfix order. A function node consumes contiguous completed child subtrees before it; the final node is the root. Each node caches structural metadata—length, depth, parent, level, and hash—derived from that order. `Tree::UpdateNodes()` refreshes it after a structural change. `Tree::Validate()` reports the first violated invariant, including malformed function arity, non-backward references, multiple roots, and stale metadata.

Constants with `Optimize == true` are the coefficient vector used by optimizers and interpreters. `GetCoefficients()` reads them in tree order; `SetCoefficients()` requires the matching order and count. A caller that changes only coefficients does not need `UpdateNodes()`.

## Problem formulation

Construct a `Problem` after the dataset, set the target and ranges, then select inputs and primitives. `SetDefaultInputs()` selects every column other than the target; `SetInputs()` instead fixes an explicit hash/name set. Training, test, and validation ranges are independent. Evaluation normally uses the training range; consumers must select another range explicitly when reporting holdout quality.

`Problem` also owns the `PrimitiveSet` and the linear-scaling policy, because both alter the model formulation rather than the mechanics of one evaluator.
