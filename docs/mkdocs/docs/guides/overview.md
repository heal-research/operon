# Architecture overview

Operon separates problem formulation, symbolic programs, execution, objectives, and search. A tree is the only model exchanged between those layers; a `Problem` carries the data-dependent context.

![Operon search architecture](../assets/diagrams/overview.svg){ .diagram }

## Data and control flow

1. `Dataset` stores column-major observations; `Problem` selects a target, inputs, row ranges, and a `PrimitiveSet`.
2. Initializers and variation operators create postfix `Tree` genotypes using that primitive vocabulary.
3. An `EvaluatorBase` optionally interprets a tree over the training range, then emits a minimization fitness vector for an `Individual`.
4. Selection, offspring generation, reinsertion, and either GP or NSGA-II own the population transition. A coefficient optimizer may improve only the tree's optimizable constants before it is scored.

This split is deliberate: changing a loss, an interpreter backend, or a search operator does not require a second expression format or a second data model.

## Choose an integration boundary

| Need | Extension point |
| --- | --- |
| Different columns, ranges, target, or vocabulary | `Dataset`, `Problem`, `PrimitiveSet` |
| Custom scalar function | `ScalarDispatch` registration plus `PrimitiveSet::AddFunction` |
| Different objective or multiple objectives | `EvaluatorBase::Score` / composite evaluator |
| Different local coefficient fit | `OptimizerBase` |
| Different population behavior | creator, selector, mutation, crossover, reinserter, or algorithm |

The [API reference](../reference/index.md) describes these contracts. The guides explain when to use them and which invariants the interfaces do not enforce for callers.
