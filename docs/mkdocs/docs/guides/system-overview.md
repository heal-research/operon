# System overview

![Operon subsystems](../assets/diagrams/system-overview.svg){ .diagram }

## Subsystems and ownership

| Layer | Owns | Depends on |
| --- | --- | --- |
| Core | datasets, problem definition, postfix trees, primitive vocabulary | no search state |
| Execution | dispatch binding, sampled values, coefficient Jacobians, interval/affine bounds | `Tree`, `Dataset`, dispatch table |
| Evaluation | objective values, evaluation budget, objective count | `Problem`, interpreter or another evaluator |
| Search | population, initialization, offspring, reinsertion, stopping | evaluator and operators |
| Optimization | numeric coefficients only | tree, problem, dispatch table |

The CLI programs assemble these objects; they are not a separate execution path. Library integrations can use the same composition directly.

## Lifetime and concurrency

`Problem` either owns its `Dataset` or borrows one; every component receiving a `Problem const*` therefore requires the problem—and its dataset—to outlive it. Evaluators and algorithms are held by pointer and are not value-like configuration objects.

An `Interpreter` lazily binds a tree and range into mutable scratch state, so one interpreter instance is required per concurrent worker. The search APIs accept a Taskflow executor for the same reason: parallel work is scheduled by the algorithm while worker-local execution state stays isolated.

## Stable boundary

`Individual` carries a `Tree`, fitness vector, rank, and crowding distance. Search sees the fitness vector and comparison; execution sees the tree and coefficients. Keep this boundary intact when extending Operon—attaching search-specific state to the tree makes reuse by enumeration, shape certification, and formatting fragile.
