# Architecture integration

Operon keeps the model representation, expression execution, evaluation, and search operators separate. Integrate at the boundary appropriate to the task:

- construct a `Dataset` and `Problem` for data and ranges;
- select a `PrimitiveSet`, creator, selectors, and variation operators;
- provide an `EvaluatorBase` implementation when the objective is custom;
- use the CLI programs when the standard symbolic-regression flow is enough.

These guides follow those boundaries so custom integrations can reuse the standard tree and evaluator contracts without duplicating the search loop.
