# Core model

![Core model relationships](../assets/diagrams/core-model.svg){ .diagram }

## Core relationships

`Dataset` owns named columns. `Problem` selects its target and training/test ranges, while `PrimitiveSet` defines the expression vocabulary available to tree creators and variation operators. An `Individual` pairs a tree genotype with its evaluated fitness.

This separation lets the same tree representation be scored by ordinary regression, a custom evaluator, or a shape-constraint wrapper without changing the search operators.
