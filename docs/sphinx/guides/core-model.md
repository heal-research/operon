# Core model

```{mermaid}
flowchart TB
    problem[Problem] -->|owns or borrows; selects ranges| dataset[Dataset]
    problem -->|owns generation vocabulary| pset[PrimitiveSet]
    creator[Creators / variation] -->|uses| pset
    creator -->|constructs| individual[Individual: Tree genotype + fitness]
    evaluator[Evaluator] -->|uses| problem
    evaluator -->|scores| individual
    evaluator -->|uses| interpreter[Interpreter]
    interpreter -->|reads| dataset
    interpreter -->|executes| individual
```

```{doxygenpage} core_model
:project: operon
```
