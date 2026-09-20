# System overview

```{mermaid}
flowchart TB
    entry[Entry points: operon_gp, operon_nsgp, operon_enum] -->|configures| search[Search: population, variation, selection, reinsertion, cache, Taskflow]
    search -->|fitness of a candidate| evaluation[Evaluation: Evaluate → Score → fitness]
    evaluation -->|values / bounds| execution[Execution: interpreter / JIT, interval / affine]
    execution --> model[Model: Problem, Dataset, PrimitiveSet, Individual, Tree]
    search -. optional local search .-> fitting[Coefficient fitting: LM, L-BFGS, SGD]
    fitting -->|residuals / Jacobians| execution
```

```{doxygenpage} architecture
:project: operon
```
