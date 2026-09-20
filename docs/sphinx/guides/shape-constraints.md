# Shape constraints

```{mermaid}
flowchart LR
    start([ ]) --> evaluate[Wrapped Evaluate: one sampled pass] --> error{Execution setup error?}
    error -->|yes: ErrMax| finish([ ])
    error -->|no| certify[Certify bounds: fit scaling from values if needed] --> feasible{Feasible?}
    feasible -->|yes| score[Wrapped Score] --> finish
    feasible -->|no| reject[WorstValue; violations++] --> finish
```

```{doxygenpage} shape_constraints
:project: operon
```
