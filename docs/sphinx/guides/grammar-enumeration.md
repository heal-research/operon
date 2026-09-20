# Grammar enumeration

```{mermaid}
flowchart TB
    start([ ]) --> grammar[Grammar and input variables] --> build[Build DP buckets by complexity] --> novel{Novel complete expression?}
    novel -->|yes| fit[Fit coefficients] --> score[Score with evaluator] --> topk[Keep best TopK] --> next[Next budget]
    novel -->|duplicate| next
    next -->|within limit| build
    next -->|done| finish([ ])
```

```{doxygenpage} grammar_enumeration
:project: operon
```
