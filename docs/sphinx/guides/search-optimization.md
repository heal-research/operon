# Search and optimization

```{mermaid}
flowchart LR
    start([ ]) --> prepare[Prepare selectors and evaluator] --> variation[Select / crossover / mutate] --> fit{Fit coefficients?}
    fit -->|yes| optimize[Coefficient fit] --> score[Score individual]
    fit -->|no| score
    score --> pool[Offspring pool] --> reinsert[Reinsert: elites protected] --> next[Next parents] --> finish([ ])
```

```{doxygenpage} search_optimization
:project: operon
```
