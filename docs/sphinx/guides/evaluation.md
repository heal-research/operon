# Evaluation guide

```{mermaid}
flowchart LR
    start([ ]) --> evaluate[Evaluate: sampled execution result] --> error{Execution setup error?}
    error -->|yes| maximum[ErrMax] --> finish([ ])
    error -->|no| proof[Optional EvaluatedBuffer] --> score[Score: objectives → fitness] --> finish
```

```{doxygenpage} evaluation
:project: operon
```
