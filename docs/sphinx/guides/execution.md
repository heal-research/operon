# Execution guide

:::{md-mermaid}
flowchart TB
    interpreter[Interpreter] -->|executes| tree[Tree]
    interpreter -->|reads| dataset[Dataset rows]
    interpreter -->|resolves| dispatch[Dispatch table]
    derivatives[Jacobians] -->|traces| interpreter
    bounds[Interval / affine bounds] -->|analyses| tree
    bounds -->|interval callbacks| dispatch
:::

```{doxygenpage} execution
:project: operon
```
