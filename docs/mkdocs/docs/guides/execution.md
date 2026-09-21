# Execution

![Expression execution](../assets/diagrams/execution.svg){ .diagram }

## Expression execution

The interpreter executes a tree against a selected dataset range through a dispatch table. The same expression structure also supports derivative evaluation for coefficient fitting and interval or affine evaluation for whole-domain bounds.

Keep sampled execution and whole-domain certification distinct: sampled values measure fit on rows, while interval and affine bounds certify properties between those rows.
