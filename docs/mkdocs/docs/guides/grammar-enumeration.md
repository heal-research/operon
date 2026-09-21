# Grammar enumeration

![Grammar enumeration flow](../assets/diagrams/grammar-enumeration.svg){ .diagram }

## Enumeration flow

Grammar enumeration builds expressions by increasing structural complexity, deduplicates complete candidates, fits coefficients when configured, and keeps the best candidates within its budget. It is a deterministic alternative to population search when the grammar and complexity limit are tractable.

The evaluator remains the common boundary: enumerated and evolved trees use the same objective and model representation.
