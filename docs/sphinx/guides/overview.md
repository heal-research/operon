# Architecture overview

An Operon run evaluates a population of symbolic-expression trees against a
dataset, then uses the resulting objective values to select, vary, and retain
candidate models. The public {doc}`Operon::Node <../reference/node>` reference
is the compact value type used to represent each element of those trees.

:::{md-mermaid}
flowchart TB
    D[Dataset] --> E[Evaluator]
    P[Population of trees] --> E
    E --> F[Objective values]
    F --> S[Selection and variation]
    S --> P
:::

The diagram is inline Mermaid markup. It remains responsive within the page and
uses the active site palette rather than an embedded Doxygen HTML frame.

## Documentation build

The documentation-only CMake configuration first produces Doxygen XML from
`include/operon`, then passes that XML to Sphinx through Breathe. The
{doc}`C++ API reference <../reference/index>` therefore stays tied to the
comments maintained next to the public C++ declarations.
