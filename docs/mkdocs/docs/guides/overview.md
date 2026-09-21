# Architecture overview

An Operon run evaluates a population of symbolic-expression trees against a dataset, then uses objective values to select, vary, and retain candidate models.

![Operon search architecture](../assets/diagrams/overview.svg){ .diagram }

## Documentation boundaries

The model, execution, evaluation, and search layers remain separate. The public headers define the C++ interface; generated API material is produced from the compilation database rather than being coupled to the site renderer.
