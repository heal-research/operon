# API reference

This reference is organized around public integration boundaries instead of mirroring every header. It identifies the types an application constructs, the methods that define their contracts, and the headers that contain the complete declarations.

## Start here

| Area | Use it for | Primary headers |
| --- | --- | --- |
| [Core model](core.md) | datasets, problem definition, trees, primitives, fitness containers | `operon/core/{dataset,problem,tree,pset,individual}.hpp` |
| [Execution and evaluation](execution.md) | dispatch, interpretation, objectives, constraints, coefficient fitting | `operon/{core/dispatch,interpreter/interpreter,operators/evaluator,operators/shape_constrained_evaluator,optimizer/optimizer}.hpp` |
| [Search and enumeration](search.md) | GP, NSGA-II, population operators, grammar enumeration | `operon/algorithms/{gp,nsga2,enumeration}.hpp` |

## API conventions

- Public APIs use `Operon::Scalar`, `Vector`, and `Span` aliases from `operon/core/types.hpp`.
- Fitness and standard error metrics are **minimized**. Multi-objective comparisons assume the same direction for every component unless a custom comparator says otherwise.
- `gsl::not_null<T*>` constructor arguments are required non-null borrowed dependencies. Their owner must outlive the receiving object.
- Fallible non-throwing interfaces return `tl::expected<T, E>`; corresponding convenience overloads may throw formatted errors.
- `Tree` topology and coefficients have distinct update paths: structural edits require `UpdateNodes()`, while `SetCoefficients()` changes only values.

> The installed headers are authoritative for overloads and templates. This reference documents semantic contracts and failure modes that are easy to miss from a signature alone.
