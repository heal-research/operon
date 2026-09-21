# Architecture integration

Operon keeps the model representation, expression execution, evaluation, and search operators separate. Integrate at the narrowest boundary that expresses the change; reusing the existing tree and evaluator contracts keeps custom work compatible with GP, NSGA-II, enumeration, formatters, and shape certification.

## Integration recipes

### Run the standard loop with a custom function

1. Build `Dataset` and `Problem`, set target/ranges/inputs, and configure the primitive set.
2. Register the function and derivative in `ScalarDispatch`; add the same function hash and arity to `PrimitiveSet`.
3. Reuse the standard creator, evaluator, optimizer, and population algorithm.

### Add an objective

Implement `EvaluatorBase::Score` or compose an existing evaluator. Return a fitness vector whose length agrees with `ObjectiveCount()`. If the objective needs interpreted values, implement `Evaluate` and consume only the associated `EvaluatedBuffer` in `Score`. Select a multi-objective algorithm and a comparison that interprets every objective as minimization.

### Add a whole-domain constraint

Describe domains and constraints in a `ShapeConstraintSet`, then wrap the ordinary evaluator in `ShapeConstrainedEvaluator` or add `ShapeViolationEvaluator` as a distinct objective. Bounds certify only what their enclosure proves; configure enforcement and unknown-violation treatment explicitly.

### Replace one search operator

Operate on a valid `Tree` and return it through the existing initializer, mutation, crossover, selector, generator, or reinserter interfaces. After changing topology, refresh derived tree metadata. Do not reimplement sampled evaluation or attach operator state to `Tree`.

## Before integrating

- Ensure the dataset and problem outlive all borrowed components.
- Keep a dispatch table and primitive set synchronized.
- Allocate one interpreter per worker thread.
- Use only `Tree::SetCoefficients()` for numeric changes; call `UpdateNodes()` after structural changes.
- Check evaluator objective count against the `Individual` fitness width and chosen comparison.

See the [API reference](../reference/index.md) for signatures and failure contracts, and the [custom primitives example](https://github.com/heal-research/operon/blob/main/example/custom_primitives.cpp) for a complete executable assembly.
