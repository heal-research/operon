# Core model API

## `Dataset`

Header: [`operon/core/dataset.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/core/dataset.hpp)

```cpp
Dataset(std::string const& csvPath, bool hasHeader = false);
Dataset(std::vector<std::string> const& names,
        std::vector<std::vector<Scalar>> const& columns);
static Dataset Wrap(gsl::not_null<Scalar const*> data, int rows, int cols);
```

`Dataset` stores columns, not rows. Owning constructors allocate padded column-major storage; `Rows()` reports logical rows and `PaddedRows()` reports the backing extent. `Wrap` has no copy: the caller MUST provide `((rows + 7) & ~7) * cols` values of `Operon::Scalar` in column-major order, zero the tail rows, and retain the storage for the dataset lifetime.

| Method | Contract |
| --- | --- |
| `GetValues(name/hash/index)` | logical-length, read-only column span |
| `GetPaddedValues(...)` | SIMD-facing pointer; valid only under the padding contract |
| `VariableNames`, `VariableHashes`, `GetVariable` | value-returning metadata lookup |
| `FindVariableName(hash)` | zero-copy name view; invalidated by variable-identity mutation |
| `SetWeights(span)` / `Weights()` | optional per-row weights |
| `Normalize`, `Standardize`, `Shuffle`, `PermuteRows`, `SetValues` | mutate an owning dataset |

## `Problem`

Header: [`operon/core/problem.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/core/problem.hpp)

```cpp
Problem(std::unique_ptr<Dataset>);            // owns data
Problem(gsl::not_null<Dataset*>);             // borrows data
problem.SetTarget("Y");
problem.SetTrainingRange({0, 250});
problem.SetTestRange({250, 500});
problem.SetInputs(inputHashes);
problem.ConfigurePrimitiveSet(PrimitiveSet::Full);
```

A problem contains a target, input-variable hashes, train/test/validation `Range`s, primitive vocabulary, and linear-scaling flags. The borrowed-dataset constructor transfers no ownership. `SetTarget` updates default inputs; `SetInputs` disables that default until `SetDefaultInputs()` is called. `TargetValues(range)` and `Weights(range)` are range-local views.

`SetLinearScalingOmitsNonFinite()` is caller-coordinated with evaluator-level non-finite handling. Mismatching these policies can fit different scales during scoring and certification.

## `Tree` and `Individual`

Headers: [`tree.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/core/tree.hpp), [`individual.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/core/individual.hpp)

```cpp
Tree tree;
tree.UpdateNodes();
auto valid = tree.Validate(); // expected<void, TreeValidationError>
tree.SetCoefficients(coefficients);
```

A `Tree` is a postfix sequence. A valid non-empty tree has exactly one final root; function children are contiguous completed subtrees; references point backward. `Validate()` checks these properties and cached metadata without running an evaluator. `Reduce()`, `Simplify()`, `Sort()`, and `Splice()` transform expressions; structural direct edits require `UpdateNodes()` before execution.

`Individual` holds `Genotype`, a minimization `Fitness` vector, `Rank`, and `Distance`. Construct it with the number of objectives. Use `SingleObjectiveComparison`, `LexicographicalComparison`, `ParetoComparison`, or `CrowdedComparison` according to the algorithm; Pareto and crowded ordering assume minimization.

## `PrimitiveSet`

Header: [`operon/core/pset.hpp`](https://github.com/heal-research/operon/blob/main/include/operon/core/pset.hpp)

```cpp
PrimitiveSet pset { PrimitiveSet::Arithmetic };
pset.AddFunction(customHash, /*arity=*/2, /*frequency=*/1);
pset.SetEnabled(customHash, true);
auto achievable = pset.AchievableLength(targetLength);
```

`Arithmetic`, `TypeCoherent`, and `Full` are presets. A primitive has a node, sampling frequency, and allowed arity range. `AddFunction` only makes a symbol generatable: the matching callable MUST already be registered in the dispatch table. `ReachableLengths()` and `AchievableLength()` cache structural feasibility. Configure the primitive set before concurrent tree creation; cache reads concurrent with mutation are unsupported.
