# Shape constraints

Shape constraints certify properties of a candidate expression over a declared input-domain box. They can constrain the expression value, a first partial derivative, or a second partial derivative. A candidate is feasible only when every requested property can be certified over the whole box.

![Shape constraint certification](../assets/diagrams/shape-constraints.svg){ .diagram }

## Configuration

Pass a JSON configuration with `--shape-constraints-config`:

```json
{
  "domains": { "theta": [1.0, 3.0] },
  "constraints": [
    { "op": "derivative", "variable": "theta", "order": 1, "sign": -1 }
  ]
}
```

`domains` maps every referenced input to its closed interval. Each constraint selects one quantity:

- `{ "op": "id", ... }` constrains the model output.
- `{ "op": "derivative", "variable": "x", "order": 1, ... }` constrains the first partial derivative with respect to `x`.
- `order: 2` constrains the corresponding unmixed second partial derivative.

Every constraint must use exactly one of:

- `"sign": 1`: output or derivative is non-negative everywhere. For a first derivative, this means non-decreasing in that variable.
- `"sign": -1`: output or derivative is non-positive everywhere. For a first derivative, this means non-increasing in that variable.
- `"bound": [lo, hi]`: the selected quantity stays in the inclusive interval.

Mixed partial derivatives are not supported.

## Canonical example: Feynman1

The [Florian Bachinger JDIQ data-validation repository](https://github.com/florianBachinger/SC-based-Data-Validation-JDIQ) defines Feynman1 on `theta ∈ [1, 3]` as `f(theta) = exp(-theta² / 2) / sqrt(2π)`.

On that domain it is decreasing. The upstream repository also specifies a non-negative second derivative:

```json
{ "op": "derivative", "variable": "theta", "order": 2, "sign": 1 }
```

That condition is mathematically true, but it reaches zero at `theta = 1`. Finite-precision interval enclosures can conservatively include a small negative value at that boundary and decline to certify the exact expression. Use the first-derivative condition as the reliable minimal example; add the curvature condition only after checking the final `shape-constraints:` status for the selected bound mode and domain. Its value lies in `[0.0044318484, 0.2419707245]`; add an `id` bound only when that property is part of the intended model contract.

## Enforcement

With a configuration, the default is `hard-reject`:

- **Hard reject** assigns `--shape-worst-value` to every objective of an infeasible candidate. It is the strictest option and changes the search trajectory.
- **Penalty** adds a weighted violation objective to the error objective.
- **Extra objective** adds shape violation as a separate NSGA-II objective.

Use `--shape-enforcement` to select one mode supported by the chosen algorithm: `hard-reject`, `penalty`, `extra-objective`, or `penalty,extra-objective` (also accepted as `extra-objective,penalty`); the modes are mutually exclusive alternatives, not combinable flags. GP supports `hard-reject` and `penalty`; its default is `hard-reject`. NSGA-II supports all four; its default is `hard-reject`.

NSGA-II objectives are `[error, tree length]`, plus a trailing shape-violation objective for `extra-objective` and `penalty,extra-objective`. `penalty` folds the weighted violation into the error, and `extra-objective` appends the measured violation as the last objective (`penalty,extra-objective` does both). Under `hard-reject` an infeasible candidate receives `--shape-worst-value` on every objective.

## Evaluation cost

Each candidate evaluation performs exactly one prediction pass for every enforcement mode: the evaluator fits the linear scaling during that pass, and the shape certificate is computed under that same transform before the error is scored (`ShapeConstrainedValueEvaluator`). Shape constraints therefore apply to the coefficients presented at the evaluation call.

Fitness describes the evaluated model: under non-Lamarckian local search the shape verdict applies to the optimized coefficients that were scored, and the restored genotype is not re-certified. To certify a model for export or reporting, measure it explicitly with `ShapeCertifier::MeasureCanonical`. With a shape-constrained evaluator the structural transposition cache (`--transposition-cache`) is bypassed, because a structural hit carries no coefficient identity to certify.

## Bound backends

`--shape-bound-mode interval` is the default. It uses interval arithmetic to certify the requested bounds over the domain box. The other modes are:

- `combined`: affine arithmetic with an interval fallback;
- `affine`: affine arithmetic only;
- `bisected` or `bisected:N`: interval arithmetic over subdivided boxes. Bare `bisected` uses the separate `--shape-bisection-depth` option (default `3`), while `bisected:N` carries an explicit depth `N` (0–20). An empty suffix (`bisected:`) is invalid. The depth is spent on the variables that occur more than once in the expression (interval arithmetic encloses a variable that occurs once exactly), preferring wide domains and frequent repeats: each level halves the axis with the largest `width * (occurrences - 1)^2`, and plain widest-first applies when no variable repeats. Ties go to the smaller variable hash, so the result does not depend on the tree's layout.
- `jet` or `jet:N`: the `interval` mode's forward-mode jet bounds for first- and second-derivative constraints, evaluated on each of the `2^N` sub-boxes of the same subdivision as `bisected:N` and unioned. Depth selection follows `bisected` (bare `jet` uses `--shape-bisection-depth`, `jet:N` carries `N` in 0–20), and `jet:0` is identical to `interval`. Only the input intervals are split; the derivative is not differentiated symbolically. If any sub-box of a derivative constraint has no finite certain enclosure, that constraint falls back to the whole-box result of `interval` mode, so the bound is never wider than `interval`'s. Identity constraints are bisected exactly as in `bisected:N`. The sub-boxes are evaluated one at a time (scalar), so the cost grows with `2^N`.

In `interval` mode a second-derivative constraint is bounded by a single forward-mode pass over the model that propagates the value, the first derivative and the pure second derivative (`Operon::Jet<T, K, Operon::JetOrder::SecondDiagonal>` over interval scalars, evaluated by `Operon::IntervalSecondOrderJetEvaluator`), instead of differentiating the symbolic first-derivative tree a second time. This stores directional pure curvature only: it does not represent or independently expose mixed partials. With the one-hot seeds used by shape certification, direction `k` is the selected variable and `w[k]` is that variable's unmixed second partial. Up to four distinct constraint variables are handled per pass; constraints that repeat a variable share one direction. Non-smooth operations are treated as opaque and report an uncertain result, so interval evaluation across a kink never certifies curvature. An operation without a differentiation rule falls back to the symbolic path.

A successful run means that the returned model was scored under the selected enforcement policy. Check the final `shape-constraints:` status line when a certificate is required: a model can be printed after an infeasible hard-reject search if no feasible individual was found.

## Choose a compatible domain

The constraints must be true of the behavior you intend to learn on the whole declared box. They are not pointwise checks on the training rows.

For example, Pagie-1 is U-shaped in each signed input over `[-5, 5]`. A non-decreasing derivative constraint in `X` or `Y` over that full box is incompatible with the target and should not be used to compare constrained and unconstrained model accuracy. Restrict the domain to a region with the intended shape, transform the input, or choose a compatible problem such as Feynman1.

## Reproducible Feynman1 invocation

Prepare a CSV with `theta` and `F` columns sampled from the formula above, save the JSON configuration as `feynman1-shape.json`, then run:

```sh
./operon_nsgp \
  --dataset feynman1.csv --train 0:2000 --target F \
  --population-size 500 --generations 150 --seed 42 --threads 1 \
  --enable-symbols exp \
  --shape-constraints-config feynman1-shape.json
```

For performance measurements, invoke the built executable directly after it has been built. Do not include `nix develop` activation in the timed region.
