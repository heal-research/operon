#!/usr/bin/env python3
"""Generate Hessian ground truth using JAX for Operon BuildHessianDag tests.

Usage (from repo root):
    cd ~/src/envs/python
    nix develop .#python312 --command bash -c \
        "uv run --python python3.12 --with jax \
         python3 ~/src/operon/tools/generate_hessian_ground_truth.py"

Writes: data/hessian_ground_truth.txt
"""

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

NUM_ROWS = 20
NUM_VARS = 5
SEED = 12345


def make_cases():
    """Return list of (operon_infix, jax_fn, coeff_values).

    jax_fn signature: (c: array[ncoeffs], x: array[nvars]) -> scalar
    coeff_values: list of floats in the same order as c[] indices.

    Coefficient values must be distinct so C++ can match ordering.
    Variable columns: X1=0, X2=1, X3=2, X4=3, X5=4.
    """
    cases = []

    def add(expr, fn, coeffs):
        cases.append((expr, fn, coeffs))

    # --- single constant (Hessian = 0 for linear) ---
    add("0.700000", lambda c, x: c[0], [0.7])

    # --- unary ops: d²f/dc² ---
    add("sin(1.200000)", lambda c, x: jnp.sin(c[0]), [1.2])
    add("cos(0.800000)", lambda c, x: jnp.cos(c[0]), [0.8])
    add("exp(0.500000)", lambda c, x: jnp.exp(c[0]), [0.5])
    add("log(1.700000)", lambda c, x: jnp.log(c[0]), [1.7])
    add("sqrt(2.300000)", lambda c, x: jnp.sqrt(c[0]), [2.3])
    add("square(1.100000)", lambda c, x: c[0] ** 2, [1.1])
    add("tan(0.300000)", lambda c, x: jnp.tan(c[0]), [0.3])
    add("tanh(0.900000)", lambda c, x: jnp.tanh(c[0]), [0.9])
    add("sinh(0.400000)", lambda c, x: jnp.sinh(c[0]), [0.4])
    add("cosh(0.600000)", lambda c, x: jnp.cosh(c[0]), [0.6])
    add("asin(0.350000)", lambda c, x: jnp.arcsin(c[0]), [0.35])
    add("acos(0.450000)", lambda c, x: jnp.arccos(c[0]), [0.45])
    add("atan(1.400000)", lambda c, x: jnp.arctan(c[0]), [1.4])
    add("cbrt(1.900000)", lambda c, x: jnp.cbrt(c[0]), [1.9])
    add("log1p(0.650000)", lambda c, x: jnp.log1p(c[0]), [0.65])
    add("logabs(1.300000)", lambda c, x: jnp.log(jnp.abs(c[0])), [1.3])

    # --- binary ops: two constants ---
    add("(1.500000 + 2.100000)", lambda c, x: c[0] + c[1], [1.5, 2.1])
    add("(1.500000 - 2.100000)", lambda c, x: c[0] - c[1], [1.5, 2.1])
    add("(1.500000 * 2.100000)", lambda c, x: c[0] * c[1], [1.5, 2.1])
    add("(1.500000 / 2.100000)", lambda c, x: c[0] / c[1], [1.5, 2.1])
    add("(1.800000 ^ 0.700000)", lambda c, x: c[0] ** c[1], [1.8, 0.7])

    # --- coefficient * variable (tests chain rule) ---
    add("(1.300000 * X1)",
        lambda c, x: c[0] * x[0], [1.3])
    add("sin((1.300000 * X1))",
        lambda c, x: jnp.sin(c[0] * x[0]), [1.3])
    add("cos((0.900000 * X2))",
        lambda c, x: jnp.cos(c[0] * x[1]), [0.9])
    add("exp((0.400000 * X1))",
        lambda c, x: jnp.exp(c[0] * x[0]), [0.4])
    add("log((1.700000 * X1))",
        lambda c, x: jnp.log(c[0] * x[0]), [1.7])
    add("sqrt((2.500000 * X1))",
        lambda c, x: jnp.sqrt(c[0] * x[0]), [2.5])
    add("square((0.800000 * X1))",
        lambda c, x: (c[0] * x[0]) ** 2, [0.8])
    add("tanh((1.100000 * X2))",
        lambda c, x: jnp.tanh(c[0] * x[1]), [1.1])

    # --- two coefficients + variable ---
    add("((1.300000 * X1) + 2.100000)",
        lambda c, x: c[0] * x[0] + c[1], [1.3, 2.1])
    add("(sin((1.300000 * X1)) + 2.700000)",
        lambda c, x: jnp.sin(c[0] * x[0]) + c[1], [1.3, 2.7])
    add("(sin((1.300000 * X1)) * 0.800000)",
        lambda c, x: jnp.sin(c[0] * x[0]) * c[1], [1.3, 0.8])
    add("((1.200000 * X1) + (0.800000 * X2))",
        lambda c, x: c[0] * x[0] + c[1] * x[1], [1.2, 0.8])
    add("((1.200000 * X1) * (0.800000 * X2))",
        lambda c, x: (c[0] * x[0]) * (c[1] * x[1]), [1.2, 0.8])
    add("((1.200000 * X1) / (0.800000 * X2))",
        lambda c, x: (c[0] * x[0]) / (c[1] * x[1]), [1.2, 0.8])

    # --- nested compositions ---
    add("sin(cos((1.300000 * X1)))",
        lambda c, x: jnp.sin(jnp.cos(c[0] * x[0])), [1.3])
    add("exp(sin((0.600000 * X1)))",
        lambda c, x: jnp.exp(jnp.sin(c[0] * x[0])), [0.6])
    add("log(square((0.900000 * X1)))",
        lambda c, x: jnp.log((c[0] * x[0]) ** 2), [0.9])
    add("tanh(exp((0.300000 * X1)))",
        lambda c, x: jnp.tanh(jnp.exp(c[0] * x[0])), [0.3])

    # --- three coefficients ---
    add("((1.300000 * sin((0.700000 * X1))) + 2.400000)",
        lambda c, x: c[0] * jnp.sin(c[1] * x[0]) + c[2],
        [1.3, 0.7, 2.4])
    add("(exp(((0.500000 * X1) + (0.300000 * X2))) * 1.700000)",
        lambda c, x: jnp.exp(c[0] * x[0] + c[1] * x[1]) * c[2],
        [0.5, 0.3, 1.7])
    add("((1.100000 * X1) / ((0.900000 * X2) + 1.500000))",
        lambda c, x: (c[0] * x[0]) / (c[1] * x[1] + c[2]),
        [1.1, 0.9, 1.5])

    # --- four coefficients ---
    add("((1.200000 * sin((0.800000 * X1))) + (0.600000 * cos((1.400000 * X2))))",
        lambda c, x: c[0] * jnp.sin(c[1] * x[0]) + c[2] * jnp.cos(c[3] * x[1]),
        [1.2, 0.8, 0.6, 1.4])

    # --- pow with variables ---
    add("((1.500000 * X1) ^ 2.300000)",
        lambda c, x: (c[0] * x[0]) ** c[1], [1.5, 2.3])
    add("(X1 ^ 1.700000)",
        lambda c, x: x[0] ** c[0], [1.7])

    # --- division quotient rule ---
    add("(sin((1.300000 * X1)) / cos((0.700000 * X2)))",
        lambda c, x: jnp.sin(c[0] * x[0]) / jnp.cos(c[1] * x[1]),
        [1.3, 0.7])

    # --- subtraction ---
    add("(sin((1.300000 * X1)) - cos((0.700000 * X2)))",
        lambda c, x: jnp.sin(c[0] * x[0]) - jnp.cos(c[1] * x[1]),
        [1.3, 0.7])

    # --- complex multi-term ---
    add("(((1.200000 * sin((0.800000 * X1))) * cos((1.400000 * X2))) + exp((0.300000 * X3)))",
        lambda c, x: c[0] * jnp.sin(c[1] * x[0]) * jnp.cos(c[2] * x[1]) + jnp.exp(c[3] * x[2]),
        [1.2, 0.8, 1.4, 0.3])

    # --- deep nesting ---
    add("sin(exp(cos((0.500000 * X1))))",
        lambda c, x: jnp.sin(jnp.exp(jnp.cos(c[0] * x[0]))), [0.5])
    add("tanh(sin(log((1.800000 * X1))))",
        lambda c, x: jnp.tanh(jnp.sin(jnp.log(c[0] * x[0]))), [1.8])

    # --- hyperbolic with chain rule ---
    add("(sinh((0.700000 * X1)) * cosh((0.500000 * X2)))",
        lambda c, x: jnp.sinh(c[0] * x[0]) * jnp.cosh(c[1] * x[1]),
        [0.7, 0.5])

    # --- inverse trig ---
    add("(asin((0.300000 * X1)) + atan((1.200000 * X2)))",
        lambda c, x: jnp.arcsin(c[0] * x[0]) + jnp.arctan(c[1] * x[1]),
        [0.3, 1.2])

    # --- cbrt + log1p ---
    add("(cbrt((1.500000 * X1)) + log1p((0.400000 * X2)))",
        lambda c, x: jnp.cbrt(c[0] * x[0]) + jnp.log1p(c[1] * x[1]),
        [1.5, 0.4])

    return cases


def compute_ground_truth(jax_fn, coeffs, data):
    """Compute residuals, Jacobian, and Hessian for all data rows."""
    c = jnp.array(coeffs, dtype=jnp.float64)

    def f_of_c(cc, xi):
        return jax_fn(cc, xi)

    jac_fn = jax.jacobian(f_of_c, argnums=0)
    hess_fn = jax.hessian(f_of_c, argnums=0)

    def per_row(xi):
        r = f_of_c(c, xi)
        j = jac_fn(c, xi)
        h = hess_fn(c, xi)
        return r, j, h

    residuals, jacobian, hessian = jax.vmap(per_row)(data)
    return (
        np.asarray(residuals, dtype=np.float64),
        np.asarray(jacobian, dtype=np.float64),
        np.asarray(hessian, dtype=np.float64),
    )


def upper_triangle(H):
    """Convert (nrows, p, p) → (nrows, p*(p+1)/2) upper triangle, row-major."""
    nrows, p, _ = H.shape
    if p == 0:
        return np.zeros((nrows, 0))
    cols = []
    for i in range(p):
        for j in range(i, p):
            cols.append(H[:, i, j])
    return np.column_stack(cols)


def write_file(path, data, results):
    nrows, nvars = data.shape
    ncases = len(results)

    with open(path, "w") as f:
        f.write(f"{ncases} {nrows} {nvars}\n")

        for row in data:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")

        for expr, coeffs, residuals, jacobian, hessian_tri in results:
            f.write(f"{expr}\n")
            nc = len(coeffs)
            f.write(f"{nc}\n")
            if nc > 0:
                f.write(" ".join(f"{v:.17g}" for v in coeffs) + "\n")

            f.write(" ".join(f"{v:.17g}" for v in residuals) + "\n")

            for row in jacobian:
                f.write(" ".join(f"{v:.17g}" for v in row) + "\n")

            for row in hessian_tri:
                f.write(" ".join(f"{v:.17g}" for v in row) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    # Data in (0.2, 0.9) to keep expressions safe for log/sqrt/asin
    data = rng.uniform(0.2, 0.9, (NUM_ROWS, NUM_VARS)).astype(np.float64)
    jdata = jnp.array(data)

    cases = make_cases()
    results = []

    for i, (expr, fn, coeffs) in enumerate(cases):
        try:
            residuals, jacobian, hessian = compute_ground_truth(fn, coeffs, jdata)
            hessian_tri = upper_triangle(hessian)
            results.append((expr, coeffs, residuals, jacobian, hessian_tri))
            print(f"[{i+1}/{len(cases)}] OK: {expr}  (p={len(coeffs)})")
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] FAIL: {expr}  — {e}", file=sys.stderr)
            raise

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo, "data", "hessian_ground_truth.txt")
    write_file(out_path, data, results)
    print(f"\nWrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
