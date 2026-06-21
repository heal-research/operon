#!/usr/bin/env python3
"""
Compare two operon binaries on the same problem using a YAML config.

Config format:
    binary_a: ./build/operon_gp
    binary_b: ./build/operon_gp_baseline
    shared_args:
      - --dataset data/Friedman-II.csv
      - --target y
      - --train 0:400
      - --test 400:500
      - --generations 500
    reps: 30
    # adaptive: true
    # max_reps: 50
    # tol: 0.005
    # window: 5
    # base_seed: 42
    # timeout: 120
    metric: r2_te          # column to compare (default: r2_te)
    output: results/compare.feather   # optional

CLI flags:
    --alpha A   significance level for Mann-Whitney U test (default: 0.05)

Note: scipy is optional. If not installed, the significance test is skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from run_operon import load_config, reps_kwargs_from_cfg, run_reps, save

try:
    from scipy import stats as _stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def compare_table(df_a: pd.DataFrame, df_b: pd.DataFrame,
                  name_a: str, name_b: str, metric: str, alpha: float) -> None:
    a = df_a[metric].dropna()
    b = df_b[metric].dropna()

    console = Console()
    table = Table(title=f"Comparison — {metric}")
    table.add_column("", style="bold")
    table.add_column(name_a, justify="right")
    table.add_column(name_b, justify="right")

    def iqr(s: pd.Series) -> float:
        return float(s.quantile(0.75) - s.quantile(0.25))

    for label, fa, fb in [
        ("n",      f"{len(a)}",           f"{len(b)}"),
        ("median", f"{a.median():.4f}",   f"{b.median():.4f}"),
        ("IQR",    f"{iqr(a):.4f}",       f"{iqr(b):.4f}"),
        ("mean",   f"{a.mean():.4f}",     f"{b.mean():.4f}"),
        ("std",    f"{a.std():.4f}",      f"{b.std():.4f}"),
    ]:
        table.add_row(label, fa, fb)

    console.print(table)

    if _HAVE_SCIPY:
        _, pvalue = _stats.mannwhitneyu(a, b, alternative="two-sided")
        sig = "significant" if pvalue < alpha else "not significant"
        console.print(f"Mann-Whitney U  p={pvalue:.4f}  ({sig} at α={alpha})")
    else:
        console.print("[yellow]scipy not available — skipping significance test[/yellow]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--alpha", type=float, default=0.05, metavar="A",
                   help="Significance level for Mann-Whitney U test (default: 0.05)")
    args = p.parse_args()

    cfg = load_config(args.config)
    shared_args: list[str] = cfg.get("shared_args", [])
    metric: str = cfg.get("metric", "r2_te")
    kwargs = reps_kwargs_from_cfg(cfg)

    name_a = cfg.get("name_a", Path(cfg["binary_a"]).name)
    name_b = cfg.get("name_b", Path(cfg["binary_b"]).name)

    logger.info(f"Running {name_a}")
    df_a = run_reps(cfg["binary_a"], shared_args, **kwargs)

    logger.info(f"Running {name_b}")
    df_b = run_reps(cfg["binary_b"], shared_args, **kwargs)

    if df_a.empty or df_b.empty:
        logger.error("One or both binaries produced no output.")
        sys.exit(1)

    if output := cfg.get("output"):
        df_a["binary"] = name_a
        df_b["binary"] = name_b
        combined = pd.concat([df_a, df_b], ignore_index=True)
        save(combined, Path(output))

    compare_table(df_a, df_b, name_a, name_b, metric, args.alpha)


if __name__ == "__main__":
    main()
