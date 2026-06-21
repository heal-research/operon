#!/usr/bin/env python3
"""
Run an operon binary across multiple problems in parallel.

Config format:
    binary: ./build/operon_gp
    jobs: 4          # problems run in parallel
    threads: 8       # --threads N passed to each invocation
    reps: 30
    # adaptive: true / max_reps: 50 / tol: 0.005 / window: 5
    # base_seed: 42
    output_dir: results/
    shared_args:
      - --generations 500
      - --population-size 1000
    problems:
      - config: data/AirfoilSelfNoise.json   # reads target + canonical ranges
      - config: data/Friedman-I.json
        train: "0:300"                        # override canonical range
        test: "300:500"
      - dataset: data/Custom.csv             # fully explicit (no json)
        target: y
        train: "0:800"
        test: "800:1000"
        name: custom                          # optional display/file name
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from run_operon import run_reps, save


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json_meta(config_path: str) -> dict:
    with open(config_path) as f:
        meta = json.load(f)["metadata"]
    return {
        "dataset": str(Path(config_path).parent / meta["filename"]),
        "target":  meta["target"],
        "train":   f"{meta['training_rows']['start']}:{meta['training_rows']['end']}",
        "test":    f"{meta['test_rows']['start']}:{meta['test_rows']['end']}",
    }


def _resolve_problem(problem: dict) -> dict:
    base: dict = {}
    if "config" in problem:
        base = _load_json_meta(problem["config"])
    resolved = {**base, **{k: v for k, v in problem.items() if k != "config"}}
    if "name" not in resolved and "dataset" in resolved:
        resolved["name"] = Path(resolved["dataset"]).stem
    return resolved


def _run_problem(binary: str, problem: dict, shared_args: list[str],
                 threads: int, reps_kw: dict, output_dir: Path) -> tuple[str, pd.DataFrame]:
    p = _resolve_problem(problem)
    name = p["name"]

    args = [
        *shared_args,
        "--dataset", p["dataset"],
        "--target",  p["target"],
        "--train",   p["train"],
        "--test",    p["test"],
        "--threads", str(threads),
    ]

    df = run_reps(binary, args, **reps_kw)
    df["problem"] = name
    output = output_dir / f"{name}.feather"
    save(df, output)
    return name, df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    args = p.parse_args()

    cfg = load_config(args.config)

    binary      = cfg["binary"]
    jobs        = cfg.get("jobs", 1)
    threads     = cfg.get("threads", 1)
    output_dir  = Path(cfg.get("output_dir", "results"))
    shared_args = cfg.get("shared_args", [])
    problems    = cfg["problems"]

    reps_kw = dict(
        all_gens=cfg.get("all_gens", False),
        reps=cfg.get("reps"),
        adaptive=cfg.get("adaptive", False),
        max_reps=cfg.get("max_reps", 50),
        tol=cfg.get("tol", 0.005),
        window=cfg.get("window", 5),
        base_seed=cfg.get("base_seed"),
        show_progress=False,
    )

    console = Console()
    console.print(f"[bold]{len(problems)} problems, {jobs} parallel, {threads} threads each[/bold]")

    results: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_problem, binary, prob, shared_args, threads, reps_kw, output_dir): prob
            for prob in problems
        }
        for future in as_completed(futures):
            prob = futures[future]
            try:
                name, df = future.result()
                results[name] = df
                r2 = df["r2_te"].median() if "r2_te" in df.columns else float("nan")
                console.print(f"  [green]✓[/green] {name:30s}  r2_te={r2:.4f}")
            except Exception as exc:
                console.print(f"  [red]✗[/red] {prob}: {exc}")
                logger.exception(exc)

    table = Table(title="Summary")
    table.add_column("Problem")
    table.add_column("Reps", justify="right")
    table.add_column("r2_te median", justify="right")
    table.add_column("r2_te IQR", justify="right")

    for name, df in sorted(results.items()):
        if "r2_te" not in df.columns:
            continue
        r2 = df["r2_te"]
        iqr = r2.quantile(0.75) - r2.quantile(0.25)
        table.add_row(name, str(len(df)), f"{r2.median():.4f}", f"{iqr:.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
