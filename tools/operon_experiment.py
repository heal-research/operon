#!/usr/bin/env python3
"""
Run an operon binary across multiple problems in parallel.

Config format (fixed reps):
    binary: ./build/operon_gp
    jobs: 4          # problems run in parallel
    threads: 8       # --threads N passed to each invocation
    reps: 30
    # adaptive: true / max_reps: 50 / tol: 0.005 / window: 5 / base_seed: 42
    # timeout: 120   # per-run timeout in seconds
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
        name: custom

Config format (with tuning — Optuna replaces fixed reps):
    ...
    tune:
      trials: 20
      metric: r2_te
      direction: maximize   # or minimize (default: maximize)
      # storage: sqlite:///optuna.db
      params:
        allowed-symbols:
          type: categorical
          choices:
            - "add,sub,mul,div"
            - "add,sub,mul,div,sqrt"
            - "add,sub,mul,div,exp,log"
        max-length: {type: int, low: 10, high: 50, step: 10}
    problems: ...

    When 'tune' is present, each problem runs an independent Optuna study.
    The best trial's result is saved to {output_dir}/{name}.feather;
    all trial data is saved to {output_dir}/{name}_trials.feather.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from run_operon import load_config, reps_kwargs_from_cfg, run_reps, save


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


def _run_problem(binary: str, problem: dict, shared_args: list[str], threads: int,
                 reps_kw: dict, tune_cfg: dict | None,
                 output_dir: Path) -> tuple[str, pd.DataFrame]:
    from _optuna import run_study  # local import — only needed in worker process

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

    if tune_cfg:
        study, all_trials = run_study(
            binary=binary,
            fixed_args=args,
            reps_kw=reps_kw,
            params=tune_cfg["params"],
            metric=tune_cfg.get("metric", "r2_te"),
            direction=tune_cfg.get("direction", "maximize"),
            trials=tune_cfg.get("trials", 20),
            study_name=f"operon_{name}",
            storage=tune_cfg.get("storage"),
            trials_output=output_dir / f"{name}_trials.feather",
        )
        # Keep only the best trial's rows as the problem result.
        df = all_trials[all_trials["trial"] == study.best_trial.number].copy()
    else:
        df = run_reps(binary, args, **reps_kw)

    df["problem"] = name
    save(df, output_dir / f"{name}.feather")
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
    tune_cfg    = cfg.get("tune")

    reps_kw = reps_kwargs_from_cfg(cfg, show_progress=False)

    console = Console()
    mode = "tuning" if tune_cfg else f"reps={cfg.get('reps', 1)}"
    console.print(f"[bold]{len(problems)} problems, {jobs} parallel, {threads} threads each, {mode}[/bold]")

    results: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_problem, binary, prob, shared_args, threads,
                        reps_kw, tune_cfg, output_dir): prob
            for prob in problems
        }
        for future in as_completed(futures):
            prob = futures[future]
            try:
                name, df = future.result()
                results[name] = df
                metric = tune_cfg.get("metric", "r2_te") if tune_cfg else "r2_te"
                val = df[metric].median() if metric in df.columns else float("nan")
                console.print(f"  [green]✓[/green] {name:30s}  {metric}={val:.4f}")
            except Exception as exc:
                console.print(f"  [red]✗[/red] {prob}: {exc}")
                logger.exception(exc)

    metric = tune_cfg.get("metric", "r2_te") if tune_cfg else "r2_te"
    table = Table(title="Summary")
    table.add_column("Problem")
    table.add_column("Reps" if not tune_cfg else "Best trial", justify="right")
    table.add_column(f"{metric} median", justify="right")
    table.add_column(f"{metric} IQR", justify="right")

    for name, df in sorted(results.items()):
        if metric not in df.columns:
            continue
        col = df[metric]
        iqr = col.quantile(0.75) - col.quantile(0.25)
        n = str(df["trial"].iloc[0]) if tune_cfg and "trial" in df.columns else str(len(df))
        table.add_row(name, n, f"{col.median():.4f}", f"{iqr:.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
