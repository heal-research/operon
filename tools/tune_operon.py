#!/usr/bin/env python3
"""
Hyperparameter search for an operon binary using Optuna.

Config format:
    binary: ./build/operon_gp
    fixed_args:
      - --dataset data/Friedman-II.csv
      - --target y
      - --train 0:400
      - --test 400:500
      - --generations 500
    reps: 10
    # adaptive: true / max_reps: 30 / tol: 0.005 / window: 5
    # base_seed: 42
    trials: 50
    study: friedman_jit_gate
    # storage: sqlite:///optuna.db   # omit for in-memory
    metric: r2_te                    # maximize this column (default: r2_te)
    output: results/tune.feather     # optional; updated after each trial
    params:
      jit-min-visits: {type: int,   low: 1,   high: 20}
      jit-max-length: {type: int,   low: 10,  high: 200}
      population-size: {type: int,  low: 500, high: 2000, step: 100}
      crossover-probability: {type: float, low: 0.1, high: 1.0}
"""
import argparse
from pathlib import Path

import optuna
import pandas as pd
import yaml
from loguru import logger

from run_operon import run_reps, save

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _reps_kwargs(cfg: dict) -> dict:
    return dict(
        all_gens=cfg.get("all_gens", False),
        reps=cfg.get("reps"),
        adaptive=cfg.get("adaptive", False),
        max_reps=cfg.get("max_reps", 50),
        tol=cfg.get("tol", 0.005),
        window=cfg.get("window", 5),
        base_seed=cfg.get("base_seed"),
        show_progress=False,
    )


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown param type '{t}' for '{name}'")


def make_objective(binary: str, fixed_args: list, reps_kw: dict,
                   params: dict, metric: str, output: Path | None,
                   all_results: list[pd.DataFrame]):
    def objective(trial: optuna.Trial) -> float:
        suggested = {name: _suggest(trial, name, spec) for name, spec in params.items()}
        args = [*fixed_args, *(f"--{k}" for kv in suggested.items() for k in kv)]

        # flatten: --name value pairs
        args = list(fixed_args)
        for name, value in suggested.items():
            args += [f"--{name}", str(value)]

        df = run_reps(binary, args, **reps_kw)
        if df.empty or metric not in df.columns:
            return float("-inf")

        score = float(df[metric].median())

        df["trial"] = trial.number
        for name, value in suggested.items():
            df[name] = value
        all_results.append(df)

        if output:
            combined = pd.concat(all_results, ignore_index=True)
            save(combined, output)

        logger.info(f"Trial {trial.number:3d}  {metric}={score:.4f}  {suggested}")
        return score

    return objective


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    args = p.parse_args()

    cfg = load_config(args.config)
    output = Path(cfg["output"]) if "output" in cfg else None
    metric: str = cfg.get("metric", "r2_te")

    study = optuna.create_study(
        direction="maximize",
        study_name=cfg.get("study", "operon_tune"),
        storage=cfg.get("storage"),
        load_if_exists=True,
    )

    all_results: list[pd.DataFrame] = []
    objective = make_objective(
        binary=cfg["binary"],
        fixed_args=cfg.get("fixed_args", []),
        reps_kw=_reps_kwargs(cfg),
        params=cfg["params"],
        metric=metric,
        output=output,
        all_results=all_results,
    )

    study.optimize(objective, n_trials=cfg.get("trials", 20))

    best = study.best_trial
    logger.info(f"Best trial {best.number}: {best.params}  {metric}={best.value:.4f}")


if __name__ == "__main__":
    main()
