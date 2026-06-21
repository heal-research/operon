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
    # timeout: 120
    trials: 50
    study: friedman_jit_gate
    # storage: sqlite:///optuna.db   # omit for in-memory
    metric: r2_te                    # column to optimise (default: r2_te)
    direction: maximize              # or minimize (default: maximize)
    output: results/tune.feather     # all trial data; optional
    params:
      jit-min-visits: {type: int,   low: 1,   high: 20}
      jit-max-length: {type: int,   low: 10,  high: 200}
      population-size: {type: int,  low: 500, high: 2000, step: 100}
      crossover-probability: {type: float, low: 0.1, high: 1.0}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from loguru import logger

from _optuna import run_study
from run_operon import load_config, reps_kwargs_from_cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    args = p.parse_args()

    cfg = load_config(args.config)
    metric: str = cfg.get("metric", "r2_te")

    study, _ = run_study(
        binary=cfg["binary"],
        fixed_args=cfg.get("fixed_args", []),
        reps_kw=reps_kwargs_from_cfg(cfg),
        params=cfg["params"],
        metric=metric,
        direction=cfg.get("direction", "maximize"),
        trials=cfg.get("trials", 20),
        study_name=cfg.get("study", "operon_tune"),
        storage=cfg.get("storage"),
        trials_output=Path(cfg["output"]) if "output" in cfg else None,
    )

    best = study.best_trial
    logger.info(f"Best trial {best.number}: {best.params}  {metric}={best.value:.4f}")


if __name__ == "__main__":
    main()
