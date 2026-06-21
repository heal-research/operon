"""Shared Optuna helpers used by tune_operon.py and operon_experiment.py."""
from pathlib import Path

import optuna
import pandas as pd
from loguru import logger

from run_operon import run_reps, save

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown param type '{t}' for '{name}'")


def run_study(
    binary: str,
    fixed_args: list[str],
    reps_kw: dict,
    params: dict,
    *,
    metric: str = "r2_te",
    trials: int = 20,
    study_name: str = "operon_tune",
    storage: str | None = None,
    trials_output: Path | None = None,
) -> tuple[optuna.Study, pd.DataFrame]:
    """Run an Optuna study; return (study, all-trials DataFrame)."""
    all_frames: list[pd.DataFrame] = []

    def objective(trial: optuna.Trial) -> float:
        suggested = {name: suggest(trial, name, spec) for name, spec in params.items()}
        args = list(fixed_args)
        for name, value in suggested.items():
            args += [f"--{name}", str(value)]

        df = run_reps(binary, args, **reps_kw)
        if df.empty or metric not in df.columns:
            return float("-inf")

        score = float(df[metric].median())

        df = df.copy()
        df["trial"] = trial.number
        for name, value in suggested.items():
            df[name] = value
        all_frames.append(df)

        if trials_output:
            save(pd.concat(all_frames, ignore_index=True), trials_output)

        logger.info(f"Trial {trial.number:3d}  {metric}={score:.4f}  {suggested}")
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=trials)
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return study, combined
