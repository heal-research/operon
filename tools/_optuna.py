"""Shared Optuna helpers used by tune_operon.py and operon_experiment.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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
        log  = spec.get("log", False)
        step = spec.get("step")
        if log and step is not None:
            raise ValueError(f"Param '{name}': log=true and step are mutually exclusive in Optuna")
        return trial.suggest_float(name, spec["low"], spec["high"], log=log, step=step)
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
    direction: str = "maximize",
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
            return float("-inf") if direction == "maximize" else float("inf")

        score = float(df[metric].median())

        df = df.copy()
        df["trial"] = trial.number
        for name, value in suggested.items():
            df[name] = value
        all_frames.append(df)

        logger.info(f"Trial {trial.number:3d}  {metric}={score:.4f}  {suggested}")
        return score

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=trials)
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # When resuming from persistent storage, all_frames only contains the trials
    # run in this session. Merge with any pre-existing trials_output so the file
    # always reflects the full history.
    if trials_output and not combined.empty:
        save(combined, trials_output, append=trials_output.exists())

    return study, combined
