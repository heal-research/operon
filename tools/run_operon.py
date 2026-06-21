#!/usr/bin/env python3
"""
Run an operon binary, capture per-generation stats, save to CSV or feather.

Usage:
    run_operon.py BINARY --output PATH [--all-gens] [--reps N]
                  [--adaptive [--max-reps N] [--tol F] [--window K]]
                  [--base-seed N] [--timeout S] [--append] [-- BINARY_ARGS...]
"""
import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_output(stdout: str, all_gens: bool) -> pd.DataFrame:
    lines = stdout.strip().splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if "iteration" in ln), None)
    if header_idx is None:
        return pd.DataFrame()

    header = lines[header_idx].split()
    n = len(header)

    def _is_data_line(ln: str) -> bool:
        parts = ln.split()
        if len(parts) != n:
            return False
        try:
            float(parts[0])   # first column is always a numeric iteration/generation counter
            return True
        except ValueError:
            return False

    data_lines = [ln for ln in lines[header_idx + 1:] if ln.strip() and _is_data_line(ln)]
    if not data_lines:
        return pd.DataFrame()

    rows = [ln.split() for ln in data_lines]
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df if all_gens else df.iloc[[-1]]


def run_once(binary: str, args: list[str], all_gens: bool, timeout: int | None = None) -> pd.DataFrame:
    try:
        result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"Binary timed out after {timeout}s")
        return pd.DataFrame()
    if result.returncode != 0:
        logger.warning(f"Binary exited {result.returncode}: {result.stderr.strip()[:200]}")
        return pd.DataFrame()
    return parse_output(result.stdout, all_gens)


def run_reps(
    binary: str,
    binary_args: list[str],
    *,
    all_gens: bool = False,
    reps: int | None = None,
    adaptive: bool = False,
    max_reps: int = 50,
    tol: float = 0.005,
    window: int = 5,
    base_seed: int | None = None,
    metric: str = "r2_te",
    timeout: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    n = reps if reps is not None else max_reps
    frames: list[pd.DataFrame] = []
    metric_history: list[float] = []

    def _args_for_rep(i: int) -> list[str]:
        if base_seed is None:
            return binary_args
        return [*binary_args, "--seed", str(base_seed + i)]

    def _do_reps(progress=None, task=None):
        for i in range(n):
            df = run_once(binary, _args_for_rep(i), all_gens, timeout)
            if df.empty:
                continue
            df.insert(0, "rep", i)
            frames.append(df)

            desc = f"rep {i + 1}/{n}"
            if adaptive and not df.empty and metric in df.columns:
                metric_history.append(float(df[metric].iloc[-1]))
                if len(metric_history) >= window:
                    std = statistics.stdev(metric_history[-window:])
                    desc = f"rep {i + 1}  {metric} σ={std:.4f}"
                    if std < tol:
                        logger.info(f"Adaptive stop at rep {i + 1}: σ={std:.4f} < tol={tol}")
                        if progress:
                            progress.update(task, advance=1, description=desc)
                        break

            if progress:
                progress.update(task, advance=1, description=desc)

    if show_progress:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), MofNCompleteColumn(), transient=True) as prog:
            task = prog.add_task("Starting...", total=n)
            _do_reps(prog, task)
    else:
        _do_reps()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(df: pd.DataFrame, output: Path, *, append: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".feather":
        if append and output.exists():
            existing = pd.read_feather(output)
            if "rep" in existing.columns and "rep" in df.columns:
                df = df.copy()
                df["rep"] += int(existing["rep"].max()) + 1
            df = pd.concat([existing, df], ignore_index=True)
        df.to_feather(output)
    elif output.suffix == ".csv":
        df.to_csv(output, index=False,
                  mode="a" if append else "w",
                  header=not (append and output.exists()))
    else:
        raise ValueError(f"Unsupported format '{output.suffix}': use .csv or .feather")
    logger.info(f"Saved {len(df)} rows → {output}")


def reps_kwargs_from_cfg(cfg: dict, *, show_progress: bool = True) -> dict:
    """Extract run_reps keyword arguments from a YAML config dict."""
    adaptive = cfg.get("adaptive", False)
    return dict(
        all_gens=cfg.get("all_gens", False),
        reps=cfg.get("reps", None if adaptive else 1),
        adaptive=adaptive,
        max_reps=cfg.get("max_reps", 50),
        tol=cfg.get("tol", 0.005),
        window=cfg.get("window", 5),
        base_seed=cfg.get("base_seed"),
        metric=cfg.get("metric", "r2_te"),
        timeout=cfg.get("timeout"),
        show_progress=show_progress,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Arguments after -- are passed verbatim to the binary.",
    )
    p.add_argument("binary", help="Path to operon binary")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--all-gens", action="store_true", help="Keep all generations (default: last only)")
    p.add_argument("--reps", type=int, default=None, help="Fixed number of repetitions")
    p.add_argument("--adaptive", action="store_true", help="Stop when metric variation settles")
    p.add_argument("--max-reps", type=int, default=50, metavar="N",
                   help="Safety cap for adaptive mode (default: 50)")
    p.add_argument("--tol", type=float, default=0.005, metavar="F",
                   help="Rolling std threshold for adaptive stop (default: 0.005)")
    p.add_argument("--window", type=int, default=5, metavar="K",
                   help="Window size for adaptive stopping (default: 5)")
    p.add_argument("--metric", default="r2_te", metavar="COL",
                   help="Column used for adaptive stopping (default: r2_te)")
    p.add_argument("--base-seed", type=int, default=None, metavar="N",
                   help="Seed for rep 0; rep i gets base-seed+i")
    p.add_argument("--timeout", type=int, default=None, metavar="S",
                   help="Per-run timeout in seconds; hung binary is killed and skipped")
    p.add_argument("--append", action="store_true",
                   help="Append to existing output (feather: accumulate across experiments)")
    return p


def main() -> None:
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
        our_argv, binary_args = argv[:sep], argv[sep + 1:]
    except ValueError:
        our_argv, binary_args = argv, []

    args = _build_parser().parse_args(our_argv)

    if not args.reps and not args.adaptive:
        args.reps = 1

    df = run_reps(
        args.binary,
        binary_args,
        all_gens=args.all_gens,
        reps=args.reps,
        adaptive=args.adaptive,
        max_reps=args.max_reps,
        tol=args.tol,
        window=args.window,
        metric=args.metric,
        base_seed=args.base_seed,
        timeout=args.timeout,
    )

    if df.empty:
        logger.warning("No data produced.")
        sys.exit(1)

    save(df, args.output, append=args.append)


if __name__ == "__main__":
    main()
