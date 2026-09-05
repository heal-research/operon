#!/usr/bin/env python3
"""Run one replayable Operon tree-GP benchmark and write immutable SRBench results.

The output directory is created exclusively and contains `manifest.json` and
`results.json`.  `results.json` deliberately retains the field names used by
SRBench's experiment/optimize_model.py: dataset, algorithm, params,
random_state, time_time, symbolic_model, mse_{train,test}, mae_{train,test},
r2_{train,test}, and model_size.  The nested `operon` object links those
comparison fields to the complete replay provenance in `manifest.json`.

The separator `--` is required; everything after it is passed unchanged to
operon_gp.  The passed command must include `--dataset` and `--seed`, which
makes the dataset and random stream explicit in the recorded invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

SCHEMA = "https://operon.dev/schemas/srbench-run/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option_value(arguments: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix):]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def cpu_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {"architecture": platform.machine(), "processor": platform.processor()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text()
        model = re.search(r"^model name\s*: (.+)$", text, re.MULTILINE)
        flags = re.search(r"^flags\s*: (.+)$", text, re.MULTILINE)
        if model:
            identity["model"] = model.group(1)
        if flags:
            identity["features"] = flags.group(1).split()
    return identity


def parse_gp_output(stdout: str) -> tuple[dict[str, float], str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines) if "iteration" in line.split() and "r2_tr" in line.split()), None)
    if header_index is None:
        raise ValueError("operon_gp output did not contain its metrics table")
    fields = lines[header_index].split()
    rows: list[dict[str, float]] = []
    for line in lines[header_index + 1:]:
        values = line.split()
        if len(values) != len(fields):
            break
        try:
            rows.append(dict(zip(fields, map(float, values), strict=True)))
        except ValueError:
            break
    if not rows:
        raise ValueError("operon_gp metrics table did not contain a data row")
    model_lines = lines[header_index + 1 + len(rows):]
    if not model_lines:
        raise ValueError("operon_gp did not print a final symbolic model")
    return rows[-1], model_lines[-1]


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("error: use '--' before operon_gp arguments")
    separator = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="path to operon_gp")
    parser.add_argument("--output", required=True, type=Path, help="new immutable output directory")
    parsed = parser.parse_args(argv[:separator])
    gp_args = argv[separator + 1:]

    dataset_arg = option_value(gp_args, "--dataset")
    seed_arg = option_value(gp_args, "--seed")
    if dataset_arg is None or seed_arg is None:
        raise SystemExit("error: replayable runs require explicit --dataset and --seed after '--'")
    try:
        seed = int(seed_arg)
    except ValueError as error:
        raise SystemExit(f"error: --seed must be an integer: {error}") from error
    dataset = Path(dataset_arg).resolve()
    if not dataset.is_file():
        raise SystemExit(f"error: dataset is not a file: {dataset}")
    binary = parsed.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"error: operon_gp binary is not a file: {binary}")
    try:
        parsed.output.mkdir(mode=0o755)
    except FileExistsError as error:
        raise SystemExit(f"error: immutable output directory already exists: {parsed.output}") from error

    started = time.time()
    completed = subprocess.run([str(binary), *gp_args], capture_output=True, text=True, check=False)
    (parsed.output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (parsed.output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"error: operon_gp exited with status {completed.returncode}; see {parsed.output}/stderr.txt")
    metrics, model = parse_gp_output(completed.stdout)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    revision = re.search(r"operon rev\. (\S+)", version)
    repository = Path(__file__).resolve().parents[1]
    lock = repository / "flake.lock"
    elapsed = time.time() - started

    manifest = {
        "$schema": SCHEMA + "/manifest",
        "schema_version": 1,
        "schema_description": "Immutable replay provenance. Hashes are lowercase SHA-256; ranges use Operon's half-open start:end syntax.",
        "implementation": {"name": "Operon", "revision": revision.group(1) if revision else "unknown", "version": version},
        "dependency_lock": {"path": "flake.lock", "sha256": sha256(lock)} if lock.is_file() else None,
        "compiler": {"identity": version.splitlines()[-1] if version else "unknown"},
        "cpu": cpu_identity(),
        "seed": seed,
        "dataset": {"path": dataset.name, "sha256": sha256(dataset)},
        "primitive_set": {"enable_symbols": option_value(gp_args, "--enable-symbols"), "disable_symbols": option_value(gp_args, "--disable-symbols")},
        "algorithm": {"name": "operon_gp", "configuration": gp_args},
        "evaluator": {"objective": option_value(gp_args, "--objective") or "r2", "jit": option_value(gp_args, "--jit") or ""},
        "budget": {key: option_value(gp_args, "--" + key) for key in ("generations", "evaluations", "timelimit", "population-size", "pool-size", "iterations", "threads")},
        "result_schema": SCHEMA + "/results",
    }
    results = {
        "$schema": SCHEMA + "/results",
        "schema_version": 1,
        "schema_description": "SRBench-compatible comparison record; train/test metrics are evaluated by Operon on its configured ranges.",
        "dataset": dataset.stem,
        "algorithm": "Operon-GP",
        "params": {"operon_gp_arguments": gp_args},
        "random_state": seed,
        "time_time": elapsed,
        "grid_time": 0.0,
        "symbolic_model": model,
        "mse_train": metrics["mse_tr"],
        "mse_test": metrics["mse_te"],
        "mae_train": metrics["mae_tr"],
        "mae_test": metrics["mae_te"],
        "r2_train": metrics["r2_tr"],
        "r2_test": metrics["r2_te"],
        "model_size": int(metrics["best_len"]),
        "operon": {"manifest": "manifest.json", "best_fitness": metrics["best_fit"], "evaluator_calls": int(metrics["eval_cnt"]), "elapsed_seconds": metrics["elapsed"]},
    }
    write_exclusive(parsed.output / "manifest.json", manifest)
    write_exclusive(parsed.output / "results.json", results)
    print(parsed.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
