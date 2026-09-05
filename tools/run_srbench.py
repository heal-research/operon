#!/usr/bin/env python3
"""Run one replayable Operon GP benchmark.

The wrapper consumes operon_gp's versioned ``--report-json`` report, rather
than scraping human terminal output.  Output is published atomically only
after the child exits successfully and report validation has completed.
Results are Operon-specific schema v2: ``tree_node_count`` is the raw Operon
genotype length and is *not* SRBench model complexity.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, re, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

SCHEMA = "https://operon.dev/schemas/srbench-run/v2"
REPLAY_COMPARISON = {
    "required_equal": ["symbolic_model", "tree_node_count", "metrics.r2_train", "metrics.r2_test", "metrics.mae_train", "metrics.mae_test", "metrics.mse_train", "metrics.mse_test", "metrics.nmse_train", "metrics.nmse_test", "metrics.best_fitness", "metrics.evaluator_calls", "metrics.result_evaluations", "params.canonical_argv", "random_state", "operon.manifest", "manifest.provenance"],
    "observational_excluded": ["time_time", "metrics.elapsed_seconds", "metrics.optimizer_seconds"],
    "note": "Replay equality requires model, loss metrics, effective configuration/budget, and provenance equality. Wall-clock and optimizer timing are observational and excluded because they vary between otherwise identical runs."
}
SINGLETONS = {
    "dataset", "seed", "train", "test", "target", "inputs", "objective", "jit",
    "shuffle", "standardize", "linear-scaling", "skip-nonfinite", "population-size", "pool-size",
    "generations", "evaluations", "iterations", "timelimit", "threads", "maxlength", "maxdepth",
    "creator", "creator-mindepth", "creator-maxdepth", "crossover-probability", "crossover-internal-probability",
    "mutation-probability", "local-search-probability", "lamarckian-probability", "symbolic", "transposition-cache",
    "cache-max-age", "female-selector", "male-selector", "offspring-generator", "reinserter", "mutators", "elitism",
    "enable-symbols", "disable-symbols", "report-json", "probes-config", "shape-constraints-config", "shape-enforcement",
    "shape-penalty-weight", "shape-unknown-violation", "shape-worst-value", "shape-bound-mode",
}
VALUE_OPTIONS = SINGLETONS - {"shuffle", "standardize", "linear-scaling", "skip-nonfinite", "symbolic", "transposition-cache"}
PATH_OPTIONS = {"dataset", "shape-constraints-config", "probes-config", "checkpoint-file", "resume", "pareto-front"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"machine report field {name!r} must be finite")
    return float(value)

def parse_args(args: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--": raise ValueError("unexpected '--' in GP arguments")
        if not token.startswith("--"): raise ValueError(f"unexpected positional GP argument {token!r}")
        name, eq, value = token[2:].partition("=")
        if name not in SINGLETONS and name not in {"probes-config"}:
            i += 1; continue
        if name in values: raise ValueError(f"duplicate singleton GP option --{name}")
        if not eq:
            if name in VALUE_OPTIONS:
                if i + 1 >= len(args) or args[i + 1].startswith("--"):
                    raise ValueError(f"option --{name} requires a value")
                i += 1; value = args[i]
            else: value = True
        values[name] = value
        i += 1
    return values

def canonical_args(args: list[str], cwd: Path) -> tuple[list[str], dict[str, Any]]:
    vals = parse_args(args)
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]; name, eq, value = token[2:].partition("=") if token.startswith("--") else ("", "", "")
        if name in PATH_OPTIONS:
            if not eq:
                i += 1; value = args[i]
            resolved = str((cwd / value).resolve())
            out.append(f"--{name}={resolved}")
        elif name in SINGLETONS or name == "probes-config":
            if not eq and name in VALUE_OPTIONS:
                i += 1; value = args[i]
            out.append(f"--{name}={value}" if eq or name in VALUE_OPTIONS else f"--{name}")
        else: out.append(token)
        i += 1
    return out, vals

def cpu_identity() -> dict[str, Any]:
    result: dict[str, Any] = {"architecture": platform.machine(), "processor": platform.processor()}
    p = Path("/proc/cpuinfo")
    if p.exists():
        text = p.read_text(errors="replace")
        for key, field in (("model name", "model"), ("flags", "features")):
            m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE)
            if m: result[field] = m.group(1).split() if field == "features" else m.group(1)
    return result
def validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("report_kind") != "operon_gp" or report.get("schema_version") != 1:
        raise ValueError("machine report has unsupported kind or schema version")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict): raise ValueError("machine report lacks metrics object")
    for k, v in metrics.items():
        if isinstance(v, float): finite(v, k)
    required = {"r2_train", "r2_test", "mae_train", "mae_test", "mse_train", "mse_test", "best_fitness", "elapsed_seconds"}
    missing = required - metrics.keys()
    if missing: raise ValueError(f"machine report missing fields: {', '.join(sorted(missing))}")
    return report

def compare_artifacts(first: Path, second: Path) -> None:
    """Validate deterministic replay fields, excluding declared timing fields."""
    a, b = (json.loads((p / "results.json").read_text()) for p in (first, second))
    policy = json.loads((first / "manifest.json").read_text())["replay_comparison"]
    for path in policy["required_equal"]:
        if path == "manifest.provenance": continue
        left = a; right = b
        for component in path.split("."):
            left = left[component]; right = right[component]
        if left != right: raise ValueError(f"replay mismatch in required field {path}")
    if policy["observational_excluded"] != ["time_time", "metrics.elapsed_seconds", "metrics.optimizer_seconds"]:
        raise ValueError("unsupported replay comparison policy")

def main() -> int:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, nargs="?")
    parser.add_argument("--output", required=False, type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0, help="finite wrapper timeout in seconds")
    parser.add_argument("--replay-manifest", type=Path, help="compare this artifact with --against-artifact")
    parser.add_argument("--against-artifact", type=Path, help="second artifact for --replay-manifest comparison")
    parsed = parser.parse_args(argv[:split] if "--" in argv else argv)
    if parsed.replay_manifest:
        if not parsed.against_artifact: raise SystemExit("error: --replay-manifest requires --against-artifact")
        try: compare_artifacts(parsed.replay_manifest.resolve(), parsed.against_artifact.resolve())
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e: raise SystemExit(f"error: replay comparison failed: {e}")
        print("replay comparison passed (timing fields excluded by manifest policy)"); return 0
    if "--" not in argv: raise SystemExit("error: use '--' before operon_gp arguments")
    if parsed.binary is None or parsed.output is None: raise SystemExit("error: binary and --output are required")
    gp_args = argv[split + 1:]
    if not math.isfinite(parsed.timeout) or parsed.timeout <= 0: raise SystemExit("error: --timeout must be finite and positive")
    cwd = Path.cwd().resolve(); binary = parsed.binary.resolve()
    if not binary.is_file(): raise SystemExit(f"error: operon_gp binary is not a file: {binary}")
    canonical, opts = canonical_args(gp_args, cwd)
    if "dataset" not in opts or "seed" not in opts: raise SystemExit("error: replayable runs require explicit --dataset and --seed")
    try: seed = int(opts["seed"])
    except (TypeError, ValueError): raise SystemExit("error: --seed must be an integer")
    budget = opts.get("evaluations", 0); limit = opts.get("timelimit", 2**64 - 1)
    try: budget_finite = int(budget) > 0; limit_finite = int(limit) < 2**64 - 1 and int(limit) > 0
    except (TypeError, ValueError): budget_finite = limit_finite = False
    if not budget_finite and not limit_finite: raise SystemExit("error: require a finite positive --evaluations or --timelimit")
    dataset = Path(str(opts["dataset"])).resolve()
    if not dataset.is_file(): raise SystemExit(f"error: dataset is not a file: {dataset}")
    output = parsed.output.resolve()
    if output.exists(): raise SystemExit(f"error: immutable output already exists: {output}")
    parent = output.parent; parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    report_path = stage / "machine-report.json"
    # report path is wrapper-owned and cannot be overridden by duplicate CLI input.
    command = [str(binary), *gp_args, "--report-json", str(report_path)]
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try: stdout, stderr = proc.communicate(timeout=parsed.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); stdout, stderr = proc.communicate()
            raise RuntimeError(f"operon_gp exceeded wrapper timeout of {parsed.timeout:g}s")
        if proc.returncode != 0: raise RuntimeError(f"operon_gp exited with status {proc.returncode}: {stderr.strip()}")
        if not report_path.is_file(): raise RuntimeError("operon_gp did not produce --report-json")
        try: report = validate_report(json.loads(report_path.read_text()))
        except (json.JSONDecodeError, ValueError) as e: raise RuntimeError(f"invalid machine report: {e}") from e
        (stage / "stdout.txt").write_text(stdout); (stage / "stderr.txt").write_text(stderr)
        version_run = subprocess.run([str(binary), "--version"], cwd=cwd, capture_output=True, text=True, check=False)
        version = version_run.stdout + version_run.stderr
        inputs = []
        for key in PATH_OPTIONS:
            if key in opts:
                p = Path(str(opts[key])).resolve()
                if p.is_file(): inputs.append({"option": f"--{key}", "path": str(p), "sha256": sha256(p), "size": p.stat().st_size})
        stat = binary.stat()
        manifest = {
            "$schema": SCHEMA + "/manifest", "schema_version": 2,
            "schema_description": "Replay provenance; timing fields are observational and not replay-equality claims.",
            "replay_comparison": REPLAY_COMPARISON,
            "invocation": {"argv": [str(binary), *canonical], "cwd": str(cwd), "machine_report_option": "--report-json <private-staging-path>"},
            "inputs": sorted(inputs, key=lambda x: x["option"]), "effective_configuration": {k: opts[k] for k in sorted(opts)},
            "implementation": {"name": "Operon", "version_output": version, "executable": {"path": str(binary), "size": stat.st_size, "sha256": sha256(binary)}},
            "dependency_lock": {"path": str((cwd / "flake.lock").resolve()), "sha256": sha256(cwd / "flake.lock")} if (cwd / "flake.lock").is_file() else None,
            "compiler": None, "cpu": cpu_identity(), "resource_budget": {"wrapper_timeout_seconds": parsed.timeout, "evaluations": budget if budget_finite else None, "timelimit": limit if limit_finite else None, "status": "completed"},
            "dataset": {"path": str(dataset), "sha256": sha256(dataset), "size": dataset.stat().st_size}, "seed": seed, "result_schema": SCHEMA + "/results",
        }
        metrics = report["metrics"]
        results = {"$schema": SCHEMA + "/results", "schema_version": 2, "schema_description": "Operon-specific record; tree_node_count is genotype length, not SRBench model complexity.", "srbench_compatibility": False, "dataset": dataset.stem, "algorithm": "Operon-GP", "params": {"canonical_argv": canonical}, "random_state": seed, "time_time": finite(metrics["elapsed_seconds"], "elapsed_seconds"), "symbolic_model": report["symbolic_model"], "tree_node_count": report["tree_node_count"], "metrics": metrics, "operon": {"manifest": "manifest.json"}}
        for payload, name in ((manifest, "manifest.json"), (results, "results.json")):
            with (stage / name).open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
        os.replace(stage, output); print(output / "results.json"); return 0
    except (OSError, RuntimeError, ValueError) as e:
        shutil.rmtree(stage, ignore_errors=True); print(f"error: {e}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
