#!/usr/bin/env python3
"""Run one replayable Operon GP benchmark with verifiable provenance."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, re, shutil, signal, stat, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

SCHEMA = "https://operon.dev/schemas/srbench-run/v2"
# This policy is wrapper code, not data read from either artifact.  Timing and host
# observations are deliberately excluded; every other published field is classified.
REPLAY_COMPARISON = {
    "required_equal": ["implementation", "inputs", "dependency_lock", "effective_configuration", "seed", "resource_budget", "result.deterministic"],
    "observational_excluded": ["result.time_time", "result.metrics.elapsed_seconds", "result.metrics.optimizer_seconds", "manifest.cpu", "manifest.invocation.cwd"],
    "policy_version": 1,
}

# One authoritative description of every effective operon_gp option.  The C++ CLI
# uses the same names/defaults (see cli/source/util.cpp); this table is only a
# parser/canonicalizer, never a second accepted-option implementation.
BOOL_OPTIONS = {"shuffle", "standardize", "linear-scaling", "skip-nonfinite", "symbolic", "transposition-cache", "show-primitives", "debug"}
VALUE_DEFAULTS = {
    "epsilon": "1e-6", "objective": "r2", "jit": "", "jit-max-length": "0", "jit-min-visits": "1",
    "population-size": "1000", "pool-size": "1000", "seed": "0", "generations": "1000", "evaluations": "1000000",
    "iterations": "0", "selection-pressure": "100", "maxlength": "50", "maxdepth": "10",
    "crossover-probability": "1.0", "crossover-internal-probability": "0.9", "mutation-probability": "0.25",
    "creator": "btc", "creator-mindepth": "1", "creator-maxdepth": "100", "female-selector": "tournament",
    "male-selector": "tournament", "offspring-generator": "basic", "reinserter": "keep-best",
    "mutators": "onepoint:1,changevar:1,changefunc:1,replacesubtree:1,insertsubtree:1,removesubtree:1,discretepoint:1",
    "local-search-probability": "1.0", "lamarckian-probability": "1.0", "threads": "0",
    "timelimit": str(2**64 - 1), "cache-max-age": "0", "model-selection": "obj0", "mdl-likelihood": "gaussian",
    "checkpoint-interval": "0", "checkpoint-file": "checkpoint.beve", "nonfinite-penalty-weight": "1.0",
    "shape-penalty-weight": "1.0", "shape-unknown-violation": "1.0", "shape-worst-value": "1.0", "shape-bound-mode": "combined",
}
VALUE_OPTIONS = set(VALUE_DEFAULTS) | {"dataset", "train", "test", "target", "inputs", "enable-symbols", "disable-symbols", "pareto-front", "resume", "probes-config", "shape-constraints-config", "elitism"}
ALL_OPTIONS = set(VALUE_OPTIONS) | BOOL_OPTIONS
PATH_OPTIONS = {"dataset", "shape-constraints-config", "probes-config", "resume"}
OUTPUT_PATH_OPTIONS = {"checkpoint-file", "pareto-front"}
OBSERVATIONAL_RESULT_FIELDS = {"time_time", "metrics.elapsed_seconds", "metrics.optimizer_seconds"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def snapshot(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "sha256": sha256(path), "size": st.st_size, "mode": stat.S_IMODE(st.st_mode)}


def finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"machine report field {name!r} must be finite")
    return float(value)


def _json_strict(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonstandard JSON constant {x}")))


def parse_args(args: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--": raise ValueError("unexpected '--' in GP arguments")
        if not token.startswith("--"): raise ValueError(f"unexpected positional GP argument {token!r}")
        name, eq, value = token[2:].partition("=")
        if name == "report-json": raise ValueError("--report-json is wrapper-owned and must not be supplied")
        if name not in ALL_OPTIONS: raise ValueError(f"unknown operon_gp option --{name}")
        if name in values: raise ValueError(f"duplicate singleton GP option --{name}")
        if not eq:
            if name == "jit": value = "all"
            elif name in VALUE_OPTIONS:
                if i + 1 >= len(args) or args[i + 1].startswith("--"): raise ValueError(f"option --{name} requires a value")
                i += 1; value = args[i]
            else: value = True
        elif name in BOOL_OPTIONS: raise ValueError(f"boolean option --{name} does not take a value")
        values[name] = value
        i += 1
    return values


def canonical_args(args: list[str], cwd: Path) -> tuple[list[str], dict[str, Any]]:
    vals = parse_args(args); out: list[str] = []; i = 0
    while i < len(args):
        token = args[i]; name, eq, value = token[2:].partition("=")
        if not eq and name == "jit": value = "all"
        elif not eq and name in VALUE_OPTIONS: i += 1; value = args[i]
        if name in PATH_OPTIONS or name in OUTPUT_PATH_OPTIONS: value = str((cwd / value).resolve())
        out.append(f"--{name}" if name in BOOL_OPTIONS else f"--{name}={value}"); i += 1
    effective = dict(VALUE_DEFAULTS)
    effective.update({k: v for k, v in vals.items() if k not in BOOL_OPTIONS})
    for k in BOOL_OPTIONS: effective[k] = bool(vals.get(k, False if k != "linear-scaling" else True))
    effective["jit"] = vals.get("jit", VALUE_DEFAULTS["jit"])
    effective["canonical_argv"] = out
    return out, effective


def cpu_identity() -> dict[str, Any]:
    result: dict[str, Any] = {"architecture": platform.machine(), "processor": platform.processor()}
    p = Path("/proc/cpuinfo")
    if p.exists():
        text = p.read_text(errors="replace")
        for key, field in (("model name", "model"), ("flags", "features")):
            m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE)
            if m: result[field] = m.group(1).split() if field == "features" else m.group(1)
    return result


def _number(value: Any, name: str, integer: bool = False) -> None:
    if integer:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0: raise ValueError(f"machine report field {name!r} must be a non-negative integer")
    else: finite(value, name)


def validate_report(report: Any) -> dict[str, Any]:
    root = {"report_kind", "schema_version", "symbolic_model", "tree_node_count", "seed", "metrics"}
    if not isinstance(report, dict) or set(report) != root or report.get("report_kind") != "operon_gp" or report.get("schema_version") != 1:
        raise ValueError("machine report has unsupported or incomplete schema")
    if not isinstance(report["symbolic_model"], str): raise ValueError("machine report symbolic_model must be a string")
    _number(report["tree_node_count"], "tree_node_count", True); _number(report["seed"], "seed", True)
    metric_types = {"r2_train": 0, "r2_test": 0, "mae_train": 0, "mae_test": 0, "nmse_train": 0, "nmse_test": 0, "mse_train": 0, "mse_test": 0, "best_fitness": 0, "average_fitness": 0, "average_tree_node_count": 0, "elapsed_seconds": 0, "evaluator_calls": 1, "result_evaluations": 1, "jacobian_evaluations": 1, "optimizer_seconds": 0}
    metrics = report["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(metric_types): raise ValueError("machine report metrics schema is incomplete or has extra fields")
    for key, integer in metric_types.items(): _number(metrics[key], key, bool(integer))
    return report


def _manifest_and_results(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _json_strict(root / "manifest.json"); results = _json_strict(root / "results.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2: raise ValueError("invalid manifest schema")
    if not isinstance(results, dict) or results.get("schema_version") != 2: raise ValueError("invalid results schema")
    return manifest, results


def compare_artifacts(first: Path, second: Path) -> None:
    ma, ra = _manifest_and_results(first); mb, rb = _manifest_and_results(second)
    # Validate both independently, and reject artifact-supplied comparison policies.
    for m in (ma, mb):
        if m.get("replay_comparison") != REPLAY_COMPARISON: raise ValueError("artifact contains unsupported replay policy")
    for path in ("implementation", "inputs", "dependency_lock", "effective_configuration", "seed", "resource_budget"):
        if ma.get(path) != mb.get(path): raise ValueError(f"replay mismatch in manifest field {path}")
    deterministic_a = {k: v for k, v in ra.items() if k not in {"time_time"}}
    deterministic_b = {k: v for k, v in rb.items() if k not in {"time_time"}}
    for path in OBSERVATIONAL_RESULT_FIELDS:
        if path.startswith("metrics."): deterministic_a["metrics"].pop(path.split(".")[1], None); deterministic_b["metrics"].pop(path.split(".")[1], None)
    if deterministic_a != deterministic_b: raise ValueError("replay mismatch in deterministic result fields")


def _run_bounded(command: list[str], cwd: Path, timeout: float) -> tuple[str, str, int]:
    proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try: stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGTERM)
        try: stdout, stderr = proc.communicate(timeout=min(5.0, max(0.1, timeout)))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); stdout, stderr = proc.communicate()
        raise RuntimeError(f"helper exceeded timeout of {timeout:g}s") from exc
    return stdout, stderr, proc.returncode


def main() -> int:
    argv = sys.argv[1:]; split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, nargs="?"); parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0); parser.add_argument("--replay-manifest", type=Path); parser.add_argument("--against-artifact", type=Path)
    try: parsed = parser.parse_args(argv[:split] if "--" in argv else argv)
    except SystemExit as e: return int(e.code)
    try:
        if parsed.replay_manifest:
            if not parsed.against_artifact: raise ValueError("--replay-manifest requires --against-artifact")
            compare_artifacts(parsed.replay_manifest.resolve(), parsed.against_artifact.resolve()); print("replay comparison passed"); return 0
        if "--" not in argv: raise ValueError("use '--' before operon_gp arguments")
        if parsed.binary is None or parsed.output is None: raise ValueError("binary and --output are required")
        if not math.isfinite(parsed.timeout) or parsed.timeout <= 0: raise ValueError("--timeout must be finite and positive")
        cwd = Path.cwd().resolve(); binary = parsed.binary.resolve()
        if not binary.is_file(): raise ValueError(f"operon_gp binary is not a file: {binary}")
        gp_args = argv[split + 1:]; canonical, opts = canonical_args(gp_args, cwd)
        if "dataset" not in opts or "seed" not in opts: raise ValueError("replayable runs require explicit --dataset and --seed")
        try: seed = int(opts["seed"])
        except (TypeError, ValueError): raise ValueError("--seed must be an integer")
        if int(opts.get("evaluations", 0)) <= 0 and int(opts.get("timelimit", 2**64 - 1)) >= 2**64 - 1: raise ValueError("require a finite positive --evaluations or --timelimit")
        output = parsed.output.resolve()
        if output.exists(): raise ValueError(f"immutable output already exists: {output}")
        dataset = Path(str(opts["dataset"])).resolve()
        if not dataset.is_file(): raise ValueError(f"dataset is not a file: {dataset}")
        lock = cwd / "flake.lock"; lock_snap = snapshot(lock) if lock.is_file() else None
        binary_snap = snapshot(binary)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)); staged_bin = stage / binary.name; shutil.copyfile(binary, staged_bin); staged_bin.chmod(binary.stat().st_mode | stat.S_IXUSR)
        input_snaps: list[dict[str, Any]] = []; staged_args = list(gp_args)
        for key in PATH_OPTIONS:
            if key not in opts: continue
            original = Path(str(opts[key])).resolve()
            if not original.is_file(): raise ValueError(f"input is not a file: {original}")
            snap = snapshot(original); staged = stage / (key + "-" + original.name); shutil.copyfile(original, staged); staged.chmod(stat.S_IRUSR | stat.S_IRGRP)
            input_snaps.append({"option": "--" + key, **snap, "staged_path": str(staged)})
            for j, token in enumerate(staged_args):
                if token == str(original) or (j and staged_args[j-1] == "--" + key): staged_args[j] = str(staged)
                elif token == "--" + key + "=" + str(original): staged_args[j] = "--" + key + "=" + str(staged)
        # Version is obtained from exactly the immutable staged executable, bounded.
        version_out, version_err, version_rc = _run_bounded([str(staged_bin), "--version"], cwd=stage, timeout=min(parsed.timeout, 30.0))
        if version_rc != 0: raise ValueError(f"operon_gp --version failed: {version_err.strip()}")
        report_path = stage / "machine-report.json"
        proc_out, proc_err, rc = _run_bounded([str(staged_bin), *staged_args, "--report-json", str(report_path)], cwd=cwd, timeout=parsed.timeout)
        if rc != 0: raise RuntimeError(f"operon_gp exited with status {rc}: {proc_err.strip()}")
        if not report_path.is_file(): raise RuntimeError("operon_gp did not produce --report-json")
        report = validate_report(_json_strict(report_path))
        if report["seed"] != seed: raise ValueError("machine report seed does not match requested seed")
        for item in input_snaps:
            now = snapshot(Path(item["path"]))
            if now["sha256"] != item["sha256"] or now["size"] != item["size"]: raise RuntimeError(f"input mutated during run: {item['path']}")
        if lock_snap and (sha256(lock) != lock_snap["sha256"] or lock.stat().st_size != lock_snap["size"]): raise RuntimeError("dependency lock mutated during run")
        output.parent.mkdir(parents=True, exist_ok=True)
        executable = snapshot(binary)
        inputs = sorted(({k: v for k, v in item.items() if k != "staged_path"} for item in input_snaps), key=lambda x: x["option"])
        manifest = {"$schema": SCHEMA + "/manifest", "schema_version": 2, "replay_comparison": REPLAY_COMPARISON, "invocation": {"argv": [str(binary), *canonical], "machine_report_option": "--report-json <private-staging-path>"}, "inputs": inputs, "effective_configuration": opts, "implementation": {"name": "Operon", "version_output": version_out, "executable": executable}, "dependency_lock": lock_snap, "compiler": None, "cpu": cpu_identity(), "resource_budget": {"wrapper_timeout_seconds": parsed.timeout, "evaluations": opts.get("evaluations"), "timelimit": opts.get("timelimit"), "status": "completed"}, "seed": seed, "result_schema": SCHEMA + "/results"}
        metrics = report["metrics"]
        results = {"$schema": SCHEMA + "/results", "schema_version": 2, "srbench_compatibility": False, "dataset": dataset.stem, "algorithm": "Operon-GP", "params": {"canonical_argv": canonical}, "random_state": seed, "time_time": metrics["elapsed_seconds"], "symbolic_model": report["symbolic_model"], "tree_node_count": report["tree_node_count"], "metrics": metrics, "operon": {"manifest": "manifest.json"}}
        (stage / "stdout.txt").write_text(proc_out); (stage / "stderr.txt").write_text(proc_err)
        for payload, name in ((manifest, "manifest.json"), (results, "results.json")):
            with (stage / name).open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
        os.replace(stage, output); print(output / "results.json"); return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError, OverflowError) as e:
        if 'stage' in locals(): shutil.rmtree(stage, ignore_errors=True)
        print(f"error: {e}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
