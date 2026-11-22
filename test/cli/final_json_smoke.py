#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
"""Smoke test of `--final-json` for operon_gp and operon_nsgp.

Runs both programs on a toy dataset under every way a run can end (generation limit, evaluation limit,
selection-pressure limit, time limit, SIGTERM, SIGINT, error) and checks the operon.final document each one
leaves behind: conformance to docs/schemas/final.schema.json (stdlib validator, schema_lite.py), internal
consistency, agreement with the evaluator counters printed in the stdout table, atomic publication, exit status.
Standard library only. The SIGTERM/SIGINT scenarios run on POSIX only and are skipped, with a printed note, elsewhere;
every other scenario is platform independent.

    python3 final_json_smoke.py --gp build/cli/operon_gp --nsgp build/cli/operon_nsgp [--keep DIR]
                                [--validator-command "CMD ARGS" [--validator-cwd DIR] [--validator-tolerate CODE]]

With --validator-command every document is additionally run through that external validator command (`CMD ARGS
PATH`); an error finding fails the test unless its code is listed with --validator-tolerate.
"""

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import schema_lite

RUN_ID = "a" * 64
ATTEMPT_ID = "11111111-2222-4333-8444-555555555555"
UNLIMITED = str(1 << 62)
PRECEDENCE = ["engine_error", "signal", "callback", "generator_termination", "evaluation_limit", "pressure_limit", "generation_limit", "time_limit"]

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print(f"  FAIL: {message}")


def write_toy(path):
    rows = ["x,y"]
    for i in range(40):
        x = i / 10.0
        rows.append(f"{x},{x * x + 0.3 * x + 1.0}")
    path.write_text("\n".join(rows) + "\n")


def base_args(dataset, extra):
    return [
        "--dataset", str(dataset), "--target", "y", "--train", "0:30", "--test", "30:40",
        "--population-size", "16", "--pool-size", "16", "--iterations", "1", "--objective", "nmse",
        "--threads", "1", "--maxlength", "12", "--maxdepth", "6", "--creator", "ptc2", "--seed", "7",
    ] + extra


def table_rows(stdout):
    """The per-generation table of the CLI: header, one row per report, the model on the last line."""
    lines = [l.split() for l in stdout.splitlines() if l.strip()]
    if not lines or lines[0][0] != "iteration":
        return None, []
    header = lines[0]
    rows = [dict(zip(header, l)) for l in lines[1:] if len(l) == len(header)]
    return header, rows


def run(program, args, final=None, timeout=120, run_id=RUN_ID, attempt_id=ATTEMPT_ID, ids=True):
    cmd = [str(program)] + args
    if final is not None:
        cmd += ["--final-json", str(final)]
        if ids:
            cmd += ["--run-id", run_id, "--attempt-id", attempt_id]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p


def load(path):
    check(path.exists(), f"{path.name} was written")
    if not path.exists():
        return None
    text = path.read_text()
    check(text.endswith("\n") and text.count("\n") == 1, f"{path.name} is one JSON document and a newline")

    def no_constant(name):
        raise ValueError(f"bare {name}")

    return json.loads(text, parse_constant=no_constant)


def leftovers(directory):
    return [p.name for p in directory.iterdir() if ".tmp-" in p.name]


def verify(doc, label, expect_reasons=None, expect_primary=None, exit_status="ok", table=None, program=None):
    """Checks that hold for every document."""
    if doc is None:
        return
    schema_errors = schema_lite.validate(doc, "final.schema.json")
    check(not schema_errors, f"{label}: docs/schemas/final.schema.json violations {schema_errors[:5]}")
    req = ["schema", "schema_version", "run_id", "attempt_id", "exit_status", "termination", "generations_completed",
           "incomplete_generation", "budgets", "counters", "final_model", "front_ref", "segments", "record_count",
           "instrumentation_errors"]
    check(all(k in doc for k in req), f"{label}: all required keys present")
    check(set(doc) <= set(req) | {"producer"}, f"{label}: no unknown keys")
    check(doc["schema"] == "operon.final" and doc["schema_version"] == "1.2", f"{label}: schema name/version")
    check(doc["run_id"] == RUN_ID and doc["attempt_id"] == ATTEMPT_ID, f"{label}: identifiers are echoed")
    check(doc["exit_status"] == exit_status, f"{label}: exit_status {doc['exit_status']} == {exit_status}")
    term = doc["termination"]
    reasons = term["reasons"]
    check(term["precedence_version"] == 2, f"{label}: precedence_version")
    check(len(set(reasons)) == len(reasons) and reasons, f"{label}: reasons unique and non-empty")
    check(reasons == sorted(reasons, key=PRECEDENCE.index), f"{label}: reasons listed in precedence order")
    check(term["primary"] == reasons[0], f"{label}: primary is the highest-precedence reason")
    if expect_reasons is not None:
        check(reasons == expect_reasons, f"{label}: reasons {reasons} == {expect_reasons}")
    if expect_primary is not None:
        check(term["primary"] == expect_primary, f"{label}: primary {term['primary']} == {expect_primary}")
    check((doc["exit_status"] == "error") == ("engine_error" in reasons), f"{label}: error <=> engine_error")

    c = doc["counters"]["logical"]
    check(doc["counters"]["since_restart"] == c, f"{label}: since_restart == logical")
    check(c["residual_total"] == c["residual_scoring"] + c["residual_optimizer"], f"{label}: residual split sums")
    check(c["jacobian_total"] == c["jacobian_scoring"] + c["jacobian_optimizer"], f"{label}: jacobian split sums")
    ev = doc["budgets"]["evaluations"]
    check(ev["unit"] == "residual+jacobian", f"{label}: budget unit")
    check(ev["consumed"] == c["residual_total"] + c["jacobian_total"], f"{label}: consumed == residual + jacobian")
    expected_over = None if ev["limit"] is None else max(0, ev["consumed"] - ev["limit"])
    check(ev["overshoot"] == expected_over, f"{label}: overshoot")
    check(c["cache_hits"] is None and c["cache_misses"] is None, f"{label}: no cache configured -> cache counters null")
    check(isinstance(c["optimizer_seconds"], (int, float)) and c["optimizer_seconds"] >= 0, f"{label}: optimizer_seconds")
    check(doc["record_count"] == 0 and doc["segments"] == [] and doc["instrumentation_errors"] == 0,
          f"{label}: no generation stream is claimed")
    unsupported = doc.get("producer", {}).get("unsupported_fields", [])
    check("segments" in unsupported and "record_count" in unsupported, f"{label}: stream fields declared unsupported")
    inc = doc["incomplete_generation"]
    if inc is not None:
        check(inc["completed_offspring"] < inc["pool_count"], f"{label}: incomplete generation has an unfilled pool")
        check(inc["attempted_offspring"] >= inc["completed_offspring"], f"{label}: attempted >= completed")
    b = doc["budgets"]
    check(b["time"]["elapsed_s"] >= 0, f"{label}: elapsed_s")
    if "generation_limit" in reasons:
        check(doc["generations_completed"] == b["generations"]["limit"], f"{label}: generation_limit => limit reached")
        check(inc is None, f"{label}: generation_limit stops at a boundary")
    if "evaluation_limit" in reasons:
        check(ev["limit"] is not None and ev["consumed"] >= ev["limit"], f"{label}: evaluation_limit => consumed >= limit")
    if "pressure_limit" in reasons:
        p = b["pressure"]
        check(p["max"] is not None and p["last"] is not None and p["last"] > p["max"], f"{label}: pressure last > max")
    if "time_limit" in reasons:
        check(b["time"]["limit_s"] is not None and b["time"]["elapsed_s"] > b["time"]["limit_s"], f"{label}: elapsed > limit")
    m = doc["final_model"]
    if m is not None:
        check(len(m["model_ref"]) == 64 and all(ch in "0123456789abcdef" for ch in m["model_ref"]), f"{label}: model_ref")
        check(m["selection_rule"] in ("obj0", "mdl", "bic", "aic"), f"{label}: selection rule")
        check(len(m["infix"]) > 0, f"{label}: infix")
    # the table's last row was printed from the same evaluator: for a run without an unfinished generation the final
    # counters equal it
    if table and inc is None and program == "gp":
        last = table[-1]
        check(int(last["eval_cnt"]) == c["evaluator_calls"], f"{label}: evaluator_calls == last table eval_cnt")
        check(int(last["res_eval"]) == c["residual_total"], f"{label}: residual_total == last table res_eval")
        check(int(last["jac_eval"]) == c["jacobian_total"], f"{label}: jacobian_total == last table jac_eval")
        check(int(last["iteration"]) == doc["generations_completed"], f"{label}: generations_completed == last iteration")


def validate(doc_path, label, opts):
    """Run the optional external validator command over one document."""
    if not opts.validator_command:
        return
    p = subprocess.run([*shlex.split(opts.validator_command), str(doc_path)], cwd=opts.validator_cwd,
                       capture_output=True, text=True)
    findings = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    errors = [f for f in findings if f.get("severity") == "error" or f.get("level") == "error"]
    codes = sorted({f.get("code") for f in errors})
    unexpected = [c for c in codes if c not in opts.validator_tolerate]
    check(not unexpected, f"{label}: external validator errors {unexpected} (tolerated: {opts.validator_tolerate})")
    print(f"  external validator {label}: errors={codes or 'none'}")


def scenario_runs(program_name, binary, workdir, dataset, opts):
    def final_path(name):
        return workdir / f"{program_name}-{name}.json"

    common = base_args(dataset, [])
    unlimited = ["--evaluations", UNLIMITED]

    def go(name, extra, expect_reasons=None, expect_primary=None, exit_code=0, exit_status="ok", extra_check=None):
        print(f"[{program_name}] {name}")
        path = final_path(name)
        p = run(binary, base_args(dataset, extra), final=path)
        check(p.returncode == exit_code, f"{program_name}/{name}: exit status {p.returncode} == {exit_code}\n{p.stderr}")
        doc = load(path)
        _, rows = table_rows(p.stdout)
        verify(doc, f"{program_name}/{name}", expect_reasons, expect_primary, exit_status, rows, program_name)
        check(not leftovers(workdir), f"{program_name}/{name}: no temporary files left")
        if extra_check and doc is not None:
            extra_check(doc, p)
        validate(path, f"{program_name}/{name}", opts)
        return doc

    def gen_limit(doc, p):
        check(doc["generations_completed"] == 3, "generation limit: 3 generations")
        check(doc["budgets"]["evaluations"]["limit"] is None and doc["budgets"]["evaluations"]["overshoot"] is None,
              "generation limit: unlimited evaluations reported as null")
        check(doc["incomplete_generation"] is None, "generation limit: nothing incomplete")
        check(doc["budgets"]["pressure"] == {"max": None, "last": None}, "basic generator: no pressure budget")
        check(doc["final_model"] is not None, "generation limit: a final model")
        model_line = p.stdout.strip().splitlines()[-1]
        check(doc["final_model"]["infix"] == model_line, "final_model.infix is the printed model")

    go("generation-limit", ["--generations", "3"] + unlimited, ["generation_limit"], "generation_limit", extra_check=gen_limit)

    def eval_limit(doc, p):
        ev = doc["budgets"]["evaluations"]
        check(ev["limit"] == 100 and ev["consumed"] >= 100, "evaluation limit: consumed >= 100")

    doc = go("evaluation-limit", ["--generations", "100", "--evaluations", "100"], None, "evaluation_limit",
             extra_check=eval_limit)
    if doc:
        check("evaluation_limit" in doc["termination"]["reasons"], "evaluation limit listed")

    def press_zero(doc, p):
        pr = doc["budgets"]["pressure"]
        check(pr["max"] == 0 and pr["last"] is not None and pr["last"] > 0, "pressure limit: max 0, last > 0")

    go("pressure-limit-at-start", ["--generations", "100", "--offspring-generator", "os:0"] + unlimited, None,
       "pressure_limit", extra_check=press_zero)

    def press_mid(doc, p):
        pr = doc["budgets"]["pressure"]
        check(pr["max"] == 1, "pressure limit: max 1")
        check(doc["incomplete_generation"] is not None, "pressure limit: stopped inside a generation")

    go("pressure-limit-mid-generation", ["--generations", "100", "--offspring-generator", "os:1:1000"] + unlimited, None,
       "pressure_limit", extra_check=press_mid)

    def time_limit(doc, p):
        check(doc["budgets"]["time"]["limit_s"] == 0, "time limit: limit_s 0")

    go("time-limit", ["--generations", "100000000", "--timelimit", "0", "--population-size", "200", "--pool-size", "200"]
       + unlimited, None, "time_limit", extra_check=time_limit)

    # shape of the two evaluation-limit variants: also with the transposition cache, whose counters become non-null
    print(f"[{program_name}] transposition cache")
    path = final_path("cache")
    p = run(binary, base_args(dataset, ["--generations", "5", "--transposition-cache"] + unlimited), final=path)
    check(p.returncode == 0, f"{program_name}/cache: exit status {p.returncode}")
    doc = load(path)
    if doc:
        c = doc["counters"]["logical"]
        check(c["cache_hits"] is not None and c["cache_misses"] is not None, "cache configured -> cache counters present")
        check(c["cache_hits"] + c["cache_misses"] > 0, "cache lookups were counted")
        check(doc["counters"]["since_restart"] == c, "cache: since_restart == logical")

    # errors
    print(f"[{program_name}] error: unknown objective")
    path = final_path("error-objective")
    args = base_args(dataset, ["--generations", "3", "--objective", "bogus"] + unlimited)
    p = run(binary, args, final=path)
    check(p.returncode == 1, f"{program_name}/error-objective: exit status 1 (got {p.returncode})")
    doc = load(path)
    verify(doc, f"{program_name}/error-objective", ["engine_error"], "engine_error", "error")
    if doc:
        check(doc["final_model"] is None, "error before a model exists: final_model null")
        check(doc["counters"]["logical"]["evaluator_calls"] == 0, "error before evaluation: zero counters")
        check(doc["budgets"]["generations"]["limit"] == 3, "error record keeps the configured limits")
    validate(path, f"{program_name}/error-objective", opts)

    print(f"[{program_name}] error: missing dataset file")
    path = final_path("error-dataset")
    p = run(binary, ["--dataset", str(workdir / "nope.csv"), "--target", "y", "--seed", "1"], final=path)
    check(p.returncode != 0, f"{program_name}/error-dataset: non-zero exit status")
    doc = load(path)
    verify(doc, f"{program_name}/error-dataset", ["engine_error"], "engine_error", "error")

    print(f"[{program_name}] options")
    path = final_path("no-ids")
    p = run(binary, base_args(dataset, ["--generations", "2"]), final=path, ids=False)
    check(p.returncode == 1 and not path.exists(), f"{program_name}/no-ids: refused before running, nothing written")
    check("--run-id" in p.stderr, f"{program_name}/no-ids: message names the options")
    path = final_path("bad-id")
    p = run(binary, base_args(dataset, ["--generations", "2"]), final=path, run_id="XYZ")
    check(p.returncode == 1 and not path.exists(), f"{program_name}/bad-id: refused, nothing written")
    p = subprocess.run([str(binary)] + base_args(dataset, ["--generations", "2"] + unlimited), capture_output=True, text=True)
    check(p.returncode == 0, f"{program_name}: without --final-json nothing changes")

    # signals (POSIX only: Windows has no SIGTERM/SIGINT delivery to a child via Popen.send_signal)
    posix_signals = ((signal.SIGTERM, "sigterm", 143), (signal.SIGINT, "sigint", 130)) if os.name == "posix" else ()
    if not posix_signals:
        print(f"[{program_name}] signal scenarios skipped: POSIX only")
    for sig, name, code in posix_signals:
        print(f"[{program_name}] {name} mid-run")
        path = final_path(name)
        cmd = [str(binary)] + base_args(dataset, ["--generations", "100000000", "--population-size", "100", "--pool-size",
                                                  "100"] + unlimited) + ["--final-json", str(path), "--run-id", RUN_ID,
                                                                         "--attempt-id", ATTEMPT_ID]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        seen = 0
        for line in proc.stdout:  # wait until the run is demonstrably iterating
            seen += 1
            if seen == 5:
                break
        proc.send_signal(sig)
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            check(False, f"{program_name}/{name}: did not stop within 60 s")
        check(proc.returncode == code, f"{program_name}/{name}: exit status {proc.returncode} == {code}")
        doc = load(path)
        verify(doc, f"{program_name}/{name}", ["signal"], "signal", "ok")
        if doc:
            check(doc["generations_completed"] >= 3, f"{program_name}/{name}: generations were completed before the signal")
            check(doc["final_model"] is not None, f"{program_name}/{name}: a final model after a signal")
        validate(path, f"{program_name}/{name}", opts)


def scenario_nsgp_front(binary, workdir, dataset, opts):
    print("[nsgp] pareto front reference")
    path = workdir / "nsgp-front.json"
    front = workdir / "front.json"
    p = run(binary, base_args(dataset, ["--generations", "3", "--evaluations", UNLIMITED, "--pareto-front", str(front),
                                        "--model-selection", "bic"]), final=path)
    check(p.returncode == 0, f"nsgp/front: exit status {p.returncode}\n{p.stderr}")
    doc = load(path)
    verify(doc, "nsgp/front", ["generation_limit"], "generation_limit")
    if doc:
        check(doc["front_ref"] == hashlib.sha256(front.read_bytes()).hexdigest(), "front_ref == sha256(front file)")
        check(doc["final_model"]["selection_rule"] == "bic", "selection rule is --model-selection")
        check("front_ref" not in doc.get("producer", {}).get("unsupported_fields", []), "front_ref supported when written")
    path2 = workdir / "nsgp-nofront.json"
    p = run(binary, base_args(dataset, ["--generations", "2", "--evaluations", UNLIMITED]), final=path2)
    doc = load(path2)
    if doc:
        check(doc["front_ref"] is None, "no front file: front_ref null")
        check("front_ref" in doc["producer"]["unsupported_fields"], "no front file: front_ref declared unsupported")


def scenario_nonfinite(binary, program_name, workdir, opts):
    """A final model whose metrics are not finite is an outcome: the run succeeds and the record is written."""
    print(f"[{program_name}] numerically failed final model")
    data = workdir / "nonfinite.csv"
    rows = ["x,y"] + [f"{i / 10 + 0.1},{(i / 10 + 0.1) ** 2}" for i in range(30)] + [f"{1 + i / 10},nan" for i in range(10)]
    data.write_text("\n".join(rows) + "\n")
    path = workdir / f"{program_name}-nonfinite.json"
    p = run(binary, base_args(data, ["--generations", "5", "--evaluations", UNLIMITED, "--population-size", "30",
                                     "--pool-size", "30"]), final=path)
    check(p.returncode == 0, f"{program_name}/nonfinite: exit status {p.returncode}\n{p.stderr}")
    _, rows_out = table_rows(p.stdout)
    nonfinite = bool(rows_out) and rows_out[-1]["mae_te"] in ("nan", "-nan", "inf", "-inf")
    check(nonfinite, f"{program_name}/nonfinite: the table shows a non-finite test MAE (scenario is effective)")
    doc = load(path)
    verify(doc, f"{program_name}/nonfinite", ["generation_limit"], "generation_limit")
    if doc:
        check(doc["final_model"] is not None, f"{program_name}/nonfinite: the final model is recorded")
    validate(path, f"{program_name}/nonfinite", opts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp", required=True, type=Path)
    ap.add_argument("--nsgp", required=True, type=Path)
    ap.add_argument("--keep", type=Path, help="write the documents here instead of a temporary directory")
    ap.add_argument("--validator-command", help="external validator command, run as `COMMAND PATH` for every document; "
                    "it prints one JSON finding per line with `code` and `severity` (or `level`)")
    ap.add_argument("--validator-cwd", help="working directory of the external validator command")
    ap.add_argument("--validator-tolerate", action="append", default=[], metavar="CODE",
                    help="finding code the external validator may report as an error (repeatable)")
    opts = ap.parse_args()

    ctx = tempfile.TemporaryDirectory() if opts.keep is None else None
    workdir = Path(ctx.name) if ctx else opts.keep
    workdir.mkdir(parents=True, exist_ok=True)
    dataset = workdir / "toy.csv"
    write_toy(dataset)

    t0 = time.time()
    scenario_runs("gp", opts.gp, workdir, dataset, opts)
    scenario_runs("nsgp", opts.nsgp, workdir, dataset, opts)
    scenario_nsgp_front(opts.nsgp, workdir, dataset, opts)
    scenario_nonfinite(opts.gp, "gp", workdir, opts)
    scenario_nonfinite(opts.nsgp, "nsgp", workdir, opts)

    print(f"\n{checks} checks, {len(failures)} failed, {time.time() - t0:.1f} s")
    for f in failures:
        print("FAILED:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
