#!/usr/bin/env python3
"""Engine-native stream canaries, determinism, kill safety and interleaved overhead.

POSIX only: the harness uses `resource` (child CPU time), SIGKILL (kill-safety scenario) and POSIX process-exit
semantics (`-signal.SIGKILL`). On any other platform it prints a SKIP line and exits with status 77 (the
conventional "skipped" code) before importing anything POSIX-specific. It is a manual harness, not registered in
CTest: the benchmark and kill scenarios are timing dependent.

Run with --gp/--nsgp paths, --data pointing at a directory holding I_6_20-without_noise-rep00.csv and
I_6_20-shape.json, and --out a fresh scratch directory. --benchmark adds seven interleaved pairs per cell.
Every generation record and every final document is checked against docs/schemas/generation.schema.json and
final.schema.json with the stdlib validator in schema_lite.py. A different validator can be run over the kept
output directories (OUT/<cell>/generation-*.jsonl, OUT/<cell>/final.json) afterwards.
"""
import os
import sys

if os.name != "posix":
    print(f"SKIP generation_json_smoke.py: requires POSIX (resource, SIGKILL); running on os.name={os.name!r}")
    sys.exit(77)

import argparse
import copy
import gzip
import hashlib
import json
import resource
import signal
import statistics
import subprocess
import time
from pathlib import Path

import schema_lite

RUN_ID = "a" * 64
ATTEMPT_ID = "11111111-2222-4333-8444-555555555555"


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def records(directory):
    return [json.loads(line) for p in sorted(directory.glob("generation-*.jsonl")) for line in p.read_text().splitlines()]


def scientific(rows):
    rows = copy.deepcopy(rows)
    for r in rows:
        r.pop("time")
        for block in r["counters"].values():
            block.pop("optimizer_seconds")
    return rows


def check_stream(directory):
    rows = records(directory)
    final = load(directory / "final.json")
    for schema, documents in (("generation.schema.json", rows), ("final.schema.json", [final])):
        for document in documents:
            errors = schema_lite.validate(document, schema)
            assert not errors, (directory, schema, errors[:5])
    assert final["instrumentation_errors"] == 0
    assert final["record_count"] == len(rows) == final["generations_completed"] + 1
    assert [r["seq"] for r in rows] == list(range(len(rows)))
    assert rows[0]["event"] == "initialized"
    assert rows[0]["time"]["process_elapsed_s"] > 0
    for row in rows:
        for objective in row["objectives"]:
            if objective["finite_count"]:
                assert objective["min"] <= objective["mean"] <= objective["max"], objective
            assert sum(objective[k + "_count"] for k in ("finite", "rejected", "nan", "posinf", "neginf")) == row["structure"]["population_count"]
    assert not list(directory.glob("*.part"))
    for s in final["segments"]:
        path = directory / f'generation-{s["segment_id"]:06d}.jsonl'
        data = path.read_bytes()
        part = [json.loads(line) for line in data.splitlines()]
        assert hashlib.sha256(data).hexdigest() == s["sha256"]
        assert s["count"] == len(part)
        assert (s["first_seq"], s["last_seq"]) == (part[0]["seq"], part[-1]["seq"])
    models = [json.loads(line) for line in (directory / "models.jsonl.txt").read_text().splitlines()]
    emitted = [r for r in rows if r["selected_model"]["model_object_emitted"]]
    assert len(models) == len(emitted)
    for m, r in zip(models, emitted):
        assert hashlib.sha256(m["serialization"].encode()).hexdigest() == m["model_ref"] == r["selected_model"]["model_ref"]
    assert final["final_model"]["model_ref"] == rows[-1]["selected_model"]["model_ref"]
    if rows[-1]["nsga2"] is not None:
        front = directory / "front.json.txt"
        if not front.exists():
            front = directory / "pareto-front.json.txt"
        assert hashlib.sha256(front.read_bytes()).hexdigest() == final["front_ref"]
        assert len(load(front)) == rows[-1]["nsga2"]["rank_histogram"][0]
    raw = b"".join(p.read_bytes() for p in sorted(directory.glob("generation-*.jsonl")))
    return {"records": len(rows), "raw_B_per_record": len(raw) / len(rows),
            "gzip6_B_per_record": len(gzip.compress(raw, compresslevel=6, mtime=0)) / len(rows),
            "median_report_s": statistics.median(r["time"]["report_s"] for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gp", type=Path, required=True)
    p.add_argument("--nsgp", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--benchmark", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    base = ["--dataset", str(a.data / "I_6_20-without_noise-rep00.csv"), "--target", "y",
            "--train", "0:100", "--test", "100:200", "--seed", "7", "--objective", "nmse",
            "--population-size", "128", "--pool-size", "128", "--maxlength", "50", "--maxdepth", "20",
            "--creator", "ptc2", "--evaluations", "unlimited", "--generations", "30", "--iterations", "0"]

    def command(binary, directory, extra=(), log=True, threads=1):
        cmd = [str(binary), *base, "--threads", str(threads), *extra, "--run-id", RUN_ID,
               "--attempt-id", ATTEMPT_ID, "--final-json", str(directory / "final.json")]
        if log:
            cmd += ["--generation-log", str(directory)]
        if binary == a.nsgp:
            cmd += ["--pareto-front", str(directory / "front.json.txt")]
        return cmd

    def run(name, binary, extra=(), log=True, threads=1):
        directory = a.out / name
        directory.mkdir()
        cmd = command(binary, directory, extra, log, threads)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter()
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            result = subprocess.run(cmd, stdout=out, stderr=err)
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        assert result.returncode == 0, (cmd, result.returncode, (directory / "stderr.log").read_text())
        (directory / "command.txt").write_text(" ".join(cmd) + "\n")
        return directory, wall, cpu

    report = {"smoke": {}, "determinism": {}, "benchmark": {}}
    cells = {"gp": (a.gp, []), "nsgp": (a.nsgp, []),
             "os": (a.gp, ["--offspring-generator", "os:100:1"]),
             "shape": (a.gp, ["--shape-constraints-config", str(a.data / "I_6_20-shape.json"),
                               "--shape-bound-mode", "interval:2"])}
    for name, (binary, extra) in cells.items():
        directory, _, _ = run(name, binary, extra)
        report["smoke"][name] = check_stream(directory)
        expected = scientific(records(directory))
        for suffix, threads in (("repeat", 1), ("t4", 4)):
            other, _, _ = run(f"{name}-{suffix}", binary, extra, threads=threads)
            check_stream(other)
            assert scientific(records(other)) == expected, f"non-time records differ: {name}-{suffix}"
            report["determinism"][f"{name}-{suffix}"] = "identical"
    for selection in ("mdl", "bic", "aic"):
        d, _, _ = run("nsgp-" + selection, a.nsgp, ["--model-selection", selection])
        report["smoke"]["nsgp-" + selection] = check_stream(d)
    d = a.out / "nsgp-default-front"
    d.mkdir()
    cmd = command(a.nsgp, d)
    index = cmd.index("--pareto-front")
    del cmd[index:index + 2]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert load(d / "final.json")["front_ref"] == hashlib.sha256((d / "pareto-front.json.txt").read_bytes()).hexdigest()
    report["smoke"]["nsgp-default-front"] = check_stream(d)
    for binary in (a.gp, a.nsgp):
        for suffix, extra in (("cache", ["--transposition-cache"]), ("partial", ["--evaluations", "220"])):
            d, _, _ = run(binary.name + "-" + suffix, binary, extra)
            report["smoke"][binary.name + "-" + suffix] = check_stream(d)
            if suffix == "cache":
                assert records(d)[-1]["counters"]["logical"]["cache_hits"] is not None
            else:
                assert load(d / "final.json")["termination"]["primary"] == "evaluation_limit"
        d, _, _ = run(binary.name + "-sync-record", binary, ["--sync", "record", "--segment-generations", "3"])
        check_stream(d)
        directory = a.out / (binary.name + "-killed")
        directory.mkdir()
        cmd = command(binary, directory, ["--generations", "unlimited", "--segment-generations", "3"])
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            child = subprocess.Popen(cmd, stdout=out, stderr=err)
            deadline = time.monotonic() + 20
            try:
                while len(list(directory.glob("generation-*.jsonl"))) < 2:
                    assert child.poll() is None, (cmd, (directory / "stderr.log").read_text())
                    assert time.monotonic() < deadline, "no sealed records before kill deadline"
                    time.sleep(0.002)
                child.send_signal(signal.SIGKILL)
                assert child.wait() == -signal.SIGKILL
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()
        parts = list(directory.glob("*.part"))
        assert len(parts) <= 1
        sealed_rows = records(directory)
        assert sealed_rows
        parsed = len(sealed_rows)
        for part in parts:
            data = part.read_bytes()
            complete = data.split(b"\n")[:-1]
            for line in complete:
                json.loads(line)
            parsed += len(complete)
        report.setdefault("kill", {})[binary.name] = {"sealed_segments": len(list(directory.glob("generation-*.jsonl"))),
                                                       "parts": len(parts), "complete_parseable_records": parsed}
        failure_dir = a.out / (binary.name + "-io-failure")
        failure_dir.mkdir()
        not_a_directory = failure_dir / "not-a-directory"  # a regular file where the stream directory must be
        not_a_directory.write_text("x")
        cmd = command(binary, failure_dir, log=False) + ["--generation-log", str(not_a_directory)]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 1, (cmd, result.returncode, result.stderr)
        failed = load(failure_dir / "final.json")
        assert failed["exit_status"] == "error"
        assert failed["termination"]["primary"] == "engine_error"
        assert failed["instrumentation_errors"] > 0
        report.setdefault("io_failure", {})[binary.name] = {"exit": result.returncode,
            "primary": failed["termination"]["primary"], "instrumentation_errors": failed["instrumentation_errors"]}
    if a.benchmark:
        bench = {"gp1": (a.gp, 1, ["--generations", "200"]),
                 "gp4-lm": (a.gp, 4, ["--generations", "40", "--iterations", "5"]),
                 "os": (a.gp, 1, ["--generations", "40", "--offspring-generator", "os:100:1"]),
                 "nsga2": (a.nsgp, 1, ["--generations", "200"])}
        for name, (binary, threads, extra) in bench.items():
            timings = {False: [], True: []}; reports = []
            for rep in range(7):
                for enabled in ((False, True) if rep % 2 == 0 else (True, False)):
                    d, wall, cpu = run(f"bench-{name}-{rep}-{int(enabled)}", binary,
                                       ["--population-size", "1000", "--pool-size", "1000", *extra], enabled, threads)
                    timings[enabled].append((wall, cpu))
                    if enabled:
                        check_stream(d)
                        reports.extend(r["time"]["report_s"] for r in records(d))
            off = [statistics.median(x[k] for x in timings[False]) for k in (0, 1)]
            on = [statistics.median(x[k] for x in timings[True]) for k in (0, 1)]
            report["benchmark"][name] = {"off_wall_cpu_s": off, "on_wall_cpu_s": on,
                "overhead_wall_cpu_pct": [(on[k] / off[k] - 1) * 100 for k in (0, 1)],
                "median_record_report_s": statistics.median(reports), "runs": timings}
    (a.out / "results.json.txt").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
