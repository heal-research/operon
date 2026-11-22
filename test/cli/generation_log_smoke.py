#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
"""Deterministic CTest smoke test of `--generation-log` for operon_gp and operon_nsgp.

Each program runs a small fixed-seed, single-threaded search on a generated toy dataset with `--segment-generations 3`
(and a segment time limit that cannot trigger), so the stream always splits into the same segments: seven completed
generations plus the initial population are eight records, sealed as 3 + 3 + 2. The test then checks

  * every record against docs/schemas/generation.schema.json and the final document against final.schema.json
    (stdlib validator, schema_lite.py);
  * record identity and ordering (run/attempt ids, dense `seq`, `segment_id`, generation numbers, events);
  * the final document's segment table (ids, ranges, counts, byte SHA-256 of every sealed segment) and `record_count`;
  * the model sidecar models.jsonl.txt (SHA-256 of every serialization, one line per `model_object_emitted`, `model_ref`
    carried forward across unchanged models, agreement with the final model);
  * the NSGA-II front reference (`front_ref`) for the default `DIR/pareto-front.json.txt` and for `--pareto-front`;
  * reproducibility: a second run, and a run without --final-json, give identical records apart from wall-clock fields;
  * argument validation and I/O failure: a regular file where the stream directory belongs, and a directory that
    already holds a stream (which must stay untouched), both end with exit status 1 and an `engine_error` final document.

Platform: standard library only, no POSIX facilities (no signals, `resource` or fsync inspection), so it runs wherever
the CLIs run. Timing fields are never compared: they are the only nondeterministic part of a record.

    python3 generation_log_smoke.py --gp build/cli/operon_gp --nsgp build/cli/operon_nsgp [--out DIR]
"""
import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import schema_lite

RUN_ID = "a" * 64
ATTEMPT_ID = "11111111-2222-4333-8444-555555555555"
POPULATION = 16
GENERATIONS = 7
SEGMENT_GENERATIONS = 3
EXPECTED_SEGMENTS = [(0, 0, 2, 3), (1, 3, 5, 3), (2, 6, 7, 2)]  # (segment_id, first_seq, last_seq, count)


def write_toy(path):
    rows = ["x,y"]
    for i in range(40):
        x = i / 10.0
        rows.append(f"{x},{x * x + 0.3 * x + 1.0}")
    path.write_text("\n".join(rows) + "\n")


def base_args(dataset, extra=()):
    return ["--dataset", str(dataset), "--target", "y", "--train", "0:30", "--test", "30:40",
            "--population-size", str(POPULATION), "--pool-size", str(POPULATION), "--iterations", "0",
            "--objective", "nmse", "--threads", "1", "--maxlength", "12", "--maxdepth", "6", "--creator", "ptc2",
            "--seed", "7", "--generations", str(GENERATIONS), "--evaluations", "unlimited", *extra]


def run(binary, args, ok=True):
    result = subprocess.run([str(binary), *args], text=True, capture_output=True, timeout=120)
    assert result.returncode == (0 if ok else 1), (args, result.returncode, result.stderr)
    return result


def load(path):
    """Parse a JSON document rejecting the non-standard NaN/Infinity constants."""
    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse)


def stream_files(directory):
    return sorted(directory.glob("generation-*.jsonl"))


def records(directory):
    return [json.loads(line, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
            for p in stream_files(directory) for line in p.read_bytes().decode("utf-8").splitlines()]


def scientific(rows):
    """The records without wall-clock measurements, which are the only nondeterministic content."""
    rows = copy.deepcopy(rows)
    for row in rows:
        del row["time"]
        for block in row["counters"].values():
            del block["optimizer_seconds"]
    return rows


def check_stream(directory, final_path, nsga2, front_path=None):
    """Verify one completed stream against its final document; returns the records."""
    rows = records(directory)
    final = load(final_path)
    for document in rows:
        errors = schema_lite.validate(document, "generation.schema.json")
        assert not errors, (directory, errors[:5])
    errors = schema_lite.validate(final, "final.schema.json")
    assert not errors, (final_path, errors[:5])

    # directory contents: sealed segments, the sidecar and (NSGA-II only) the default front; nothing partial
    expected = {f"generation-{s[0]:06d}.jsonl" for s in EXPECTED_SEGMENTS} | {"models.jsonl.txt"}
    if nsga2 and front_path is None:
        expected.add("pareto-front.json.txt")
    assert {p.name for p in directory.iterdir()} == expected, sorted(p.name for p in directory.iterdir())

    # final document
    assert final["exit_status"] == "ok" and final["termination"]["primary"] == "generation_limit", final["termination"]
    assert final["run_id"] == RUN_ID and final["attempt_id"] == ATTEMPT_ID
    assert final["instrumentation_errors"] == 0
    assert final["generations_completed"] == GENERATIONS
    assert final["record_count"] == len(rows) == GENERATIONS + 1
    assert "segments" not in final.get("producer", {}).get("unsupported_fields", [])
    assert "record_count" not in final.get("producer", {}).get("unsupported_fields", [])

    # record identity and ordering
    assert [r["seq"] for r in rows] == list(range(len(rows)))
    assert [r["generation"] for r in rows] == list(range(len(rows)))
    assert [r["event"] for r in rows] == ["initialized"] + ["generation_complete"] * GENERATIONS
    assert [r["segment_id"] for r in rows] == [i // SEGMENT_GENERATIONS for i in range(len(rows))]
    names = ["nmse", "length"] if nsga2 else ["nmse"]
    consumed = -1
    for row in rows:
        assert row["run_id"] == RUN_ID and row["attempt_id"] == ATTEMPT_ID
        assert [o["name"] for o in row["objectives"]] == names
        assert row["structure"]["population_count"] == POPULATION and row["structure"]["pool_count"] == POPULATION
        for objective in row["objectives"]:
            counts = sum(objective[k + "_count"] for k in ("finite", "rejected", "nan", "posinf", "neginf"))
            assert counts == POPULATION, objective
            if objective["finite_count"]:
                assert objective["min"] <= objective["mean"] <= objective["max"], objective
        logical = row["counters"]["logical"]
        assert row["budget"]["consumed"] == logical["residual_total"] + logical["jacobian_total"]
        assert row["budget"]["consumed"] >= consumed  # a counter never moves backwards
        consumed = row["budget"]["consumed"]
        if nsga2:
            assert sum(row["nsga2"]["rank_histogram"]) == POPULATION
            assert row["selected_model"]["selection_rule"] == "mdl"
            assert row["time"]["sort_s"] is not None
        else:
            assert row["nsga2"] is None and row["time"]["sort_s"] is None
            assert row["selected_model"]["selection_rule"] == "obj0"
    assert rows[0]["time"]["process_elapsed_s"] > 0
    final_logical = final["counters"]["logical"]
    last_logical = rows[-1]["counters"]["logical"]
    for key in ("evaluator_calls", "residual_total", "jacobian_total"):
        assert final_logical[key] == last_logical[key], key

    # segments: table, hashes, ranges, counts
    assert [(s["segment_id"], s["first_seq"], s["last_seq"], s["count"]) for s in final["segments"]] == EXPECTED_SEGMENTS
    for s in final["segments"]:
        data = (directory / f'generation-{s["segment_id"]:06d}.jsonl').read_bytes()
        assert data.endswith(b"\n")
        part = [json.loads(line) for line in data.split(b"\n")[:-1]]
        assert hashlib.sha256(data).hexdigest() == s["sha256"]
        assert len(part) == s["count"]
        assert (part[0]["seq"], part[-1]["seq"]) == (s["first_seq"], s["last_seq"])
        assert {r["segment_id"] for r in part} == {s["segment_id"]}

    # model sidecar: one line per newly emitted model, every record refers to the latest one
    sidecar = (directory / "models.jsonl.txt").read_bytes()
    assert sidecar.endswith(b"\n")
    models = [json.loads(line) for line in sidecar.decode("utf-8").split("\n")[:-1]]
    emitted = [r for r in rows if r["selected_model"]["model_object_emitted"]]
    assert rows[0]["selected_model"]["model_object_emitted"], "the first model is always emitted"
    assert len(models) == len(emitted) >= 1
    for model, row in zip(models, emitted):
        assert set(model) == {"model_ref", "serialization"}
        assert hashlib.sha256(model["serialization"].encode("utf-8")).hexdigest() == model["model_ref"]
        assert model["model_ref"] == row["selected_model"]["model_ref"]
    current = None
    for row in rows:
        selected = row["selected_model"]
        if selected["model_object_emitted"]:
            current = selected["model_ref"]
        assert selected["model_ref"] == current, "an unchanged model keeps the last emitted model_ref"
    assert final["final_model"] is not None
    assert final["final_model"]["model_ref"] == rows[-1]["selected_model"]["model_ref"]
    assert final["final_model"]["selection_rule"] == rows[-1]["selected_model"]["selection_rule"]

    # front reference
    if nsga2:
        front = front_path if front_path is not None else directory / "pareto-front.json.txt"
        assert hashlib.sha256(front.read_bytes()).hexdigest() == final["front_ref"]
        members = load(front)
        assert isinstance(members, list) and 1 <= len(members) <= rows[-1]["nsga2"]["rank_histogram"][0]
        assert "front_ref" not in final.get("producer", {}).get("unsupported_fields", [])
    else:
        assert final["front_ref"] is None
    return rows


def check_failure(binary, args, final_path, stderr_fragment=None):
    """A run that cannot open its stream exits with 1 and still leaves an engine_error final document."""
    result = run(binary, args, ok=False)
    assert "error:" in result.stderr, result.stderr
    if stderr_fragment is not None:
        assert stderr_fragment in result.stderr, result.stderr
    final = load(final_path)
    errors = schema_lite.validate(final, "final.schema.json")
    assert not errors, errors[:5]
    assert final["exit_status"] == "error"
    assert final["termination"]["primary"] == "engine_error" and "engine_error" in final["termination"]["reasons"]
    assert final["instrumentation_errors"] > 0
    assert final["record_count"] == 0 and final["segments"] == []
    assert final["generations_completed"] == 0


def snapshot(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gp", required=True, type=Path)
    parser.add_argument("--nsgp", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="keep every output under this directory instead of a temporary one")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = args.out if args.out else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "toy.csv"
        write_toy(dataset)
        ids = ["--run-id", RUN_ID, "--attempt-id", ATTEMPT_ID]
        segmenting = ["--segment-generations", str(SEGMENT_GENERATIONS), "--segment-seconds", "3600"]

        def stream_args(directory, final=None, extra=()):
            cmd = [*base_args(dataset, extra), *segmenting, *ids, "--generation-log", str(directory)]
            return cmd + ["--final-json", str(final)] if final is not None else cmd

        for algorithm, binary, extra in (("gp", args.gp, []), ("nsga2", args.nsgp, ["--model-selection", "mdl"])):
            nsga2 = algorithm == "nsga2"

            # a stream with a final document; the directory is created on demand
            first = root / f"{algorithm}-1"
            first_final = root / f"{algorithm}-1-final.json"
            result = run(binary, stream_args(first, first_final, extra))
            assert "iteration" in result.stdout, result.stdout
            expected = scientific(check_stream(first, first_final, nsga2))
            assert not list(root.glob("*.tmp*")), "the final document is published atomically"

            # same configuration again, with --sync record and without --final-json: identical records
            second = root / f"{algorithm}-2"
            run(binary, stream_args(second, None, [*extra, "--sync", "record"]))
            assert scientific(records(second)) == expected, f"{algorithm}: records differ between runs"
            assert (second / "models.jsonl.txt").read_bytes() == (first / "models.jsonl.txt").read_bytes()
            assert [p.name for p in stream_files(second)] == [p.name for p in stream_files(first)]
            assert not list(second.glob("*.part"))
            if nsga2:
                assert (second / "pareto-front.json.txt").read_bytes() == (first / "pareto-front.json.txt").read_bytes()

            # the front may live anywhere when --pareto-front says so; the stream directory then does not hold it
            if nsga2:
                third = root / "nsga2-3"
                front = root / "nsga2-3-front.json.txt"
                third_final = root / "nsga2-3-final.json"
                run(binary, [*stream_args(third, third_final, extra), "--pareto-front", str(front)])
                assert scientific(check_stream(third, third_final, True, front)) == expected
                assert not (third / "pareto-front.json.txt").exists()

            # argument validation: identifiers are mandatory and nothing is created on refusal
            refused = root / f"{algorithm}-refused"
            result = run(binary, [*base_args(dataset, extra), "--generation-log", str(refused)], ok=False)
            assert "--run-id" in result.stderr, result.stderr
            assert not refused.exists()

            # I/O failure 1: a regular file stands where the stream directory belongs
            blocker = root / f"{algorithm}-blocker"
            blocker.write_text("not a directory")
            failure_final = root / f"{algorithm}-blocked-final.json"
            check_failure(binary, stream_args(blocker, failure_final, extra), failure_final)
            assert blocker.read_text() == "not a directory"

            # I/O failure 2: reusing a directory that already holds a stream must not touch it
            before = snapshot(first)
            reuse_final = root / f"{algorithm}-reuse-final.json"
            check_failure(binary, stream_args(first, reuse_final, extra), reuse_final,
                          "generation log directory already contains a stream")
            assert snapshot(first) == before, "an earlier stream is never modified"

        print("generation_log: PASS (gp and nsga2: schema, ids, segments, hashes, sidecar, front, determinism, I/O failures)")


if __name__ == "__main__":
    main()
