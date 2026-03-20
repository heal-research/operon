#!/usr/bin/env python3
"""Benchmark Operon GP/NSGP with and without the transposition cache.

Sweeps over:
  - dataset    : one or more problems (loaded from JSON metadata)
  - algorithm  : gp, nsgp
  - iterations : 1, 10, 50, 100  (local optimisation iterations)
  - cache      : on, off
  - seed       : --seed-start … --seed-start + --seeds - 1

For each combination the corresponding CLI binary is invoked and the
final-generation row is captured and written to a CSV file.

Dataset JSON files next to each CSV are used to resolve --target and
--train/--test splits automatically.  Extra arguments after -- are
forwarded verbatim to every CLI call (e.g. --threads, --generations).

--label tags every row with a machine/run identifier, useful when
distributing experiments across machines and merging results later.

--estimate runs one seed per unique (problem, algorithm, iterations, cache)
combo, prints a per-combo timing table, and reports how many seeds fit in a
given time budget — useful for planning overnight runs.

Usage:
    python tools/benchmark_cache.py [benchmark options] -- <operon CLI options>

Examples:
    # Estimate timing before committing (8-hour overnight budget)
    python tools/benchmark_cache.py --estimate --budget 8h -- \\
        --threads 4 --generations 150

    # All datasets in data/, default settings
    python tools/benchmark_cache.py -- --threads 16 --generations 150

    # Specific datasets only
    python tools/benchmark_cache.py \\
        --problems Chemical-I Chemical-II Nikuradse_1 -- \\
        --threads 16 --generations 150

    # Distribute across two machines (same seeds, tagged for comparison):
    #   machine A (zen3):  --label zen3 --seeds 100
    #   machine B (zen2):  --label zen2 --seeds 100
    # Or split seeds to halve wall time:
    #   machine A: --label zen3 --seed-start  0 --seeds 50
    #   machine B: --label zen2 --seed-start 50 --seeds 50
    # Merge afterwards:
    #   { head -1 zen3.csv; tail -n +2 zen3.csv; tail -n +2 zen2.csv; } > combined.csv
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build" / "cli"
DATA_DIR  = REPO_ROOT / "data"

# Columns emitted by the CLI (whitespace-separated, last data line before expression)
CLI_COLUMNS = [
    "iteration", "r2_tr", "r2_te", "mae_tr", "mae_te",
    "nmse_tr", "nmse_te", "best_fit", "avg_fit",
    "best_len", "avg_len", "eval_cnt", "res_eval",
    "jac_eval", "opt_time", "seed", "elapsed",
]
META_COLUMNS = ["label", "problem", "algorithm", "ls_iterations", "cache"]
ALL_COLUMNS  = META_COLUMNS + CLI_COLUMNS


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m"


def parse_budget(s: str) -> float:
    """Parse a duration string like '8h', '90m', '3600s' into seconds."""
    s = s.strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    raise argparse.ArgumentTypeError(f"Invalid duration '{s}': use e.g. 8h, 90m, 3600s")


def repair_csv(path: Path) -> bool:
    """Truncate any incomplete final row (no trailing newline). Returns True if repaired."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        content = f.read()
    if content.endswith(b"\n"):
        return False
    last_newline = content.rfind(b"\n")
    if last_newline < 0:
        return False
    with path.open("r+b") as f:
        f.truncate(last_newline + 1)
    return True


def load_problem(name: str) -> dict:
    """Return {name, csv_path, target, train, test} from the JSON sidecar."""
    json_path = DATA_DIR / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No metadata file: {json_path}")
    meta = json.loads(json_path.read_text())["metadata"]
    tr = meta["training_rows"]
    te = meta["test_rows"]
    return {
        "name":   name,
        "csv":    str(DATA_DIR / meta["filename"]),
        "target": meta["target"],
        "train":  f"{tr['start']}:{tr['end']}",
        "test":   f"{te['start']}:{te['end']}",
    }


def discover_problems() -> list[str]:
    """Return names of all problems that have both a CSV and a JSON file."""
    return sorted(
        p.stem for p in DATA_DIR.glob("*.json")
        if (DATA_DIR / p.stem).with_suffix(".csv").exists()
    )


def run_one(
    problem: dict,
    algorithm: str,
    iterations: int,
    cache: bool,
    seed: int,
    extra_args: list[str],
    label: str,
) -> tuple[dict[str, str] | None, float]:
    """Run one experiment; returns (row_or_None, wall_seconds)."""
    binary = BUILD_DIR / f"operon_{algorithm}"
    cmd = [
        str(binary),
        "--dataset",    problem["csv"],
        "--target",     problem["target"],
        "--train",      problem["train"],
        "--test",       problem["test"],
        "--iterations", str(iterations),
        "--seed",       str(seed),
        *extra_args,
    ]
    if cache:
        cmd.append("--transposition-cache")

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0

    if result.returncode != 0:
        lbl = f"{problem['name']} {algorithm} iter={iterations} cache={'on' if cache else 'off'} seed={seed}"
        print(f"\n  ERROR [{lbl}]: {result.stderr.strip()}", file=sys.stderr)
        return None, wall

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        print(f"\n  UNEXPECTED OUTPUT (seed={seed})", file=sys.stderr)
        return None, wall

    tokens = lines[-2].split()
    if len(tokens) != len(CLI_COLUMNS):
        print(
            f"\n  COLUMN MISMATCH (seed={seed}): expected {len(CLI_COLUMNS)}, got {len(tokens)}",
            file=sys.stderr,
        )
        return None, wall

    row: dict[str, str] = dict(zip(CLI_COLUMNS, tokens))
    row["label"]         = label
    row["problem"]       = problem["name"]
    row["algorithm"]     = algorithm
    row["ls_iterations"] = str(iterations)
    row["cache"]         = "yes" if cache else "no"
    return row, wall


def run_estimate(
    problems: list[dict],
    algorithms: list[str],
    ls_iterations: list[int],
    extra_args: list[str],
    jobs: int,
    budget_seconds: float | None,
) -> None:
    """Run one seed per combo, print a timing table, and report overnight feasibility."""
    combos = list(product(problems, algorithms, ls_iterations, [False, True]))
    print(f"Running {len(combos)} timing probes (1 seed each, {jobs} parallel) …\n")

    timings: dict[tuple, float] = {}
    futures_map = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for prob, alg, iters, cache in combos:
            fut = executor.submit(run_one, prob, alg, iters, cache, 0, extra_args, "")
            futures_map[fut] = (prob["name"], alg, iters, cache)
        done = 0
        for future in as_completed(futures_map):
            key = futures_map[future]
            _, wall = future.result()
            timings[key] = wall
            done += 1
            print(f"  [{done}/{len(combos)}] {key[0]} {key[1]} iter={key[2]} cache={'on' if key[3] else 'off'}  {wall:.1f}s",
                  end="\r", flush=True)
    print()

    # Print table
    col_w = max(len(p["name"]) for p in problems)
    header = f"  {'problem':<{col_w}}  {'alg':<5}  {'iter':>4}  {'cache':>5}  {'sec/seed':>8}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    total_per_seed = 0.0
    for prob, alg, iters, cache in sorted(combos, key=lambda c: (c[0]["name"], c[1], c[2], c[3])):
        t = timings.get((prob["name"], alg, iters, cache), float("nan"))
        total_per_seed += t
        print(f"  {prob['name']:<{col_w}}  {alg:<5}  {iters:>4}  {'yes' if cache else 'no':>5}  {t:>8.1f}")

    # Account for parallelism: wall time per seed ≈ total_per_seed / jobs
    wall_per_seed = total_per_seed / jobs
    print(f"\n  Total serial time per seed : {fmt_duration(total_per_seed)}")
    print(f"  Wall time per seed ({jobs} jobs): {fmt_duration(wall_per_seed)}")

    if budget_seconds is not None:
        feasible = int(budget_seconds / wall_per_seed)
        print(f"\n  Budget: {fmt_duration(budget_seconds)}")
        print(f"  → ~{feasible} seeds feasible  "
              f"(total wall time ≈ {fmt_duration(feasible * wall_per_seed)})")
        print(f"  → for 100 seeds you need ≈ {fmt_duration(100 * wall_per_seed)}")
    else:
        print(f"\n  For 100 seeds: ≈ {fmt_duration(100 * wall_per_seed)}  "
              f"(pass --budget 8h to check overnight feasibility)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", default="cache_benchmark.csv",
        help="Output CSV path (default: cache_benchmark.csv)",
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Number of random seeds / repetitions (default: 100)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed value; seeds run from seed-start to seed-start+seeds-1 (default: 0)",
    )
    parser.add_argument(
        "--label", default="",
        help="Tag added to every row, e.g. machine name (default: empty)",
    )
    parser.add_argument(
        "--problems", nargs="+", metavar="NAME",
        help="Problem names to include (default: all with JSON+CSV in data/)",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=["gp", "nsgp"], metavar="ALG",
        help="Algorithms to benchmark (default: gp nsgp)",
    )
    parser.add_argument(
        "--ls-iterations", nargs="+", type=int, default=[1, 10, 50, 100],
        metavar="N",
        help="Local-search iteration counts to sweep (default: 1 10 50 100)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel benchmark processes (default: 1)",
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Run one seed per combo, print timing table and exit (no CSV written)",
    )
    parser.add_argument(
        "--budget", type=parse_budget, metavar="DURATION", default=None,
        help="Time budget for --estimate feasibility report, e.g. 8h, 90m (default: none)",
    )
    parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="Options forwarded to every CLI invocation (after --)",
    )
    args = parser.parse_args()

    extra_args: list[str] = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    problem_names = args.problems or discover_problems()
    try:
        problems = [load_problem(n) for n in problem_names]
    except FileNotFoundError as e:
        parser.error(str(e))

    if args.estimate:
        run_estimate(problems, args.algorithms, args.ls_iterations,
                     extra_args, args.jobs, args.budget)
        return

    seeds  = range(args.seed_start, args.seed_start + args.seeds)
    combos = list(product(problems, args.algorithms, args.ls_iterations, [False, True], seeds))

    out_path = Path(args.output)

    # Resume: skip combos already present in the output file
    if repair_csv(out_path):
        print(f"Repaired incomplete final row in {out_path}.")
    completed: set[tuple] = set()
    if out_path.exists():
        with out_path.open(newline="") as f:
            for row in csv.DictReader(f):
                completed.add((
                    row["label"], row["problem"], row["algorithm"],
                    row["ls_iterations"], row["cache"], row["seed"],
                ))
        if completed:
            print(f"Resuming: {len(completed)} experiments already done, skipping.")

    def is_done(prob: dict, alg: str, iters: int, cache: bool, seed: int) -> bool:
        return (args.label, prob["name"], alg, str(iters),
                "yes" if cache else "no", str(seed)) in completed

    combos = [c for c in combos if not is_done(*c)]
    total  = len(combos)
    done   = 0
    start  = time.monotonic()

    print(f"Label     : {args.label or '(none)'}")
    print(f"Problems  : {', '.join(p['name'] for p in problems)}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Iterations: {args.ls_iterations}")
    print(f"Seeds     : {args.seed_start} … {args.seed_start + args.seeds - 1}  ({args.seeds} total)")
    print(f"Jobs      : {args.jobs}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    if not combos:
        print("\nAll experiments already complete.")
        return
    print(f"\nRunning {total} experiments → {out_path}\n")

    w = len(str(total))
    interrupted = False
    with out_path.open("a", newline="") as f:
        # Write header only for new files
        if not completed:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
        else:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)

        futures_map = {}
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        try:
            for prob, alg, iters, cache, seed in combos:
                fut = executor.submit(run_one, prob, alg, iters, cache, seed, extra_args, args.label)
                futures_map[fut] = (prob["name"], alg, iters, cache, seed)

            for future in as_completed(futures_map):
                prob_name, alg, iters, cache, seed = futures_map[future]
                done += 1
                row, _ = future.result()
                if row:
                    writer.writerow(row)
                    f.flush()

                elapsed = time.monotonic() - start
                rate    = done / elapsed
                eta     = (total - done) / rate if rate > 0 else 0
                pct     = 100 * done / total
                lbl     = f"{prob_name} {alg} iter={iters} cache={'on' if cache else 'off'} seed={seed}"
                print(
                    f"  [{done:>{w}}/{total}] {pct:5.1f}%  "
                    f"ETA {fmt_duration(eta):>8}  elapsed {fmt_duration(elapsed):>8}  {lbl}",
                    end="\r", flush=True,
                )
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\n\nInterrupted — cancelling pending experiments …", flush=True)
            executor.shutdown(wait=False, cancel_futures=True)

    repair_csv(out_path)
    total_done = len(completed) + done
    status = "Interrupted" if interrupted else "Done"
    print(f"\n{status} in {fmt_duration(time.monotonic() - start)}. "
          f"{total_done} experiments saved to {out_path}")


if __name__ == "__main__":
    main()
