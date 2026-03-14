#!/usr/bin/env python3
"""
Compare operon_gp / operon_nsgp outputs between two build trees.

Usage:
    python compare_operon.py <ref-operon-root> <new-operon-root> [options]

Two test modes:
  Determinism  – identical seed → byte-identical model string on the last line
  Statistics   – across many seeds, distributions of fitness metrics must be
                 statistically equivalent (Mann-Whitney U when scipy is available)

Examples:
    # Compare main branch vs refactor branch (determinism + stats)
    python compare_operon.py ../operon .

    # Quick determinism-only run, fewer datasets
    python compare_operon.py ../operon . --only-determinism --datasets Poly-10,Sextic

    # Statistical tests only, more seeds, custom significance level
    python compare_operon.py ../operon . --only-stats --stat-seeds 30 --alpha 0.05
"""

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _strip_color(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)

# ── Dataset registry ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Dataset:
    filename: str
    target:   str
    train:    str
    test:     str

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".csv")


ALL_DATASETS: list[Dataset] = [
    Dataset("Poly-10.csv",         "Y", "0:400",  "400:500"),
    Dataset("Pagie-1.csv",         "F", "0:1340", "1340:1676"),
    Dataset("Sextic.csv",          "Y", "0:4000", "4000:5000"),
    Dataset("Concrete.csv",        "Y", "0:824",  "824:1030"),
    Dataset("Vladislavleva-4.csv", "Y", "0:4819", "4819:6024"),
]

SYMBOL_SETS: list[Optional[str]] = [
    None,                               # default (arithmetic primitives)
    "add,sub,mul,div,square,sqrt",
    "add,sub,mul,div,exp,log,sin,cos",
]

CREATORS = ["btc"]           # "grow" excluded by default — it can hang with restricted symbol sets

OBJECTIVES: list[str] = ["r2", "mse", "mae"]

# Column order emitted by both binaries
_STAT_COLS = [
    "iteration", "r2_tr", "r2_te", "mae_tr", "mae_te",
    "nmse_tr",   "nmse_te", "best_fit", "avg_fit",
    "best_len",  "avg_len", "eval_cnt", "res_eval",
    "jac_eval",  "opt_time", "seed",   "elapsed",
]
REPORT_METRICS = ["r2_tr", "r2_te", "mae_tr", "mae_te", "elapsed"]

# For these metrics a negative Δmed (new < ref) means improvement.
LOWER_IS_BETTER = {"mae_tr", "mae_te", "nmse_tr", "nmse_te", "elapsed"}

# ── Run configuration & result ─────────────────────────────────────────────────
@dataclass(frozen=True)
class RunConfig:
    binary:      str            # "operon_gp" or "operon_nsgp"
    dataset:     Dataset
    seed:        int
    creator:     str = "btc"
    symbols:     Optional[str] = None
    objective:   str = "r2"
    generations: int = 5
    iterations:  int = 1

    population_size: int = 200

    def cli_args(self, datadir: Path) -> list[str]:
        a = [
            "--dataset",         str(datadir / self.dataset.filename),
            "--target",          self.dataset.target,
            "--train",           self.dataset.train,
            "--test",            self.dataset.test,
            "--seed",            str(self.seed),
            "--creator",         self.creator,
            "--objective",       self.objective,
            "--generations",     str(self.generations),
            "--iterations",      str(self.iterations),
            "--population-size", str(self.population_size),
            "--pool-size",       str(self.population_size),
            "--threads",         "2",
        ]
        if self.symbols:
            a += ["--enable-symbols", self.symbols]
        return a

    def label(self) -> str:
        sym = self.symbols or "default"
        return (f"{self.binary}  ds={self.dataset.name:<20}"
                f"  creator={self.creator}  obj={self.objective}  sym={sym!s:<30}  seed={self.seed}")

    def group_key(self) -> tuple:
        """Key that identifies a group for statistical tests (everything except seed)."""
        return (self.binary, self.dataset, self.creator, self.objective, self.symbols, self.generations, self.iterations, self.population_size)


@dataclass
class RunResult:
    returncode:  int
    model_line:  str        # last stdout line (the symbolic expression)
    final_stats: dict       # parsed last numeric row
    error:       str = ""


# ── Binary runner ──────────────────────────────────────────────────────────────
def _parse_stats_line(line: str) -> dict:
    parts = line.split()
    if not parts or not parts[0].isdigit():
        return {}
    row = {}
    for col, val in zip(_STAT_COLS, parts):
        try:
            row[col] = float(val) if ("." in val or "e" in val.lower()) else int(val)
        except ValueError:
            pass
    return row


def run_binary(binary: Path, config: RunConfig, datadir: Path, timeout: int) -> RunResult:
    cmd = [str(binary)] + config.cli_args(datadir)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(-1, "", {}, "timeout")
    except Exception as exc:
        return RunResult(-1, "", {}, str(exc))

    lines = proc.stdout.strip().splitlines()
    if not lines:
        return RunResult(proc.returncode, "", {}, proc.stderr[:400])

    model_line  = lines[-1]
    final_stats = {}
    for line in reversed(lines[:-1]):
        s = _parse_stats_line(line)
        if s:
            final_stats = s
            break

    err = proc.stderr[:200] if proc.returncode != 0 else ""
    return RunResult(proc.returncode, model_line, final_stats, err)


# ── Printing helpers ───────────────────────────────────────────────────────────
def _ok(msg, no_color=False):
    tag = "PASS" if no_color else f"{GREEN}PASS{RESET}"
    print(f"  [{tag}]  {msg}")

def _fail(msg, no_color=False):
    tag = "FAIL" if no_color else f"{RED}FAIL{RESET}"
    print(f"  [{tag}]  {msg}")

def _warn(msg, no_color=False):
    tag = "WARN" if no_color else f"{YELLOW}WARN{RESET}"
    print(f"  [{tag}]  {msg}")


# ── Determinism tests ──────────────────────────────────────────────────────────
def test_determinism(ref_bin: Path, new_bin: Path, datadir: Path, args) -> int:
    """Run every config × seed and assert model strings are byte-identical."""
    bold = "" if args.no_color else BOLD
    rst  = "" if args.no_color else RESET
    print(f"\n{bold}{'='*60}{rst}")
    print(f"{bold}  DETERMINISM TESTS{rst}")
    print(f"{bold}{'='*60}{rst}")

    seeds = list(range(1, args.det_seeds + 1))

    configs = [
        RunConfig(
            binary          = binary,
            dataset         = ds,
            seed            = seed,
            creator         = creator,
            objective       = objective,
            symbols         = symbols,
            generations     = args.generations,
            iterations      = args.iterations,
            population_size = args.population_size,
        )
        for binary, ds, seed, creator, objective, symbols in itertools.product(
            ["operon_gp", "operon_nsgp"],
            args.datasets,
            seeds,
            args.creators,
            args.objectives,
            SYMBOL_SETS,
        )
    ]

    passed = failed = skipped = 0

    def run_pair(cfg: RunConfig):
        r = run_binary(ref_bin / cfg.binary, cfg, datadir, args.timeout)
        n = run_binary(new_bin / cfg.binary, cfg, datadir, args.timeout)
        return cfg, r, n

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_pair, cfg): cfg for cfg in configs}
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())

    # Print in a stable order
    results.sort(key=lambda t: t[0].label())

    for cfg, ref, new in results:
        label = cfg.label()
        nc = args.no_color
        if ref.error and new.error:
            # Both failed — not a regression, but note it
            _warn(f"{label}  ref_err={ref.error!r}  new_err={new.error!r}", nc)
            skipped += 1
        elif ref.error or new.error:
            # Asymmetric failure — one build broke, the other didn't
            _fail(f"{label}  ref_err={ref.error!r}  new_err={new.error!r}", nc)
            failed += 1
        elif not args.exact:
            # --no-exact: only verify both runs completed successfully
            if ref.returncode == 0 and new.returncode == 0:
                _ok(f"{label}  (rc=0, no-exact)", nc)
                passed += 1
            else:
                _fail(f"{label}  ref_rc={ref.returncode}  new_rc={new.returncode}", nc)
                failed += 1
        elif ref.model_line == new.model_line:
            _ok(label, nc)
            passed += 1
        else:
            _fail(label, nc)
            if args.verbose:
                print(f"      REF: {ref.model_line[:160]}")
                print(f"      NEW: {new.model_line[:160]}")
            failed += 1

    total = passed + failed
    red = "" if args.no_color else RED
    grn = "" if args.no_color else GREEN
    rst = "" if args.no_color else RESET
    summary = (f"  {red}{failed} FAILED{rst}" if failed else f"  {grn}all match{rst}")
    print(f"\n  Determinism: {passed}/{total} passed{summary}"
          + (f"  ({skipped} skipped)" if skipped else ""))
    return failed


# ── Statistical equivalence tests ─────────────────────────────────────────────
def test_statistics(ref_bin: Path, new_bin: Path, datadir: Path, args) -> int:
    """
    For each (binary, dataset, creator, symbols) group, run --stat-seeds seeds
    and compare fitness metric distributions between the two builds.
    """
    bold = "" if args.no_color else BOLD
    rst  = "" if args.no_color else RESET
    print(f"\n{bold}{'='*60}{rst}")
    print(f"{bold}  STATISTICAL EQUIVALENCE TESTS{rst}")
    scipy_note = "scipy Mann-Whitney U" if HAS_SCIPY else "no scipy – descriptive only"
    tabulate_note = "tabulate" if HAS_TABULATE else "plain text"
    print(f"{bold}  seeds={args.stat_seeds}  α={args.alpha}  ({scipy_note}, {tabulate_note}){rst}")
    print(f"{bold}{'='*60}{rst}")

    seeds = list(range(1, args.stat_seeds + 1))

    all_configs = [
        RunConfig(
            binary          = binary,
            dataset         = ds,
            seed            = seed,
            creator         = creator,
            objective       = objective,
            symbols         = symbols,
            generations     = args.generations,
            iterations      = args.iterations,
            population_size = args.population_size,
        )
        for binary, ds, seed, creator, objective, symbols in itertools.product(
            ["operon_gp", "operon_nsgp"],
            args.datasets,
            seeds,
            args.creators,
            args.objectives,
            SYMBOL_SETS,
        )
    ]

    # Collect results in parallel
    ref_results: dict[tuple, list[RunResult]] = {}
    new_results: dict[tuple, list[RunResult]] = {}
    for cfg in all_configs:
        ref_results.setdefault(cfg.group_key(), [])
        new_results.setdefault(cfg.group_key(), [])

    def run_pair(cfg: RunConfig):
        r = run_binary(ref_bin / cfg.binary, cfg, datadir, args.timeout)
        n = run_binary(new_bin / cfg.binary, cfg, datadir, args.timeout)
        return cfg, r, n

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_pair, cfg): cfg for cfg in all_configs}
        for fut in as_completed(futures):
            cfg, r, n = fut.result()
            ref_results[cfg.group_key()].append(r)
            new_results[cfg.group_key()].append(n)

    # Report per group
    failed_groups = 0
    red  = "" if args.no_color else RED
    grn  = "" if args.no_color else GREEN
    ylw  = "" if args.no_color else YELLOW
    rst2 = "" if args.no_color else RESET
    bld  = "" if args.no_color else BOLD

    for key in sorted(ref_results.keys(), key=str):
        binary, ds, creator, objective, symbols, generations, iterations, _pop = key
        sym_label   = symbols or "default"
        group_label = (f"{binary}  ds={ds.name:<20}"
                       f"  creator={creator}  obj={objective}  sym={sym_label!s:<30}")
        print(f"\n  {bld}{group_label}{rst2}")

        ref_rows = [r.final_stats for r in ref_results[key] if r.final_stats]
        new_rows = [r.final_stats for r in new_results[key] if r.final_stats]

        if not ref_rows or not new_rows:
            print(f"    {ylw}No data collected — check for errors{rst2}")
            failed_groups += 1
            continue

        def _median(xs):
            s = sorted(xs); n = len(s)
            return (s[n//2] + s[~(n//2)]) / 2

        def _mean(xs): return sum(xs) / len(xs)

        def _stdev(xs):
            m = _mean(xs)
            return (sum((x - m)**2 for x in xs) / len(xs)) ** 0.5

        group_significant = False
        table_rows = []
        table_headers = ["metric", "ref med", "ref mean±sd", "new med", "new mean±sd", "Δmed", "p-value"]

        for metric in REPORT_METRICS:
            rv = [row[metric] for row in ref_rows if metric in row]
            nv = [row[metric] for row in new_rows if metric in row]
            if not rv or not nv:
                table_rows.append([metric, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
                continue

            ref_med = _median(rv)
            new_med = _median(nv)
            ref_mn  = _mean(rv)
            new_mn  = _mean(nv)
            delta   = new_med - ref_med

            if HAS_SCIPY:
                _, pval = scipy_stats.mannwhitneyu(rv, nv, alternative="two-sided")
                sig = pval < args.alpha
                if sig:
                    group_significant = True
                pval_str = f"{red}{pval:.4f} ***{rst2}" if sig else f"{grn}{pval:.4f}{rst2}"
            else:
                pval_str = "n/a"

            if abs(delta) <= 1e-6:
                delta_str = f"{grn}{delta:+.4f}{rst2}"
            elif (delta < 0) == (metric in LOWER_IS_BETTER):
                delta_str = f"{grn}{delta:+.4f}{rst2}"   # improvement
            else:
                delta_str = f"{red}{delta:+.4f}{rst2}"   # regression

            table_rows.append([
                metric,
                f"{ref_med:+.4f}",
                f"{ref_mn:+.4f}±{_stdev(rv):.4f}",
                f"{new_med:+.4f}",
                f"{new_mn:+.4f}±{_stdev(nv):.4f}",
                delta_str,
                pval_str,
            ])

        if HAS_TABULATE:
            tbl = tabulate(table_rows, headers=table_headers, tablefmt="simple")
            for line in tbl.splitlines():
                print(f"    {line}")
        else:
            # Plain fallback — drop sd column to stay readable
            for row in table_rows:
                metric, ref_med, _, new_med, _, delta, pval = row
                print(f"    {metric:10s}  ref={ref_med}  new={new_med}  Δmed={delta}  p={pval}")

        if group_significant:
            failed_groups += 1

    total_groups = len(ref_results)
    ok_groups = total_groups - failed_groups
    summary = (f"  {red}{failed_groups} groups differ significantly{rst2}"
               if failed_groups else f"  {grn}all equivalent{rst2}")
    print(f"\n  Statistics: {ok_groups}/{total_groups} groups passed{summary}")
    return failed_groups


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ref", type=Path, help="Reference operon root (e.g. ../operon)")
    p.add_argument("new", type=Path, help="New operon root to validate (e.g. .)")

    # Dataset selection
    ds_names = [d.name for d in ALL_DATASETS]
    p.add_argument("--datasets", default=",".join(ds_names),
                   help=f"Comma-separated dataset names to include "
                        f"(default: all — {', '.join(ds_names)})")

    # Test selection
    p.add_argument("--only-determinism", action="store_true",
                   help="Run only determinism tests")
    p.add_argument("--only-stats", action="store_true",
                   help="Run only statistical tests")

    # Run parameters
    p.add_argument("--creators", default="btc",
                   help="Comma-separated list of tree creators to test "
                        "(default: btc; 'grow' can hang with restricted symbol sets)")
    p.add_argument("--objectives", default="r2",
                   help="Comma-separated list of fitness objectives to test "
                        f"(default: r2; available: {', '.join(OBJECTIVES)})")
    p.add_argument("--no-exact", dest="exact", action="store_false", default=True,
                   help="Disable exact string comparison in determinism tests — "
                        "only verify both runs complete with rc=0 (useful when "
                        "results are statistically equivalent but not numerically identical)")
    p.add_argument("--generations",     type=int, default=5,
                   help="GP generations per run (default: 5)")
    p.add_argument("--iterations",      type=int, default=1,
                   help="Local optimisation iterations per run (default: 1)")
    p.add_argument("--population-size", type=int, default=200,
                   help="Population (and pool) size per run (default: 200; "
                        "keep small to limit memory usage)")
    p.add_argument("--det-seeds",       type=int, default=3,
                   help="Number of seeds for determinism tests (default: 3, seeds 1..N)")
    p.add_argument("--stat-seeds",      type=int, default=20,
                   help="Number of seeds for statistical tests (default: 20, seeds 1..N)")
    p.add_argument("--alpha",           type=float, default=0.01,
                   help="Significance level for Mann-Whitney U (default: 0.01)")

    # Infrastructure
    p.add_argument("--datadir", type=Path, default=None,
                   help="Dataset directory (default: <ref>/data)")
    p.add_argument("--jobs",    type=int, default=2,
                   help="Parallel run pairs (default: 2; each pair = 2 processes, "
                        "so --jobs 2 means 4 operon processes at once)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-run timeout in seconds (default: 300)")
    p.add_argument("--verbose",   "-v", action="store_true",
                   help="Show mismatched model strings in determinism failures")
    p.add_argument("--no-color",        action="store_true",
                   help="Disable ANSI colour output")

    args = p.parse_args()

    # Resolve dataset list
    selected_names = {s.strip() for s in args.datasets.split(",")}
    args.datasets = [d for d in ALL_DATASETS if d.name in selected_names]
    if not args.datasets:
        sys.exit(f"No matching datasets for: {selected_names!r}")

    # Resolve creator list
    args.creators = [c.strip() for c in args.creators.split(",") if c.strip()]
    if not args.creators:
        sys.exit("--creators must not be empty")

    # Resolve objectives list
    args.objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    if not args.objectives:
        sys.exit("--objectives must not be empty")

    ref_bin = args.ref / "build" / "cli"
    new_bin = args.new / "build" / "cli"
    datadir = args.datadir or (args.ref / "data")

    # Validate paths
    for d in (ref_bin, new_bin):
        for b in ("operon_gp", "operon_nsgp"):
            exe = d / b
            if not exe.exists():
                sys.exit(f"Binary not found: {exe}")
    if not datadir.is_dir():
        sys.exit(f"Data directory not found: {datadir}")

    bld = "" if args.no_color else BOLD
    rst = "" if args.no_color else RESET
    print(f"{bld}Operon branch comparator{rst}")
    print(f"  ref : {args.ref}")
    print(f"  new : {args.new}")
    print(f"  data: {datadir}")
    print(f"  datasets    : {[d.name for d in args.datasets]}")
    print(f"  creators    : {args.creators}")
    print(f"  objectives  : {args.objectives}")
    print(f"  exact cmp   : {args.exact}")
    print(f"  sym sets    : {len(SYMBOL_SETS)}")
    print(f"  pop size    : {args.population_size}")
    print(f"  jobs/timeout: {args.jobs} / {args.timeout}s")
    ylw = "" if args.no_color else YELLOW
    rst2 = "" if args.no_color else RESET
    if not HAS_SCIPY:
        print(f"  {ylw}scipy not found — statistical tests will be descriptive only{rst2}")
    if not HAS_TABULATE:
        print(f"  {ylw}tabulate not found — stats output will use plain text{rst2}")

    total_failures = 0

    if not args.only_stats:
        total_failures += test_determinism(ref_bin, new_bin, datadir, args)

    if not args.only_determinism:
        total_failures += test_statistics(ref_bin, new_bin, datadir, args)

    bld = "" if args.no_color else BOLD
    rst = "" if args.no_color else RESET
    red = "" if args.no_color else RED
    grn = "" if args.no_color else GREEN
    colour = red if total_failures else grn
    print(f"\n{bld}{colour}Total failures: {total_failures}{rst}")
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
