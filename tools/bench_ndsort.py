#!/usr/bin/env python3
"""
Non-dominated sort benchmark runner and reporter.

Runs operon_test "[performance][ndsort]" multiple times, collects the nanobench
CSV output from each run, and aggregates across runs using median ± IQR.

Usage:
    python tools/bench_ndsort.py [options]

Examples:
    # Quick run (3 repetitions, all sections)
    python tools/bench_ndsort.py

    # More stable estimate, 2-objective section only
    python tools/bench_ndsort.py --runs 10 --section "2 objectives"

    # Extended benchmark (includes RO, BOS)
    python tools/bench_ndsort.py --tag "[.][ndsort-extended]" --runs 5

    # Plot results
    python tools/bench_ndsort.py --plot

Output columns:
    sorter   – algorithm short name (RS, MS, ENS-SS, ENS-BS)
    n        – population size
    med_ns   – median elapsed time per iteration across runs, in nanoseconds
    iqr_ns   – interquartile range (Q3 − Q1) across runs, in nanoseconds
    cv%      – coefficient of variation (IQR / median × 100) as run-stability indicator
"""

import argparse
import csv
import io
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── CSV parsing ───────────────────────────────────────────────────────────────

# nanobench CSV columns (semicolon-delimited, no header line):
# title ; name ; unit ; batch ; median(elapsed_s) ; MdAPE% ; ...
_COL_TITLE   = 0
_COL_NAME    = 1
_COL_ELAPSED = 4   # seconds per iteration

def _parse_csv_lines(text: str) -> list[dict]:
    """Extract nanobench CSV rows from mixed stdout (Catch2 + nanobench)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        reader = csv.reader(io.StringIO(line), delimiter=';')
        try:
            fields = next(reader)
        except StopIteration:
            continue
        if len(fields) < 5:
            continue
        try:
            elapsed_s = float(fields[_COL_ELAPSED])
        except ValueError:
            continue

        raw_name = fields[_COL_NAME].strip()
        # Name format: "SORTER/N", e.g. "RS/1000"
        m = re.fullmatch(r'([^/]+)/(\d+)', raw_name)
        if not m:
            continue

        rows.append({
            'section': fields[_COL_TITLE].strip(),
            'sorter':  m.group(1),
            'n':       int(m.group(2)),
            'ns':      elapsed_s * 1e9,
        })
    return rows


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_once(binary: Path, tag: str, section: str | None, timeout: int) -> str:
    cmd = [str(binary), tag]
    if section:
        cmd += ['--section', section]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] run timed out after {timeout}s", file=sys.stderr)
        return ""
    except Exception as exc:
        print(f"  [WARN] run failed: {exc}", file=sys.stderr)
        return ""
    return proc.stdout


# ── Statistics ────────────────────────────────────────────────────────────────

def _quantile(data: list[float], q: float) -> float:
    """Linear interpolation quantile (same as numpy default)."""
    data = sorted(data)
    n = len(data)
    if n == 1:
        return data[0]
    idx = q * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return data[lo] + (idx - lo) * (data[hi] - data[lo])

def _iqr(data: list[float]) -> float:
    return _quantile(data, 0.75) - _quantile(data, 0.25)

def aggregate(all_rows: list[dict]) -> dict:
    """
    Group by (section, sorter, n) and compute median ± IQR across runs.
    Returns {(section, sorter, n): {'med': float, 'iqr': float, 'n_runs': int}}.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in all_rows:
        key = (row['section'], row['sorter'], row['n'])
        groups[key].append(row['ns'])

    result = {}
    for key, values in groups.items():
        med = statistics.median(values)
        iq  = _iqr(values)
        result[key] = {'med': med, 'iqr': iq, 'n_runs': len(values)}
    return result


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.1f} ns"

def print_tables(agg: dict, runs: int) -> None:
    sections = sorted({s for s, _, _ in agg})
    for section in sections:
        # Collect (sorter, n) pairs for this section
        entries = {(so, n): v for (se, so, n), v in agg.items() if se == section}
        if not entries:
            continue

        sorters = sorted({so for so, _ in entries})
        ns      = sorted({n  for _, n  in entries})

        print(f"\n── {section}  ({runs} run{'s' if runs != 1 else ''}) ──")

        table = []
        for n in ns:
            row = [n]
            for so in sorters:
                v = entries.get((so, n))
                if v is None:
                    row.append("—")
                else:
                    med, iqr = v['med'], v['iqr']
                    cv = 100 * iqr / med if med else 0
                    row.append(f"{_fmt_ns(med)} ±{_fmt_ns(iqr)} ({cv:.0f}%)")
            table.append(row)

        headers = ["n"] + sorters
        if HAS_TABULATE:
            print(tabulate(table, headers=headers, tablefmt="simple",
                           colalign=["right"] + ["right"] * len(sorters)))
        else:
            col_w = max(len(h) for h in headers[1:] + [str(r[1]) for r in table if len(r) > 1])
            col_w = max(col_w, 24)
            header_line = f"{'n':>8}" + "".join(f"  {h:>{col_w}}" for h in sorters)
            print(header_line)
            print("-" * len(header_line))
            for row in table:
                print(f"{row[0]:>8}" + "".join(f"  {str(v):>{col_w}}" for v in row[1:]))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(agg: dict) -> None:
    if not HAS_MPL:
        print("matplotlib not available — skipping plot", file=sys.stderr)
        return

    sections = sorted({s for s, _, _ in agg})
    ncols = len(sections)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

    for ax, section in zip(axes[0], sections):
        entries = {(so, n): v for (se, so, n), v in agg.items() if se == section}
        sorters = sorted({so for so, _ in entries})
        ns      = sorted({n  for _, n  in entries})

        for so in sorters:
            ys   = [entries[(so, n)]['med'] / 1e6 for n in ns if (so, n) in entries]
            errs = [entries[(so, n)]['iqr'] / 1e6 for n in ns if (so, n) in entries]
            xs   = [n for n in ns if (so, n) in entries]
            ax.errorbar(xs, ys, yerr=errs, label=so, marker='o', capsize=3)

        ax.set_title(section)
        ax.set_xlabel("n")
        ax.set_ylabel("median elapsed (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary", type=Path, default=None,
                   help="Path to operon_test binary "
                        "(default: <repo-root>/build/test/operon_test)")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of benchmark repetitions for IQR estimation (default: 3)")
    p.add_argument("--section", default=None,
                   help="Run only this Catch2 section, e.g. '2 objectives'")
    p.add_argument("--tag", default="[performance][ndsort]",
                   help="Catch2 tag filter (default: [performance][ndsort])")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-run timeout in seconds (default: 600)")
    p.add_argument("--plot", action="store_true",
                   help="Show a matplotlib plot of median ± IQR vs n")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    binary = args.binary or (repo_root / "build" / "test" / "operon_test")
    if not binary.exists():
        sys.exit(f"Binary not found: {binary}\nBuild first or pass --binary <path>")

    print(f"Binary : {binary}")
    print(f"Tag    : {args.tag}")
    print(f"Section: {args.section or '(all)'}")
    print(f"Runs   : {args.runs}")

    all_rows: list[dict] = []
    for i in range(1, args.runs + 1):
        print(f"  run {i}/{args.runs}...", end=" ", flush=True)
        stdout = run_once(binary, args.tag, args.section, args.timeout)
        rows = _parse_csv_lines(stdout)
        all_rows.extend(rows)
        print(f"{len(rows)} entries")

    if not all_rows:
        sys.exit("No CSV rows parsed — check that the binary outputs nanobench CSV to stdout")

    agg = aggregate(all_rows)
    print_tables(agg, args.runs)

    if args.plot:
        plot_results(agg)

    if not HAS_TABULATE:
        print("\n(install tabulate for nicer tables: pip install tabulate)", file=sys.stderr)


if __name__ == "__main__":
    main()
