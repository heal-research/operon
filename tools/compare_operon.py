#!/usr/bin/env python3
"""
Compare operon_gp / operon_nsgp outputs between two build trees.

Two test modes:
  Determinism  – same seed → byte-identical model string on both builds
  Statistics   – across many seeds, fitness distributions must be
                 statistically equivalent (Mann-Whitney U via scipy)

Usage:
    python compare_operon.py <ref-root> <new-root> [options]

Examples:
    python compare_operon.py ../operon .
    python compare_operon.py . . --ref-build build --new-build build-asmjit \\
        --new-jit --only-stats --stat-seeds 50 --generations 300 \\
        --population-size 1000 --threads 16 --jobs 4
"""

import argparse
import itertools
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from run_operon import (
    ALL_DATASETS, NAMED_SYMBOL_SETS, DEFAULT_SYMBOL_SET_NAMES,
    OBJECTIVES, REPORT_METRICS, LOWER_IS_BETTER,
    RunConfig, RunResult, Dataset,
    run_batch, median, mean, stdev,
)

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_BINARIES: list[str] = ["operon_gp", "operon_nsgp"]

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── Config expansion ──────────────────────────────────────────────────────────

def make_configs(args, seeds: list[int]) -> list[tuple[str, RunConfig]]:
    """Return a flat list of (binary_name, RunConfig) for the cartesian product."""
    return [
        (binary, RunConfig(
            dataset=ds, seed=seed, creator=creator,
            objective=objective, symbols=symbols,
            generations=args.generations, iterations=args.iterations,
            threads=args.threads, population_size=args.population_size,
            model_selection=args.model_selection,
        ))
        for binary, ds, seed, creator, objective, symbols in itertools.product(
            args.binaries, args.datasets, seeds,
            args.creators, args.objectives, args.symbol_sets,
        )
    ]


# ── Determinism tests ─────────────────────────────────────────────────────────

def test_determinism(ref_bin: Path, new_bin: Path, datadir: Path, args) -> int:
    bold = "" if args.no_color else BOLD
    rst  = "" if args.no_color else RESET
    red  = "" if args.no_color else RED
    grn  = "" if args.no_color else GREEN
    ylw  = "" if args.no_color else YELLOW
    print(f"\n{bold}{'='*60}{rst}")
    print(f"{bold}  DETERMINISM TESTS{rst}")
    print(f"{bold}{'='*60}{rst}")

    seeds   = list(range(1, args.det_seeds + 1))
    configs = make_configs(args, seeds)

    new_extra = args.new_extra_args
    ref_tasks = [(ref_bin / binary, cfg, None)      for binary, cfg in configs]
    new_tasks = [(new_bin / binary, cfg, new_extra) for binary, cfg in configs]

    all_tasks = ref_tasks + new_tasks
    all_results = run_batch(all_tasks, datadir, args.timeout, args.jobs)

    ref_results = {(binary, cfg): r for (binary, cfg), r in zip(configs, all_results[:len(configs)])}
    new_results = {(binary, cfg): r for (binary, cfg), r in zip(configs, all_results[len(configs):])}

    passed = failed = skipped = 0
    rows = sorted(configs, key=lambda t: (t[0], t[1].dataset.name, t[1].seed))

    for binary, cfg in rows:
        ref = ref_results[(binary, cfg)]
        new = new_results[(binary, cfg)]
        sym   = cfg.symbols or "default"
        label = (f"{binary}  ds={cfg.dataset.name:<20}  sym={sym:<30}  seed={cfg.seed}")
        nc    = args.no_color

        if ref.error and new.error:
            tag = "WARN" if nc else f"{ylw}WARN{rst}"
            print(f"  [{tag}]  {label}  ref_err={ref.error!r}  new_err={new.error!r}")
            skipped += 1
        elif ref.error or new.error:
            tag = "FAIL" if nc else f"{red}FAIL{rst}"
            print(f"  [{tag}]  {label}  ref_err={ref.error!r}  new_err={new.error!r}")
            failed += 1
        elif not args.exact:
            if ref.returncode == 0 and new.returncode == 0:
                tag = "PASS" if nc else f"{grn}PASS{rst}"
                print(f"  [{tag}]  {label}  (rc=0, no-exact)")
                passed += 1
            else:
                tag = "FAIL" if nc else f"{red}FAIL{rst}"
                print(f"  [{tag}]  {label}  ref_rc={ref.returncode}  new_rc={new.returncode}")
                failed += 1
        elif ref.model_line == new.model_line:
            tag = "PASS" if nc else f"{grn}PASS{rst}"
            print(f"  [{tag}]  {label}")
            passed += 1
        else:
            tag = "FAIL" if nc else f"{red}FAIL{rst}"
            print(f"  [{tag}]  {label}")
            if args.verbose:
                print(f"      REF: {ref.model_line[:160]}")
                print(f"      NEW: {new.model_line[:160]}")
            failed += 1

    total   = passed + failed
    summary = (f"  {red}{failed} FAILED{rst}" if failed else f"  {grn}all match{rst}")
    print(f"\n  Determinism: {passed}/{total} passed{summary}"
          + (f"  ({skipped} skipped)" if skipped else ""))
    return failed


# ── Statistical equivalence tests ─────────────────────────────────────────────

def _cv(values: list[float]) -> float:
    """Coefficient of variation; returns inf when too few samples or near-zero mean."""
    if len(values) < 2:
        return float('inf')
    m = mean(values)
    return stdev(values) / abs(m) if abs(m) > 1e-9 else float('inf')


def _accumulate(by_group: dict, configs, results) -> None:
    for (binary, cfg), r in zip(configs, results):
        key = (binary, cfg.group_key())
        by_group.setdefault(key, [])
        if r.stats:
            by_group[key].append(r.stats)


_ADAPTIVE_METRICS = ('r2_te', 'wall_s')


def _cv_stable(ref_by_group: dict, new_by_group: dict, min_seeds: int, threshold: float) -> bool:
    for rows in list(ref_by_group.values()) + list(new_by_group.values()):
        for metric in _ADAPTIVE_METRICS:
            vals = [r[metric] for r in rows if metric in r]
            if len(vals) < min_seeds or _cv(vals) >= threshold:
                return False
    return True


def _collect_stats(ref_bin: Path, new_bin: Path, datadir: Path, args,
                   new_extra_args: Optional[list[str]]) -> tuple[dict, dict, int]:
    """Run seeds and return (ref_by_group, new_by_group, seeds_done)."""
    ref_by_group: dict[tuple, list[dict]] = {}
    new_by_group: dict[tuple, list[dict]] = {}
    adaptive = getattr(args, 'adaptive', False)

    if adaptive:
        batch_size = max(1, args.jobs)
        seeds_done = 0
        seed = 1
        while seeds_done < args.stat_seeds:
            batch = list(range(seed, min(seed + batch_size, args.stat_seeds + 1)))
            seed += len(batch)
            configs = make_configs(args, batch)
            ref_tasks = [(ref_bin / b, cfg, None)            for b, cfg in configs]
            new_tasks = [(new_bin / b, cfg, new_extra_args)  for b, cfg in configs]
            results = run_batch(ref_tasks + new_tasks, datadir, args.timeout, args.jobs)
            _accumulate(ref_by_group, configs, results[:len(configs)])
            _accumulate(new_by_group, configs, results[len(configs):])
            seeds_done += len(batch)
            if _cv_stable(ref_by_group, new_by_group, args.min_seeds, args.cv_threshold):
                break
    else:
        seeds = list(range(1, args.stat_seeds + 1))
        configs = make_configs(args, seeds)
        ref_tasks = [(ref_bin / b, cfg, None)           for b, cfg in configs]
        new_tasks = [(new_bin / b, cfg, new_extra_args) for b, cfg in configs]
        results = run_batch(ref_tasks + new_tasks, datadir, args.timeout, args.jobs)
        _accumulate(ref_by_group, configs, results[:len(configs)])
        _accumulate(new_by_group, configs, results[len(configs):])
        seeds_done = args.stat_seeds

    return ref_by_group, new_by_group, seeds_done


def test_statistics(ref_bin: Path, new_bin: Path, datadir: Path, args) -> int:
    bold = "" if args.no_color else BOLD
    rst  = "" if args.no_color else RESET
    red  = "" if args.no_color else RED
    grn  = "" if args.no_color else GREEN
    ylw  = "" if args.no_color else YELLOW

    scipy_note    = "scipy Mann-Whitney U" if HAS_SCIPY else "no scipy — descriptive only"
    tabulate_note = "tabulate" if HAS_TABULATE else "plain text"

    adaptive = getattr(args, 'adaptive', False)
    seeds_label = (f"adaptive max={args.stat_seeds} min={args.min_seeds} cv<{args.cv_threshold:.1%}"
                   if adaptive else str(args.stat_seeds))
    print(f"\n{bold}{'='*60}{rst}")
    print(f"{bold}  STATISTICAL EQUIVALENCE TESTS{rst}")
    print(f"{bold}  seeds={seeds_label}  α={args.alpha}  ({scipy_note}, {tabulate_note}){rst}")
    print(f"{bold}{'='*60}{rst}")

    ref_by_group, new_by_group, seeds_done = _collect_stats(
        ref_bin, new_bin, datadir, args, args.new_extra_args)
    if adaptive:
        print(f"\n  {bold}Adaptive stopping: {seeds_done} seeds used{rst}")

    failed_groups = 0

    for key in sorted(ref_by_group.keys(), key=str):
        binary, (ds, creator, objective, symbols, generations, iterations, pop, model_sel) = key[0], key[1]
        sym_label   = symbols or "default"
        group_label = (f"{binary}  ds={ds.name:<20}"
                       f"  creator={creator}  obj={objective}  sym={sym_label}")
        print(f"\n  {bold}{group_label}{rst}")

        ref_rows = ref_by_group.get(key, [])
        new_rows = new_by_group.get(key, [])
        if not ref_rows or not new_rows:
            print(f"    {ylw}No data — check for errors{rst}")
            failed_groups += 1
            continue

        RATIO_METRICS = {"wall_s", "sort_ms"}
        table_headers = ["metric", "ref med", "ref mean±sd", "new med", "new mean±sd", "Δ / speedup", "p-value"]
        table_rows    = []
        group_significant = False

        for metric in REPORT_METRICS:
            rv = [r[metric] for r in ref_rows if metric in r]
            nv = [r[metric] for r in new_rows if metric in r]
            if not rv or not nv:
                table_rows.append([metric] + ["n/a"] * 6)
                continue

            ref_med = median(rv);  new_med = median(nv)
            ref_mn  = mean(rv);    new_mn  = mean(nv)
            delta   = new_med - ref_med

            if HAS_SCIPY:
                _, pval = scipy_stats.mannwhitneyu(rv, nv, alternative="two-sided")
                sig     = pval < args.alpha
                if sig:
                    group_significant = True
                pval_str = f"{red}{pval:.4f} ***{rst}" if sig else f"{grn}{pval:.4f}{rst}"
            else:
                pval_str = "n/a"
                sig      = False

            improvement = (delta < 0) == (metric in LOWER_IS_BETTER)
            if metric in RATIO_METRICS:
                speedup = ref_med / new_med if new_med != 0 else float("inf")
                faster  = speedup >= 1.0
                if abs(speedup - 1.0) < 1e-4:
                    delta_str = f"{grn}{speedup:.3f}x{rst}"
                elif faster:
                    delta_str = f"{grn}{speedup:.3f}x{rst}"
                else:
                    delta_str = f"{red}{speedup:.3f}x{rst}"
            else:
                if abs(delta) <= 1e-6:
                    delta_str = f"{grn}{delta:+.4g}{rst}"
                elif improvement:
                    delta_str = f"{grn}{delta:+.4g}{rst}"
                else:
                    delta_str = f"{red}{delta:+.4g}{rst}"

            table_rows.append([
                metric,
                f"{ref_med:.4g}", f"{ref_mn:+.4g}±{stdev(rv):.4g}",
                f"{new_med:.4g}", f"{new_mn:+.4g}±{stdev(nv):.4g}",
                delta_str, pval_str,
            ])

        if HAS_TABULATE:
            for line in tabulate(table_rows, headers=table_headers, tablefmt="simple").splitlines():
                print(f"    {line}")
        else:
            for metric, ref_med, _, new_med, _, delta, pval in table_rows:
                print(f"    {metric:12s}  ref={ref_med}  new={new_med}  Δ/speedup={delta}  p={pval}")

        if group_significant:
            failed_groups += 1

    total_groups = len(ref_by_group)
    ok_groups    = total_groups - failed_groups
    summary = (f"  {red}{failed_groups} groups differ significantly{rst}"
               if failed_groups else f"  {grn}all equivalent{rst}")
    print(f"\n  Statistics: {ok_groups}/{total_groups} groups passed{summary}")
    return failed_groups


# ── JIT gate tuning via Optuna ────────────────────────────────────────────────

def tune_jit(ref_bin: Path, new_bin: Path, datadir: Path, args) -> None:
    try:
        import optuna
    except ImportError:
        sys.exit("error: optuna is required for --tune (pip install optuna)")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bold = "" if args.no_color else BOLD
    rst  = "" if args.no_color else RESET
    red  = "" if args.no_color else RED
    grn  = "" if args.no_color else GREEN

    # Base extra args from '--' passthrough, minus any gate flags Optuna will set.
    _gate_flags = {'--jit-max-length', '--jit-min-visits'}
    base_extra: list[str] = []
    skip_next = False
    for tok in (args.new_extra_args or []):
        if skip_next:
            skip_next = False
            continue
        if tok in _gate_flags:
            skip_next = True
            continue
        base_extra.append(tok)

    print(f"{bold}JIT gate search  ({args.tune_trials} trials){rst}")
    print(f"  base extra    : {' '.join(base_extra) or '(none)'}")
    print(f"  r2 tolerance  : {args.tune_r2_tolerance:+.4f}")
    print(f"  search space  : max-length [0,{args.tune_max_length_max}]"
          f"  min-visits [1,{args.tune_min_visits_max}]")

    def objective(trial):
        max_len    = trial.suggest_int('jit_max_length',  0, args.tune_max_length_max)
        min_visits = trial.suggest_int('jit_min_visits',  1, args.tune_min_visits_max)
        extra = base_extra + ['--jit-max-length', str(max_len),
                              '--jit-min-visits', str(min_visits)]

        ref_by_group, new_by_group, seeds_done = _collect_stats(
            ref_bin, new_bin, datadir, args, extra)

        speedups = []
        for key in ref_by_group:
            ref_rows = ref_by_group[key]
            new_rows = new_by_group.get(key, [])
            if not ref_rows or not new_rows:
                return -1.0

            ref_r2  = median([r['r2_te'] for r in ref_rows if 'r2_te' in r])
            new_r2  = median([r['r2_te'] for r in new_rows if 'r2_te' in r])
            if ref_r2 - new_r2 > args.tune_r2_tolerance:
                print(f"  trial #{trial.number:3d}  max-length={max_len:3d}  min-visits={min_visits:3d}"
                      f"  seeds={seeds_done}  PRUNED (r2 drop {ref_r2-new_r2:.4f})")
                return -1.0

            ref_wall = median([r['wall_s'] for r in ref_rows if 'wall_s' in r])
            new_wall = median([r['wall_s'] for r in new_rows if 'wall_s' in r])
            speedups.append(ref_wall / new_wall if new_wall > 0 else 0.0)

        score = mean(speedups) if speedups else 0.0
        clr = grn if score >= 1.0 else red
        print(f"  trial #{trial.number:3d}  max-length={max_len:3d}  min-visits={min_visits:3d}"
              f"  seeds={seeds_done}  speedup={clr}{score:.3f}x{rst}")
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n{bold}Best trial #{best.number}{rst}"
          f"  speedup={grn}{best.value:.3f}x{rst}")
    print(f"  --jit-max-length {best.params['jit_max_length']}"
          f"  --jit-min-visits {best.params['jit_min_visits']}")

    valid = sorted(
        [t for t in study.trials if t.value is not None and t.value >= 1.0],
        key=lambda t: t.value, reverse=True)
    if valid:
        print(f"\n  Top {min(5, len(valid))} speedup trials:")
        for t in valid[:5]:
            print(f"    #{t.number:3d}  {t.value:.3f}x"
                  f"  max-length={t.params['jit_max_length']}"
                  f"  min-visits={t.params['jit_min_visits']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ref", type=Path, help="Reference operon root (e.g. ../operon)")
    p.add_argument("new", type=Path, help="New operon root to compare (e.g. .)")

    ds_names  = [d.name for d in ALL_DATASETS]
    sym_names = list(NAMED_SYMBOL_SETS.keys())

    p.add_argument("--datasets",      default=",".join(ds_names),
                   help=f"Comma-separated dataset names (default: all)")
    p.add_argument("--symbol-sets",   default=",".join(DEFAULT_SYMBOL_SET_NAMES),
                   help=f"Comma-separated symbol set names (default: {','.join(DEFAULT_SYMBOL_SET_NAMES)}; "
                        f"available: {', '.join(sym_names)})")
    p.add_argument("--binaries",      default=",".join(ALL_BINARIES),
                   help=f"Comma-separated binaries (default: all)")
    p.add_argument("--creators",      default="btc")
    p.add_argument("--objectives",    default="r2")

    p.add_argument("--only-determinism", action="store_true")
    p.add_argument("--only-stats",       action="store_true")
    p.add_argument("--no-exact", dest="exact", action="store_false", default=True,
                   help="Skip byte-identical check; only verify rc=0")

    p.add_argument("--generations",     type=int, default=5)
    p.add_argument("--iterations",      type=int, default=0)
    p.add_argument("--population-size", type=int, default=200)
    p.add_argument("--threads",         type=int, default=2,
                   help="Threads per operon process (default: 2)")
    p.add_argument("--det-seeds",       type=int, default=3)
    p.add_argument("--stat-seeds",      type=int, default=20,
                   help="Number of seeds for statistics (max cap in adaptive mode, default: 20)")
    p.add_argument("--adaptive",        action="store_true",
                   help="Stop early when r2_te CV stabilises across all groups")
    p.add_argument("--cv-threshold",    type=float, default=0.02,
                   help="CV threshold for adaptive stopping (default: 0.02 = 2%%)")
    p.add_argument("--min-seeds",       type=int, default=5,
                   help="Minimum seeds before adaptive stopping is checked (default: 5)")
    p.add_argument("--alpha",           type=float, default=0.01)

    p.add_argument("--ref-build",  default="build",
                   help="Build subdirectory under ref root (default: build)")
    p.add_argument("--new-build",  default="build",
                   help="Build subdirectory under new root (default: build)")
    p.add_argument("--model-selection", default="mdl",
                   help="Pareto front model selection: obj0, mdl, bic, aic (default: mdl)")
    p.add_argument("--jobs",       type=int, default=1,
                   help="Concurrent operon processes (default: 1 = sequential)")
    p.add_argument("--timeout",    type=int, default=300)
    p.add_argument("--datadir",    type=Path, default=None,
                   help="Dataset directory (default: <ref>/data)")
    p.add_argument("--verbose",    "-v", action="store_true",
                   help="Show mismatched model strings in determinism failures")
    p.add_argument("--no-color",   action="store_true")

    # Optuna JIT gate tuning
    p.add_argument("--tune",               action="store_true",
                   help="Search for optimal JIT gate params using Optuna (requires optuna)")
    p.add_argument("--tune-trials",        type=int, default=30,
                   help="Number of Optuna trials (default: 30)")
    p.add_argument("--tune-r2-tolerance",  type=float, default=0.005,
                   help="Max allowed r2_te drop vs ref before a trial is pruned (default: 0.005)")
    p.add_argument("--tune-max-length-max",  type=int, default=100,
                   help="Upper bound of jit-max-length search range (default: 100)")
    p.add_argument("--tune-min-visits-max",  type=int, default=50,
                   help="Upper bound of jit-min-visits search range (default: 50)")

    # Split on '--' before argparse sees argv so REMAINDER doesn't eat our own flags.
    import sys as _sys
    argv = _sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        our_argv, extra_argv = argv[:idx], argv[idx+1:]
    else:
        our_argv, extra_argv = argv, []

    args = p.parse_args(our_argv)

    # Resolve lists
    selected_ds = {s.strip() for s in args.datasets.split(",")}
    args.datasets = [d for d in ALL_DATASETS if d.name in selected_ds]
    if not args.datasets:
        sys.exit(f"No matching datasets for: {selected_ds!r}")

    selected_sym = [s.strip() for s in args.symbol_sets.split(",") if s.strip()]
    unknown_sym  = [s for s in selected_sym if s not in NAMED_SYMBOL_SETS]
    if unknown_sym:
        sys.exit(f"Unknown symbol sets: {unknown_sym!r}  (available: {sym_names})")
    args.symbol_sets = [NAMED_SYMBOL_SETS[k] for k in selected_sym]

    args.binaries   = [b.strip() for b in args.binaries.split(",")   if b.strip()]
    args.creators   = [c.strip() for c in args.creators.split(",")   if c.strip()]
    args.objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]

    args.new_extra_args = extra_argv if extra_argv else None
    if args.new_extra_args:
        args.exact = False

    ref_bin = args.ref / args.ref_build / "cli"
    new_bin = args.new / args.new_build / "cli"
    datadir = args.datadir or (args.ref / "data")

    for d in (ref_bin, new_bin):
        for b in args.binaries:
            exe = d / b
            if not exe.exists():
                sys.exit(f"Binary not found: {exe}")
    if not datadir.is_dir():
        sys.exit(f"Data directory not found: {datadir}")

    bld = "" if args.no_color else BOLD
    rst = "" if args.no_color else RESET
    ylw = "" if args.no_color else YELLOW
    print(f"{bld}Operon branch comparator{rst}")
    print(f"  ref         : {args.ref}  (build: {args.ref_build})")
    print(f"  new         : {args.new}  (build: {args.new_build})")
    print(f"  data        : {datadir}")
    print(f"  binaries    : {args.binaries}")
    print(f"  datasets    : {[d.name for d in args.datasets]}")
    print(f"  symbol sets : {len(args.symbol_sets)}")
    print(f"  generations : {args.generations}   iterations: {args.iterations}")
    print(f"  pop size    : {args.population_size}")
    print(f"  threads/job : {args.threads}   jobs: {args.jobs}   timeout: {args.timeout}s")
    print(f"  model-sel   : {args.model_selection}")
    if args.new_extra_args:
        print(f"  new extra   : {' '.join(args.new_extra_args)}")
    if not HAS_SCIPY:
        print(f"  {ylw}scipy not found — statistical tests will be descriptive only{rst}")
    if not HAS_TABULATE:
        print(f"  {ylw}tabulate not found — falling back to plain text{rst}")

    if args.tune:
        tune_jit(ref_bin, new_bin, datadir, args)
        sys.exit(0)

    failures = 0
    if not args.only_stats:
        failures += test_determinism(ref_bin, new_bin, datadir, args)
    if not args.only_determinism:
        failures += test_statistics(ref_bin, new_bin, datadir, args)

    clr = (""  if args.no_color else RED  if failures else GREEN)
    rst = ("" if args.no_color else RESET)
    print(f"\n{bld}{clr}Total failures: {failures}{rst}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
