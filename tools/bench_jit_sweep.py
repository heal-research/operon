#!/usr/bin/env python3
"""
Sweep LM iterations to find JIT breakeven point.

Runs interp vs jit-all vs jit-jac at increasing iteration counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_operon import run_reps, save
from results_db import ResultsDB

BINARY = "build/cli/operon_nsgp"
DB = Path("results.duckdb")

DATASETS = {
    "Poly-10":       {"file": "data/Poly-10.csv",        "target": "Y",      "train": "0:300",   "test": "300:500"},
    "Friedman-I":    {"file": "data/Friedman-I.csv",     "target": "Y",      "train": "0:7000",  "test": "7000:10000"},
    "Friedman-II":   {"file": "data/Friedman-II.csv",    "target": "Y",      "train": "0:7000",  "test": "7000:10000"},
    "Breiman-I":     {"file": "data/Breiman-I.csv",      "target": "Y",      "train": "0:7000",  "test": "7000:10001"},
    "Chemical-I":    {"file": "data/Chemical-I.csv",     "target": "target", "train": "0:3500",  "test": "3500:4999"},
    "Vlad-4":        {"file": "data/Vladislavleva-4.csv","target": "Y",      "train": "0:4000",  "test": "4000:6024"},
    "Flow-stress":   {"file": "data/Flow_stress.csv",    "target": "target", "train": "0:5500",  "test": "5500:7800"},
}

BASE_ARGS = [
    "--generations", "200",
    "--population-size", "1000",
    "--threads", "16",
    "--enable-symbols", "add,sub,mul,div,exp,log,sqrt,sin,cos,tanh",
]

ITERATIONS = [0, 5, 10, 20, 50]

CONFIGS = {
    "interp":   [],
    "jit-all":  ["--jit", "all"],
    "jit-jac":  ["--jit", "jac"],
}


def main():
    from rich.console import Console
    console = Console()

    for iters in ITERATIONS:
        for ds_name, ds in DATASETS.items():
            ds_args = ["--dataset", ds["file"], "--target", ds["target"],
                       "--train", ds["train"], "--test", ds["test"]]

            for cfg_name, cfg_args in CONFIGS.items():
                label = f"iter={iters} {ds_name} / {cfg_name}"
                console.print(f"[bold]{label}...[/bold]")

                args = [*BASE_ARGS, "--iterations", str(iters), *ds_args, *cfg_args]
                df = run_reps(
                    BINARY, args,
                    adaptive=True, max_reps=50, tol=0.005, window=10,
                    base_seed=42, metric="elapsed",
                )

                if df.empty:
                    console.print(f"  [red]No data[/red]")
                    continue

                save(df, DB,
                     binary=BINARY, dataset=ds_name,
                     problem=f"{cfg_name}/iter{iters}",
                     tags=["jit-sweep", cfg_name, f"iter{iters}"],
                     args=args)

                med_elapsed = df["elapsed"].median()
                med_r2 = df["r2_te"].median()
                console.print(f"  [green]✓[/green] {len(df)} reps, elapsed={med_elapsed:.3f}s, r2_te={med_r2:.4f}")

    console.print("\n[bold]Summary[/bold]")
    with ResultsDB(DB) as db:
        print(db.query("""
            SELECT
                REGEXP_EXTRACT(e.problem, '([^/]+)', 1) AS mode,
                CAST(e.config->>'iterations' AS INT) AS iters,
                e.dataset,
                COUNT(DISTINCT r.rep) AS reps,
                ROUND(MEDIAN(r.elapsed), 3) AS elapsed,
                ROUND(MEDIAN(r.r2_te), 4) AS r2_te,
                ROUND(MEDIAN(r.opt_time), 3) AS opt_time
            FROM experiments e JOIN runs r USING (experiment_id)
            WHERE list_contains(e.tags, 'jit-sweep')
            GROUP BY ALL
            ORDER BY e.dataset, iters, mode
        """).to_string(index=False))


if __name__ == "__main__":
    main()
