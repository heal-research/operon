#!/usr/bin/env python3
"""
Benchmark JIT vs interpreter evaluation (0 LM iterations).

Runs adaptive reps for each configuration and stores results in DuckDB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_operon import run_reps, save
from results_db import ResultsDB

BINARY = "build/cli/operon_nsgp"
DB = Path("results.duckdb")

DATASETS = {
    "Poly-10":    {"file": "data/Poly-10.csv",   "target": "Y",  "train": "0:300",  "test": "300:500"},
    "Friedman-I": {"file": "data/Friedman-I.csv", "target": "Y",  "train": "0:7000", "test": "7000:10000"},
}

SHARED_ARGS = [
    "--iterations", "0",
    "--generations", "200",
    "--population-size", "1000",
    "--threads", "16",
    "--enable-symbols", "add,sub,mul,div,exp,log,sqrt,sin,cos,tanh",
]

CONFIGS = {
    "interp":   [],
    "jit-all":  ["--jit", "all"],
    "jit-jac":  ["--jit", "jac"],
}


def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for ds_name, ds in DATASETS.items():
        ds_args = ["--dataset", ds["file"], "--target", ds["target"],
                   "--train", ds["train"], "--test", ds["test"]]

        for cfg_name, cfg_args in CONFIGS.items():
            label = f"{ds_name} / {cfg_name}"
            console.print(f"[bold]Running {label}...[/bold]")

            args = [*SHARED_ARGS, *ds_args, *cfg_args]
            df = run_reps(
                BINARY, args,
                adaptive=True, max_reps=50, tol=0.003, window=10,
                base_seed=42, metric="elapsed",
            )

            if df.empty:
                console.print(f"  [red]No data for {label}[/red]")
                continue

            save(df, DB,
                 binary=BINARY, dataset=ds_name, problem=cfg_name,
                 tags=["jit-eval", "iter0", cfg_name],
                 args=args)

            med = df["elapsed"].median()
            n = len(df)
            console.print(f"  [green]✓[/green] {label}: {n} reps, elapsed={med:.3f}s")

    with ResultsDB(DB) as db:
        summary = db.experiments()
        table = Table(title="Experiment Summary")
        for col in summary.columns:
            table.add_column(col)
        for _, row in summary.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)


if __name__ == "__main__":
    main()
