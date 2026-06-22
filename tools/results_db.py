#!/usr/bin/env python3
"""
Persistent experiment storage backed by DuckDB.

Usage from other tools:
    from results_db import ResultsDB
    db = ResultsDB("results.duckdb")
    db.store(df, binary="build/cli/operon_nsgp", dataset="Poly-10.csv")
    db.summary()
"""
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        return subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
        ) != 0
    except FileNotFoundError:
        return False


def _parse_args(args: list[str]) -> dict:
    result = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                result[key] = args[i + 1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            i += 1
    return result


class ResultsDB:
    def __init__(self, path: str | Path = "results.duckdb"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                "timestamp"   TIMESTAMPTZ,
                git_commit    VARCHAR,
                git_dirty     BOOLEAN,
                "binary"      VARCHAR,
                dataset       VARCHAR,
                problem       VARCHAR,
                tags          VARCHAR[],
                config        JSON
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                experiment_id VARCHAR,
                rep           INTEGER,
                iteration     INTEGER,
                r2_tr         DOUBLE, r2_te         DOUBLE,
                mae_tr        DOUBLE, mae_te        DOUBLE,
                nmse_tr       DOUBLE, nmse_te       DOUBLE,
                best_fit      DOUBLE, avg_fit       DOUBLE,
                best_len      DOUBLE, avg_len       DOUBLE,
                eval_cnt      BIGINT, res_eval      BIGINT,
                jac_eval      BIGINT,
                opt_time      DOUBLE,
                seed          BIGINT,
                sort_ms       DOUBLE,
                elapsed       DOUBLE,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

    def store(
        self,
        df: pd.DataFrame,
        *,
        binary: str = "",
        dataset: str = "",
        problem: str = "",
        tags: list[str] | None = None,
        config: dict | None = None,
        args: list[str] | None = None,
    ) -> str:
        if df.empty:
            return ""

        experiment_id = uuid.uuid4().hex[:12]
        commit = _git_commit()
        dirty = _git_dirty()
        now = datetime.now(timezone.utc)

        if args is not None:
            parsed = _parse_args(args)
            config = {**(config or {}), **parsed}

        import json
        self.con.execute(
            """INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [experiment_id, now, commit, dirty, binary, dataset, problem,
             tags or [], json.dumps(config) if config else None],
        )

        run_cols = [c for c in df.columns if c in {
            "rep", "iteration",
            "r2_tr", "r2_te", "mae_tr", "mae_te",
            "nmse_tr", "nmse_te", "best_fit", "avg_fit",
            "best_len", "avg_len", "eval_cnt", "res_eval",
            "jac_eval", "opt_time", "seed", "sort_ms", "elapsed",
        }]

        run_df = df[run_cols].copy()
        run_df.insert(0, "experiment_id", experiment_id)
        self.con.execute("INSERT INTO runs SELECT * FROM run_df")

        return experiment_id

    def experiments(self, limit: int = 20) -> pd.DataFrame:
        return self.con.execute(f"""
            SELECT e.experiment_id, e.timestamp, e.git_commit, e.git_dirty,
                   e.binary, e.dataset, e.problem, e.tags,
                   COUNT(DISTINCT r.rep) AS reps,
                   MEDIAN(r.r2_te) AS r2_te_median,
                   MEDIAN(r.elapsed) AS elapsed_median
            FROM experiments e
            JOIN runs r USING (experiment_id)
            GROUP BY ALL
            ORDER BY e.timestamp DESC
            LIMIT {limit}
        """).df()

    def runs(self, experiment_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY rep, iteration",
            [experiment_id],
        ).df()

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
