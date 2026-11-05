"""Verify the SQL in sql/ against the Python results on a real PostgreSQL server.

The script creates the schema and loads the generated CSV files with psql,
runs each analysis query, and compares the result sets with
``reports/results.json`` produced by ``scripts/run_analysis.py``.

Usage (after `python scripts/run_analysis.py`):

    python scripts/verify_sql_postgres.py \
        --dsn "host=127.0.0.1 port=5432 user=postgres password=postgres dbname=postgres"

The DSN uses libpq keyword syntax so the same command works on Windows and
on the GitHub Actions runner. Requires the `psql` client on PATH (or set
PSQL to its full path).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"
RESULTS = ROOT / "reports" / "results.json"


def psql_binary() -> str:
    env = os.environ.get("PSQL")
    if env:
        return env
    found = shutil.which("psql")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\PostgreSQL\18\bin\psql.exe",
        r"C:\Program Files\PostgreSQL\17\bin\psql.exe",
        r"C:\Program Files\PostgreSQL\16\bin\psql.exe",
    ):
        if Path(candidate).exists():
            return candidate
    raise SystemExit("psql not found; set the PSQL environment variable")


def run_psql(dsn: str, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [psql_binary(), dsn, "-v", "ON_ERROR_STOP=1", *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PGCLIENTENCODING": "UTF8"},
    )


def load(dsn: str) -> None:
    for script in ("01_schema.sql",):
        proc = run_psql(dsn, ["-q", "-f", str(SQL / script)])
        if proc.returncode:
            raise SystemExit(f"{script} failed:\n{proc.stderr}")
    # Make the load idempotent: the workflow may already have loaded the same CSV.
    proc = run_psql(dsn, ["-q", "-c", "TRUNCATE events, experiment_assignments;"])
    if proc.returncode:
        raise SystemExit(f"truncate failed:\n{proc.stderr}")
    proc = run_psql(dsn, ["-q", "-f", str(SQL / "05_load.psql")], stdin=None)
    if proc.returncode:
        raise SystemExit(f"05_load.psql failed:\n{proc.stderr}")


def query(dsn: str, sql_text: str) -> pd.DataFrame:
    body = sql_text.strip().rstrip(";")
    copy = f"COPY (\n{body}\n) TO STDOUT WITH (FORMAT csv, HEADER true);"
    proc = run_psql(dsn, ["-q", "-f", "-"], stdin=copy)
    if proc.returncode:
        raise SystemExit(f"query failed:\n{proc.stderr}")
    return pd.read_csv(io.StringIO(proc.stdout))


def strip_comments(text: str) -> str:
    return re.sub(r"(?m)^--.*$", "", text)


def compare(label: str, sql_df: pd.DataFrame, py_rows: list[dict], keys: list[str],
            numeric: list[str]) -> bool:
    py_df = pd.DataFrame(py_rows)
    ok = True
    if len(sql_df) != len(py_df):
        print(f"[FAIL] {label}: rows {len(sql_df)} != {len(py_df)}")
        return False
    sql_df = sql_df.sort_values(keys).reset_index(drop=True)
    py_df = py_df.sort_values(keys).reset_index(drop=True)
    for column in keys:
        if not (sql_df[column].astype(str) == py_df[column].astype(str)).all():
            print(f"[FAIL] {label}: key mismatch in {column}")
            ok = False
    for column in numeric:
        if not (abs(sql_df[column].astype(float) - py_df[column].astype(float)) < 1e-9).all():
            print(f"[FAIL] {label}: value mismatch in {column}")
            ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {len(py_df)} rows")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="host=127.0.0.1 port=5432 user=postgres dbname=postgres")
    args = parser.parse_args()

    if not RESULTS.exists():
        raise SystemExit("reports/results.json not found; run scripts/run_analysis.py first")
    expected = json.loads(RESULTS.read_text(encoding="utf-8"))

    load(args.dsn)

    funnel_sql = query(args.dsn, strip_comments((SQL / "02_funnel.sql").read_text(encoding="utf-8")))
    retention_sql = query(args.dsn, strip_comments((SQL / "03_retention.sql").read_text(encoding="utf-8")))
    srm_sql = query(args.dsn, strip_comments((SQL / "04_srm.sql").read_text(encoding="utf-8")))

    checks = [
        compare(
            "funnel", funnel_sql, expected["funnel"], ["variant"],
            ["n_assigned", "n_landing_view", "n_signup", "n_activation", "n_purchase"],
        ),
        compare(
            "retention_by_cohort", retention_sql, expected["retention_by_cohort"],
            ["cohort_day_utc", "variant"], ["cohort_users", "d1_users", "d7_users"],
        ),
    ]

    experiment = expected["experiment"]
    srm_row = srm_sql.iloc[0]
    srm_ok = (
        int(srm_row["n_control"]) == int(experiment["n_control"])
        and int(srm_row["n_treatment"]) == int(experiment["n_treatment"])
    )
    print(f"[{'PASS' if srm_ok else 'FAIL'}] srm_assignment_counts: "
          f"control={int(srm_row['n_control'])} treatment={int(srm_row['n_treatment'])}")
    checks.append(srm_ok)

    if not all(checks):
        raise SystemExit(1)
    print("PostgreSQL results match the Python analysis.")


if __name__ == "__main__":
    main()
