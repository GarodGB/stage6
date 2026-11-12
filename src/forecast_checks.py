"""The safety checks around the forecasting DAG, one command per DAG task.

Run from the repo root:
    python -m src.forecast_checks check_data          exit 0 = new work, 99 = nothing new (Airflow: skipped), 1 = broken
    python -m src.forecast_checks check_output        after predict: did it really write good rows? exit 0 = OK, 1 = problem
    python -m src.forecast_checks update_checkpoint   write "month done" in the notebook (the LAST task)
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config

SKIP = 99                                   # Airflow's BashOperator marks exit code 99 as "skipped"
NEEDED_COLUMNS = ["account_id", "txn_date_time", "billing_amount"]
MIN_VOLUME_RATIO = 0.8                      # check_output: at least 80% of the rows of the last stamped month
MAX_SKIPPED_SHARE = 0.5                     # check_output: at most 50% of accounts with no forecast (no history)

CHECKPOINT_TABLE = """
CREATE TABLE IF NOT EXISTS forecast_runs (       -- the notebook: one row per month forecast
    target_month   TEXT NOT NULL,                -- YYYY-MM
    model_version  TEXT NOT NULL,
    rows_written   INTEGER NOT NULL,
    finished_at    TEXT NOT NULL,
    dag_run_id     TEXT,
    PRIMARY KEY (target_month, model_version)
);
"""


def open_checkpoint(db_path):
    """Open the database and make sure the notebook table exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(CHECKPOINT_TABLE)
    return con


def current_model_version(models_dir):
    """The model predict.py will use = the one named in metadata.json."""
    meta = json.loads(Path(models_dir, "metadata.json").read_text())
    return meta["model_version"]


def month_to_forecast(tx):
    """Newest month in the clean data + 1 (the same rule predict.py uses)."""
    newest = pd.to_datetime(tx["txn_date_time"]).dt.to_period("M").max()
    return (newest + 1).strftime("%Y-%m"), newest.strftime("%Y-%m")


def check_data(cfg, clean_cfg_path="config/clean_config.json"):
    p = cfg["paths"]

    # 1. the clean file exists, has what predict needs, and is not empty
    csv = Path(p["transactions"])
    if not csv.exists():
        print(f"FAIL: {csv} does not exist (did clean run?)")
        return 1
    columns = pd.read_csv(csv, nrows=0).columns
    missing = [c for c in NEEDED_COLUMNS if c not in columns]
    if missing:
        print(f"FAIL: {csv} is missing columns {missing}")
        return 1
    tx = pd.read_csv(csv, usecols=NEEDED_COLUMNS)
    rows = len(tx)
    if rows == 0:
        print(f"FAIL: {csv} has no rows")
        return 1

    # 2. clean writes where predict reads
    clean_out = json.loads(Path(clean_cfg_path).read_text())["output"]["clean_csv"]
    if Path(clean_out) != csv:
        print(f"FAIL: clean writes {clean_out} but predict reads {csv} (fix one of the 2 configs)")
        return 1

    # 3. which month is next
    target, newest = month_to_forecast(tx)
    version = current_model_version(p["models"])

    # 4. already in the notebook?
    con = open_checkpoint(p["db"])
    done = con.execute("SELECT finished_at FROM forecast_runs WHERE target_month = ? AND model_version = ?",
                       (target, version)).fetchone()
    con.close()
    print(f"data: {rows} rows, newest month {newest} -> month to forecast {target} | model {version}")
    if done:
        print(f"NOTHING NEW: {target} was already forecast by this model at {done[0]} -> skip")
        return SKIP
    print(f"NEW WORK: {target} is not in the notebook yet -> continue")
    return 0


def check_output(cfg):
    """After predict: look at what really landed in the database. Collect every problem, then decide."""
    p = cfg["paths"]
    tx = pd.read_csv(p["transactions"], usecols=["txn_date_time"])
    target, _ = month_to_forecast(tx)
    version = current_model_version(p["models"])

    con = open_checkpoint(p["db"])
    rows, skipped, empty = con.execute("""
        SELECT COUNT(*),
               SUM(model_used = 'none'),                                   -- accounts with no history
               SUM(model_used != 'none' AND predicted_balance IS NULL)     -- a forecast with no number
        FROM predictions WHERE target_month = ? AND model_version = ?""", (target, version)).fetchone()
    last = con.execute("""
        SELECT target_month, rows_written FROM forecast_runs
        WHERE NOT (target_month = ? AND model_version = ?)
        ORDER BY finished_at DESC LIMIT 1""", (target, version)).fetchone()   # the last stamped month
    con.close()
    skipped, empty = skipped or 0, empty or 0
    print(f"output for {target} | model {version}: {rows} rows, {skipped} without history, {empty} empty")

    problems = []
    # 1. something was written
    if rows == 0:
        problems.append("no rows were written")
    # 2. volume: not much fewer rows than the last stamped month
    if last and rows < MIN_VOLUME_RATIO * last[1]:
        problems.append(f"volume drop: {rows} rows vs {last[1]} for {last[0]} (limit {MIN_VOLUME_RATIO:.0%})")
    elif not last:
        print("volume: no stamped month yet to compare with")
    # 3. every forecast has a number
    if empty:
        problems.append(f"{empty} forecasts have no number")
    # 4. not too many accounts without a forecast
    if rows and skipped / rows > MAX_SKIPPED_SHARE:
        problems.append(f"{skipped / rows:.0%} of accounts have no forecast (limit {MAX_SKIPPED_SHARE:.0%})")

    if problems:
        for pr in problems:
            print("PROBLEM:", pr)
        print("CHECK FAILED -> no stamp, the month will be retried")
        return 1
    print("CHECK PASSED")
    return 0


def update_checkpoint(cfg):
    """The stamp: write "this month is done" in the notebook. Runs LAST, only if every task before succeeded."""
    p = cfg["paths"]

    # 1. which month and which model (the same 2 helpers as check_data)
    tx = pd.read_csv(p["transactions"], usecols=["txn_date_time"])
    target, _ = month_to_forecast(tx)
    version = current_model_version(p["models"])

    # 2. count what predict really wrote for that month and model
    con = open_checkpoint(p["db"])
    rows = con.execute("SELECT COUNT(*) FROM predictions WHERE target_month = ? AND model_version = ?",
                       (target, version)).fetchone()[0]
    if rows == 0:
        print(f"FAIL: no predictions for {target} / {version} -> nothing to stamp")
        con.close()
        return 1

    # 3. write the stamp (a rerun replaces the same line, never adds a second one)
    run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "by-hand")        # Airflow fills this in; by hand -> "by-hand"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with con:
        con.execute("INSERT OR REPLACE INTO forecast_runs VALUES (?, ?, ?, ?, ?)",
                    (target, version, rows, now, run_id))
    con.close()
    print(f"STAMPED: {target} done | model {version} | {rows} rows | run {run_id}")
    return 0


COMMANDS = {"check_data": check_data, "check_output": check_output, "update_checkpoint": update_checkpoint}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m src.forecast_checks [{' | '.join(COMMANDS)}]")
    sys.exit(COMMANDS[sys.argv[1]](load_config()))