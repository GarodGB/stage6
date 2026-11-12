# Runbook: balance forecast pipeline (Stage 6)

For whoever is on call. Everything runs in Docker on one machine; the Airflow UI is the place to look.

## What runs

| DAG | Schedule | Tasks | Writes |
|---|---|---|---|
| `forecast` | daily | `clean → check_data → fill_actuals → predict → check_output → update_checkpoint` | `data/processed/transactions_clean.csv`, `db/forecasts.db` |
| `retrain` | monthly (1st, 00:00) | `train → test` | `models/versions/<version>/`, and `models/` only if the new model is better |

- **Input:** the raw Excel `data/raw/synthetic_dirty_transactions_v4.xlsx` (path set in `config/clean_config.json`).
- **A normal quiet day:** `clean` green, `check_data` pink (skipped) and everything after it pink. Nothing new = nothing to do.
- **A normal new month:** all 6 tasks green; `update_checkpoint` writes one row in `forecast_runs`.

**Retrain policy: fixed monthly schedule.** The data arrives once a month, so retraining more often would train on the same data. A new model replaces the live one only if it beats it by at least $2 on the same walk-forward folds; every trained model is kept in `models/versions/`.

## How you'll know something is wrong

A **red** run on the Airflow home page (`http://localhost:8080`). Open the DAG → **Grid** → click the **red** square → **Logs**. The last lines say why. Tasks after a red one are orange (`upstream_failed`): they did not run, fix the red one first. Every task retries on its own (2 times for `forecast`, 1 time for `retrain`) before turning red, so a red task is not a short hiccup.

## What can break, and what to do

| What breaks | How you'll know | What to do |
|---|---|---|
| **Data late** (new month not delivered yet) | Nothing is red. `check_data` stays pink and `forecast_runs` has no row for the new month after the 3rd of the month. | Nothing to fix in the pipeline: as soon as the file lands, the next daily run forecasts it. Chase the data provider. If `retrain` already ran on the 1st with old data, it kept the live model; after the data lands, run `retrain` again (▶). |
| **Data not arriving at all** (file missing) | `clean` red: `FileNotFoundError ... data/raw/...` | Get the file, put it in `data/raw/` with the name in `config/clean_config.json`, then **Clear** `clean` (the rest follows). |
| **Data malformed** (bad dates, amounts, columns) | `clean` red: `QUALITY GATE FAILED` + the list of problems, or `STOP: missing columns [...]`. Nothing was written: the previous clean file is still used. | Read the problems in the log (or `reports/clean_report.json`). Ask for a corrected file, replace it in `data/raw/`, **Clear** `clean`. Never loosen the gate to make a bad file pass. |
| **New month rejected as "date_outside_window"** | `clean` red, problem `date_outside_window: N rows` | The cleaning rules accept dates up to `window_end` in `config/clean_config.json`. Move it to the end of the new month, `docker compose build`, `docker compose up -d`, **Clear** `clean`. |
| **Output looks wrong** (too few rows, empty forecasts) | `check_output` red: `volume drop`, `forecasts have no number`, or `% of accounts have no forecast`. No stamp was written. | Check the raw file is complete (all accounts). If it was incomplete: replace it, **Clear** `clean`. If the drop is real (accounts really closed): stamp by hand (command below). |
| **Clean file missing or configs disagree** | `check_data` red: `does not exist` or `clean writes X but predict reads Y` | Make `output.clean_csv` in `config/clean_config.json` and `paths.transactions` in `config.yaml` the same path, rebuild, **Clear**. |
| **Model missing or wrong scikit-learn** | `predict` red: `run python -m train.train first` or `scikit-learn X is installed, but the model was trained with Y` | Missing: run `retrain` (▶). Version: install the version in `requirements.txt`, `docker compose build`. |
| **Training stopped by the data checks** | `retrain` → `train` red: `DataCheckError [...]` | Read `reports/data_checks.json`. The live model is untouched, forecasting continues with it. |
| **A promoted model turns out bad** | Forecasts look wrong after a retrain | Roll back (command below). The next daily run sees the model version changed and forecasts the current month with the rolled-back model (unless that model already forecast it). |
| **`database is locked`** | Any task red with this message after its retries | Something else holds `db/forecasts.db` open (e.g. an unsaved edit in DBeaver). Close it, **Clear** the task. |
| **Airflow itself down** | `localhost:8080` does not open | `docker compose ps`, then `docker compose up -d`. |

## Useful commands (in the `stage6` folder)

| Need | Command |
|---|---|
| Start / stop everything | `docker compose up -d` / `docker compose down` |
| After changing code in `src/ train/ inference/ config/` | `docker compose build` then `docker compose up -d` |
| Rerun one failed task | Airflow UI → red square → **Clear task** |
| Stamp a month by hand | `docker compose exec airflow_scheduler bash -c "cd /opt/airflow/project && /opt/ml-venv/bin/python -m src.forecast_checks update_checkpoint"` |
| Forecast a month again | DBeaver on `db/forecasts.db`: `DELETE FROM forecast_runs WHERE target_month = 'YYYY-MM';` then ▶ `forecast` |
| Roll back the model | Copy the 4 files of `models/versions/<old version>/` into `models/` (overwrite) |

## Known limits

- A new raw Excel must contain **all months**, not only the new one: `clean` rebuilds the whole clean file from it.
- The clean file's fingerprint (in the model name) changes on every `clean` run because the file stores `cleaned_at`; the numbers inside do not.
- Alerts are the red runs in the Airflow UI; no email or chat alert is set up.
