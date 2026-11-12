# Balance Forecast

Predicts each account's balance at the **end of next month**, standing on day 1 of that month.
One model for all accounts. Input: the clean transactions from the Stage 3 pipeline.

## Setup (once)

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the clean transactions in `data/processed/transactions_clean.csv`
(columns: `account_id, txn_date_time, billing_amount`; export of `transactions_cleaned`).
Optional: `data/processed/features_targets.csv` (Stage 4 output) to cross-check the features.

## Run (from the repo root)

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m train.train` | checks the data, compares models on walk-forward folds, tunes with Optuna, promotes a model, saves it |
| 2 | `python -m inference.test` | scores the saved model on the test month once + nested walk-forward |
| 3 | `python -m inference.predict` | forecasts next month for every account into `db/forecasts.db` |
| 4 | `python -m src.db fill_actuals` | when that month closes, copies the real balances in |
| 5 | `pytest` | runs the tests |

Quick run while developing: `python -m train.train --trials 5`.

## Rules

- A transaction counts in the month of its **transaction date**, not its settle date.
- Pending and unmatched transactions **are** counted in the balance (same rule in training and prediction).
- Every feature for month M uses only transactions **before** month M. `MATCHES_STATUS` is never a feature.
- The balance is the running total of net flows from 0 (no opening balance in the data).
- Split by **whole months**, never at random: test = the last 20% of months; walk-forward folds inside the rest.
- **Promotion:** boosting is used only if it beats the drift baseline by >= $2 with a bootstrap 95% interval below 0.
  Otherwise the drift baseline (last balance + median past move) is the forecast.

## Outputs

| Folder | Files |
|---|---|
| `models/` | `model.pkl`, `optuna_study.pkl`, `feature_columns.json`, `metadata.json` |
| `reports/` | `data_checks.json`, `model_comparison.csv`, `feature_ablation.csv`, `test_metrics.json`, `errors_by_segment.csv`, `worst_accounts.csv`, `nested_walk_forward.csv` |
| `db/forecasts.db` | tables `predictions`, `model_runs`, `test_month_log`; view `monthly_scores` |

## Latest results

_Fill in after running on the real data (see `reports/`)._

## Findings

_One page: see the Stage 5 findings._


## Stage 6: orchestration (Airflow in Docker)

The same pipeline, now run by Airflow on a schedule, with retries, checks between steps, and a UI.
What to do when something breaks: see [RUNBOOK.md](RUNBOOK.md).

### Run it

```
docker compose build          # once, and again after any change in src/ train/ inference/ config/
docker compose up -d          # starts Postgres, the Airflow scheduler and webserver
```

Open http://localhost:8080 (user and password from `.env`). Stop with `docker compose down`.

`.env` (not in git) holds: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `AIRFLOW_ADMIN_USER`, `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_SECRET_KEY`.

### The two DAGs

| DAG | Schedule | Tasks |
|---|---|---|
| `forecast` | daily | `clean -> check_data -> fill_actuals -> predict -> check_output -> update_checkpoint` |
| `retrain` | monthly | `train -> test` |

| Task | Command | Does |
|---|---|---|
| `clean` | `python -m src.clean` | raw Excel -> `data/processed/transactions_clean.csv`; stops at the quality gate, writing nothing |
| `check_data` | `python -m src.forecast_checks check_data` | is there a month not forecast yet? if not: skip the rest (exit 99) |
| `fill_actuals` | `python -m src.db fill_actuals` | copies the real balances next to old forecasts |
| `predict` | `python -m inference.predict` | features + forecast + save (upsert, never duplicates) |
| `check_output` | `python -m src.forecast_checks check_output` | rows written? not a volume drop? no empty forecasts? |
| `update_checkpoint` | `python -m src.forecast_checks update_checkpoint` | marks the month done in `forecast_runs`, only after everything above passed |
| `train` | `python -m train.train` | trains, saves the model in `models/versions/<version>/`, makes it live only if better |
| `test` | `python -m inference.test` | scores the live model on the test month |

### Design decisions

- **One image, two Pythons.** `apache/airflow:2.9.3-python3.12` runs Airflow; our libraries (pandas 3, scikit-learn 1.9.1) live in `/opt/ml-venv`, so they never clash with Airflow's. Every task is a `BashOperator` running the same `python -m ...` command as by hand. (pandas 3 needs Python 3.11+, so not the 3.10 image.)
- **`clean` is the Stage 3 Spark job in pandas**, same rules and same config, plus the Stage 2 fix for the 23 billing rows that are 100x too small (`billing_fixed`). Tested against Spark 3.5 on test data (same output), and on the real data it gives the same 2296 rows as the Stage 5 clean file.
- **Features, predict and save are one task**: they share large tables in memory and the save is a single transaction.
- **Safe to rerun**: predictions are upserted on `(account_id, target_month, model_version)`; the checkpoint table `forecast_runs` is written last, so a failed run is simply retried the next day.
- **Models are never overwritten**: every trained model is kept in `models/versions/`; `models/` holds the live one, replaced only if the new model beats it by at least $2 on the same walk-forward folds. Rollback = copy an old version back.
- **Retrain policy: monthly.** The data arrives once a month, so retraining more often would train on the same data.
- **Monitoring**: `check_output` detects a volume drop (fewer than 80% of the last stamped month), logs it, and turns the run red.
- **One writer**: only the `forecast` DAG cleans; `retrain` reads the clean file it produces.
