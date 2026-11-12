"""Training: check data, compare models on walk-forward folds, tune, promote, save.

Run from the repo root:  python -m train.train            (full run)
                         python -m train.train --trials 5  (quick run while developing)
It never looks at the test month's target.
"""
import argparse
import hashlib
import json
import logging
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn

from src import db
from src.config import load_config
from src.data import DataCheckError, load_stage4, run_checks
from src.evaluate import fold_predictions, fold_table, paired_bootstrap
from src.features import build_features, load_transactions
from src.models import (DriftModel, HGBChangeModel, MedianLinearModel, NaiveModel,
                        ThreeMonthAverageModel)
from src.split import make_split, useless_columns
from src.tuning import tune

log = logging.getLogger("train")
ABLATION_GROUPS = {
    "balance": ["balance_lag1", "balance_lag2", "balance_lag3", "balance_mean_3m", "balance_std_3m"],
    "money flow": ["credit_lag1", "debit_lag1", "net_lag1", "credit_avg_3m", "debit_avg_3m", "net_avg_3m"],
    "activity": ["txn_count_lag1"],
}


ARTIFACT_FILES = ["model.pkl", "optuna_study.pkl", "feature_columns.json", "metadata.json"]


def save_version(models_dir, version, final_model, study, features, metadata):
    """Save a trained model in its OWN folder models/versions/<version>/. Never touches the live model."""
    vdir = Path(models_dir, "versions", version)
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, vdir / "model.pkl")
    joblib.dump(study, vdir / "optuna_study.pkl")
    (vdir / "feature_columns.json").write_text(json.dumps(features, indent=2))
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return vdir


def promote(models_dir, vdir):
    """Make a saved version the LIVE model: copy its 4 files into models/.
    The old live model is kept in models/versions/ first, so it can always be brought back."""
    live = Path(models_dir)
    meta = live / "metadata.json"
    if meta.exists():
        old_dir = live / "versions" / json.loads(meta.read_text())["model_version"]
        if not old_dir.exists():                  # a live model saved before versions/ existed
            old_dir.mkdir(parents=True)
            for f in ARTIFACT_FILES:
                if (live / f).exists():
                    shutil.copy2(live / f, old_dir / f)
    for f in ARTIFACT_FILES:                      # metadata.json last: it names the live version
        shutil.copy2(vdir / f, live / f)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(rows):
    return (rows["target_balance"] - rows["pred"]).abs().to_numpy()


def main(config_path="config.yaml", n_trials=None):
    # 1. config + seed
    cfg = load_config(config_path)
    p, mcfg, ecfg = cfg["paths"], cfg["model"], cfg["evaluation"]
    random.seed(mcfg["seed"]); np.random.seed(mcfg["seed"])
    Path(p["models"]).mkdir(parents=True, exist_ok=True)
    Path(p["reports"]).mkdir(parents=True, exist_ok=True)

    # 2 + 4 + 5. load transactions, build features and the target with the shared function
    tx = load_transactions(p["transactions"])
    feat = build_features(tx)
    log.info("feature table: %d rows, %d accounts, months %s -> %s", len(feat), feat["account_id"].nunique(),
             feat["month"].min().strftime("%Y-%m"), feat["month"].max().strftime("%Y-%m"))

    # 3. data checks (stop on a hard failure)
    s4 = p.get("stage4_features")
    stage4 = load_stage4(s4) if s4 and Path(s4).exists() else None
    checks = run_checks(feat, tx, stage4)
    Path(p["reports"], "data_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    if checks["hard_fail"]:
        raise DataCheckError(checks["hard_fail"])
    for w in checks["warnings"]:
        log.warning(w)

    # 6 + 7. split by whole months, walk-forward folds inside the training months
    train_months, test_months, fold_months = make_split(feat, cfg)
    train = feat[feat["month"].isin(train_months)].copy()     # the test month is not used below
    log.info("train %s | test %s | folds %s", [str(m)[:7] for m in train_months],
             [str(m)[:7] for m in test_months], [str(m)[:7] for m in fold_months])

    # 8. constant / duplicate columns, decided on the training months only
    dropped = useless_columns(train, cfg["features"]["use"])
    features = [c for c in cfg["features"]["use"] if c not in dropped]
    if dropped:
        log.warning("dropped on the training months: %s", dropped)

    # 9. baselines + the default boosting on the folds
    makers = {
        "naive": NaiveModel,
        "drift": DriftModel,
        "3-month average": ThreeMonthAverageModel,
        "median linear": lambda: MedianLinearModel(features),
        "hgb default": lambda: HGBChangeModel(features, mcfg["default_params"], mcfg["loss"], mcfg["seed"]),
    }
    preds = {name: fold_predictions(make, train, fold_months) for name, make in makers.items()}

    # 10. group ablation (evidence only: the feature list itself was chosen in the Stage 5 study)
    abl = []
    base = errors(preds["hgb default"])
    for group, cols in ABLATION_GROUPS.items():
        without = [c for c in features if c not in cols]
        if not without:
            continue
        rows = fold_predictions(lambda: HGBChangeModel(without, mcfg["default_params"], mcfg["loss"],
                                                       mcfg["seed"]), train, fold_months)
        change, lo, hi = paired_bootstrap(base - errors(rows), ecfg["bootstrap_samples"], ecfg["seed"])
        for m, g in rows.groupby("month"):
            keep = preds["hgb default"]["month"] == m
            abl.append({"group": group, "fold": str(m)[:7],
                        "mae_with": float(errors(preds["hgb default"][keep]).mean()),
                        "mae_without": float(errors(g).mean()),
                        "pooled_change_when_kept": change, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(abl).to_csv(Path(p["reports"], "feature_ablation.csv"), index=False)

    # 11. Optuna on the folds
    study = tune(train, fold_months, features, cfg, n_trials=n_trials)
    default_mae = float(errors(preds["hgb default"]).mean())
    use_tuned = study.best_value <= default_mae - cfg["tuning"]["min_gain_over_default"]
    params = study.best_params if use_tuned else dict(mcfg["default_params"])
    preds["hgb tuned"] = fold_predictions(
        lambda: HGBChangeModel(features, study.best_params, mcfg["loss"], mcfg["seed"]), train, fold_months)
    log.info("optuna best %.2f vs default %.2f -> %s", study.best_value, default_mae,
             "tuned" if use_tuned else "default (gain < min_gain_over_default)")

    # 12 + 13. compare everything on the same folds (train MAE next to validation MAE)
    comparison = pd.concat([fold_table(r, name) for name, r in preds.items()], ignore_index=True)
    comparison.to_csv(Path(p["reports"], "model_comparison.csv"), index=False)

    # promotion rule: boosting must beat drift by a real margin, else drift is the forecast
    hgb_rows = preds["hgb tuned" if use_tuned else "hgb default"]
    gain, lo, hi = paired_bootstrap(errors(hgb_rows) - errors(preds["drift"]),
                                    ecfg["bootstrap_samples"], ecfg["seed"])
    pr = cfg["promotion"]
    beats_drift = gain <= -pr["min_gain_vs_drift"] and (hi < 0 or not pr["require_ci_below_zero"])
    model_type = "hgb" if beats_drift else "drift"
    log.info("hgb vs drift on folds: %.2f (95%%: %.2f to %.2f) -> promote %s", gain, lo, hi, model_type)
    winner_rows = hgb_rows if model_type == "hgb" else preds["drift"]

    # 14. refit the winner on ALL training months
    def make_winner():
        return HGBChangeModel(features, params, mcfg["loss"], mcfg["seed"]) if model_type == "hgb" else DriftModel()
    final_model = make_winner().fit(train)

    # 15. save, following the overwrite rule
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    data_hash = file_hash(p["transactions"])
    version = f"{stamp}-{data_hash[:8]}-{cfg['_hash'][:8]}"
    promoted = True
    meta_path = Path(p["models"], "metadata.json")
    if meta_path.exists():                 # re-score the SAVED model's recipe on the SAME folds
        old = json.loads(meta_path.read_text())
        old_make = (lambda: HGBChangeModel(old["features"], old["params"], mcfg["loss"], mcfg["seed"])) \
            if old["model_type"] == "hgb" else DriftModel
        old_rows = fold_predictions(old_make, train, fold_months)
        d, _, old_hi = paired_bootstrap(errors(winner_rows) - errors(old_rows),
                                        ecfg["bootstrap_samples"], ecfg["seed"])
        promoted = d <= -pr["min_gain_vs_saved"] and old_hi < 0
        log.info("new vs saved model on the same folds: %.2f -> %s", d, "replace" if promoted else "keep the saved one")

    folds_summary = comparison[comparison["model"] == ("drift" if model_type == "drift" else
                                                       ("hgb tuned" if use_tuned else "hgb default"))]
    metadata = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": model_type,
        "features": features,
        "params": params if model_type == "hgb" else {"median_move": final_model.median_move_},
        "hgb_vs_drift_on_folds": {"mae_change": gain, "ci_low": lo, "ci_high": hi},
        "data": {"file": p["transactions"], "sha256": data_hash, "rows": int(len(tx)),
                 "first_month": str(feat["month"].min())[:7], "last_month": str(feat["month"].max())[:7]},
        "months": {"train": [str(m)[:7] for m in train_months], "folds": [str(m)[:7] for m in fold_months],
                   "test": [str(m)[:7] for m in test_months]},
        "per_fold": folds_summary.to_dict(orient="records"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "optuna": optuna.__version__, "joblib": joblib.__version__},
        "config": {k: v for k, v in cfg.items() if k != "_hash"},
    }
    metadata["promoted"] = promoted
    vdir = save_version(p["models"], version, final_model, study, features, metadata)   # ALWAYS: its own folder
    if promoted:                                                                         # only if better: make it live
        promote(p["models"], vdir)

    # 16. model_runs row FIRST, then the cv predictions
    con = db.init_db(p["db"])
    cv = winner_rows
    db.insert_model_run(con, {
        "model_version": version, "trained_at": metadata["trained_at"], "model_type": model_type,
        "params_json": json.dumps(metadata["params"]),
        "train_months": ",".join(metadata["months"]["train"]), "test_month": ",".join(metadata["months"]["test"]),
        "train_mae": float(folds_summary["train_MAE"].mean()), "cv_mae": float(errors(cv).mean()),
        "cv_naive_mae": float((cv["target_balance"] - cv["pred_naive"]).abs().mean()),
        "cv_drift_mae": float((cv["target_balance"] - cv["pred_drift"]).abs().mean()),
        "promoted": int(promoted)})
    db.upsert_predictions(con, [{
        "account_id": r.account_id, "target_month": r.month.strftime("%Y-%m"),
        "balance_lag1": r.balance_lag1, "predicted_change": r.pred - r.balance_lag1,
        "predicted_balance": r.pred, "drift_balance": r.pred_drift, "model_used": model_type,
        "model_version": version, "split": "cv", "actual_balance": r.target_balance, "features_json": None}
        for r in cv.itertuples()])

    # 17. summary
    summary = comparison.groupby("model")[["train_MAE", "val_MAE"]].mean().sort_values("val_MAE")
    print("\n=== walk-forward folds (pooled) ===\n" + summary.round(1).to_string())
    print(f"\nhgb vs drift: {gain:+.2f} (95%: {lo:.2f} to {hi:.2f}) -> promoted model type: {model_type}")
    print(f"model_version {version} | saved to {vdir} | live model now: {'this one' if promoted else 'the old one (kept)'}")
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--trials", type=int, default=None, help="quick run, e.g. 5")
    a = ap.parse_args()
    main(a.config, a.trials)
