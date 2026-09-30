"""Benchmark the pipeline against the real NASA C-MAPSS data and write the numbers to
reports/benchmark_cmapss.json.

Two things are measured:

1. Model accuracy. For each subset (FD001-FD004) the model is trained exactly the way
   pdm.training.train does it (same adapter, features, split and params), then scored on
   the official NASA test set: the last observed cycle of every test engine against
   RUL_FD00x.txt. This is the number the C-MAPSS literature reports. Repeated over
   several seeds because the train/val split only has ~20 validation engines.

2. Drift-check behaviour. pdm.drift.run_drift_check.evaluate_window is run on 500-row
   windows (config/drift.yaml's lookback_rows) against an FD001 reference, for scenarios
   with and without a real distribution shift, once with the original Evidently defaults
   and once with config/drift.yaml as shipped. Reported: how often the retrain trigger
   fires and how often the sensor-fault alert fires.

Usage:
    python scripts/benchmark_real_data.py --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.cmapss import load_test, load_train  # noqa: E402
from pdm.data.datasets import get_adapter  # noqa: E402
from pdm.data.features import build_feature_matrix  # noqa: E402
from pdm.drift.life_stage import add_oof_predictions  # noqa: E402
from pdm.drift.run_drift_check import evaluate_window  # noqa: E402
from pdm.training.evaluate import mae, rmse  # noqa: E402
from pdm.training.train import _fit_model  # noqa: E402

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 0, 1, 2, 3]
DRIFT_TRIALS = 20


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (d > 0) are penalised harder. Lower is better."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def test_last_cycle_features(raw_dir: Path, subset: str, features_cfg: dict):
    test_df, true_rul = load_test(raw_dir, subset)
    sensors = features_cfg["sensor_columns"]
    windows = features_cfg["rolling_windows"]
    feats = build_feature_matrix(test_df, sensors, windows, max(windows))
    last = feats.groupby("unit_number").tail(1).set_index("unit_number").sort_index()
    return last, true_rul.loc[last.index]


def evaluate_subset(raw_dir: Path, config: dict, subset: str) -> dict:
    dataset_cfg = {**config["dataset"], "subset": subset}
    features_cfg, model_cfg = config["features"], config["model"]
    rul_cap = dataset_cfg["rul_cap"]
    adapter = get_adapter("cmapss")
    feature_df, cols = adapter["load"](raw_dir, dataset_cfg, features_cfg)
    test_last, y_test = test_last_cycle_features(raw_dir, subset, features_cfg)
    y_test_capped = y_test.clip(upper=rul_cap)

    runs = []
    for seed in SEEDS:
        params = {**model_cfg["params"], "random_state": seed, "verbose": -1}
        tr, va = adapter["split"](feature_df, model_cfg["val_split"], seed)
        train_df, val_df = feature_df.iloc[tr], feature_df.iloc[va]
        model = _fit_model(model_cfg["algorithm"], params, train_df[cols], train_df["rul"])
        val_pred = model.predict(val_df[cols])
        test_pred = model.predict(test_last[cols])
        # Baseline: always predict the training-set mean RUL.
        base = np.full(len(y_test), train_df["rul"].mean())
        runs.append(
            {
                "seed": seed,
                "val_rmse_all_cycles": rmse(val_df["rul"], val_pred),
                "test_rmse": rmse(y_test_capped, test_pred),
                "test_rmse_uncapped_truth": rmse(y_test, test_pred),
                "test_mae": mae(y_test_capped, test_pred),
                "test_nasa_score": nasa_score(y_test_capped, test_pred),
                "baseline_test_rmse": rmse(y_test_capped, base),
            }
        )

    summary = {}
    for key in runs[0]:
        if key == "seed":
            continue
        vals = np.array([r[key] for r in runs])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    return {
        "subset": subset,
        "n_train_engines": int(feature_df["unit_number"].nunique()),
        "n_test_engines": int(len(y_test)),
        "summary": summary,
        "runs": runs,
    }


def drift_windows(raw_dir: Path, config: dict, drift_cfg: dict) -> dict:
    features_cfg, model_cfg = config["features"], config["model"]
    model_cfg = {**model_cfg, "params": {**model_cfg["params"], "verbose": -1}}
    sensors = features_cfg["sensor_columns"]
    windows = features_cfg["rolling_windows"]
    pw = max(windows)
    cols = drift_cfg["drift"]["columns"]
    threshold = drift_cfg["drift"]["threshold"]
    n = drift_cfg["current_window"]["lookback_rows"]

    def feats(df):
        return build_feature_matrix(df, sensors, windows, pw)

    fd001, feat_cols = get_adapter("cmapss")["load"](
        raw_dir, {**config["dataset"], "subset": "FD001"}, features_cfg
    )
    units = np.array(sorted(fd001["unit_number"].unique()))
    rng = np.random.default_rng(42)
    ref_units = rng.choice(units, size=80, replace=False)
    held = fd001[~fd001["unit_number"].isin(ref_units)]
    # Reference built exactly like scripts/seed_reference_data.py (out-of-fold predicted
    # RUL); a model trained on it plays the served model for the current windows.
    reference = add_oof_predictions(
        fd001[fd001["unit_number"].isin(ref_units)], feat_cols, model_cfg
    )
    served = _fit_model(
        model_cfg["algorithm"], model_cfg["params"], reference[feat_cols], reference["rul"]
    )

    test001, _ = load_test(raw_dir, "FD001")

    # Small-fleet windows: n rows drawn from exactly k unseen FD001 engines. Both checks
    # need ~5+ engines per window; fewer and one engine's quirks read as drift/faults.
    def k_engine_window(k):
        def make(t):
            u = np.random.default_rng(t).choice(held["unit_number"].unique(), k, replace=False)
            sub = held[held["unit_number"].isin(u)]
            return sub.sample(n=min(n, len(sub)), random_state=t)

        return make

    def one_sensor_offset(t):
        w = held.sample(n=n, random_state=t).copy()
        col = cols[t % len(cols)]
        w[col] = w[col] + reference[col].std()
        return w

    scenarios = {
        "control_same_engines_as_reference": reference,
        "no_drift_heldout_fd001_engines": held,
        "no_drift_1_unseen_fd001_engine": k_engine_window(1),
        "no_drift_2_unseen_fd001_engines": k_engine_window(2),
        "no_drift_3_unseen_fd001_engines": k_engine_window(3),
        "no_drift_5_unseen_fd001_engines": k_engine_window(5),
        "fd001_official_test_set": feats(test001),
        "fd001_heldout_near_failure_rul_le_30": held[held["rul"] <= 30],
        "fd003_new_fault_mode": feats(load_train(raw_dir, "FD003")),
        "fd002_six_operating_conditions": feats(load_train(raw_dir, "FD002")),
        "one_sensor_offset_1std": one_sensor_offset,
    }

    # "evidently_default" = the original check: Evidently's own per-column test against
    # the plain reference, no sensor check. "configured" = config/drift.yaml as shipped
    # (life-stage matching, calibrated threshold, sensor check), via evaluate_window.
    off = {"enabled": False}
    settings = {
        "evidently_default": {
            **drift_cfg,
            "drift": {**drift_cfg["drift"], "stattest": None, "stattest_threshold": None},
            "life_stage_matching": off,
            "sensor_check": off,
        },
        "configured": drift_cfg,
    }
    out = {
        "threshold": threshold,
        "window_rows": n,
        "trials": DRIFT_TRIALS,
        "reference_rows": int(len(reference)),
        "scenarios": {},
    }
    for name, pool in scenarios.items():
        out["scenarios"][name] = {"pool_rows": None if callable(pool) else int(len(pool))}
        for label, cfg in settings.items():
            shares, alerts = [], []
            for t in range(DRIFT_TRIALS):
                if callable(pool):
                    window = pool(t)
                else:
                    window = pool.sample(n=min(n, len(pool)), random_state=t)
                window = window.copy()
                window["prediction"] = served.predict(window[feat_cols])
                result = evaluate_window(reference, window, cols, cfg)
                shares.append(result["drift"]["share_of_drifted_columns"])
                alerts.append(bool(result["faulty_sensors"]))
            shares = np.array(shares)
            out["scenarios"][name][label] = {
                "mean_drift_share": float(shares.mean()),
                "retrain_trigger_rate": float((shares > threshold).mean()),
                "sensor_alert_rate": float(np.mean(alerts)),
            }
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports/benchmark_cmapss.json")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    config = load_yaml("training.yaml")
    drift_cfg = load_yaml("drift.yaml")

    result = {"model": [], "drift": None}
    for subset in SUBSETS:
        r = evaluate_subset(raw_dir, config, subset)
        s = r["summary"]
        print(
            f"{subset}: val_rmse={s['val_rmse_all_cycles']['mean']:.2f}"
            f"±{s['val_rmse_all_cycles']['std']:.2f}  "
            f"test_rmse={s['test_rmse']['mean']:.2f}±{s['test_rmse']['std']:.2f}  "
            f"test_mae={s['test_mae']['mean']:.2f}  nasa={s['test_nasa_score']['mean']:.0f}  "
            f"baseline_rmse={s['baseline_test_rmse']['mean']:.2f}"
        )
        result["model"].append(r)

    result["drift"] = drift_windows(raw_dir, config, drift_cfg)
    for name, s in result["drift"]["scenarios"].items():
        print(
            f"drift {name}: "
            f"default retrain={s['evidently_default']['retrain_trigger_rate']:.2f}  "
            f"configured retrain={s['configured']['retrain_trigger_rate']:.2f} "
            f"sensor_alert={s['configured']['sensor_alert_rate']:.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
