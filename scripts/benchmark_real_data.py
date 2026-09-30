"""Benchmark the pipeline against the real NASA C-MAPSS data and write the numbers to
reports/benchmark_cmapss.json.

Two things are measured:

1. Model accuracy. For each subset (FD001-FD004) the model is trained exactly the way
   pdm.training.train does it (same adapter, features, split and params), then scored on
   the official NASA test set: the last observed cycle of every test engine against
   RUL_FD00x.txt. This is the number the C-MAPSS literature reports. Repeated over
   several seeds because the train/val split only has ~20 validation engines.

2. Drift detector behaviour. pdm.drift.run_drift_check.compute_drift_score is run on
   500-row windows (config/drift.yaml's lookback_rows) against the FD001 reference, for
   scenarios with and without a real distribution shift, and the share of windows that
   cross config/drift.yaml's threshold is reported.

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
from pdm.drift.run_drift_check import compute_drift_score  # noqa: E402
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
    features_cfg = config["features"]
    sensors = features_cfg["sensor_columns"]
    windows = features_cfg["rolling_windows"]
    pw = max(windows)
    cols = drift_cfg["drift"]["columns"]
    threshold = drift_cfg["drift"]["threshold"]
    n = drift_cfg["current_window"]["lookback_rows"]

    def feats(df):
        return build_feature_matrix(df, sensors, windows, pw)

    fd001 = load_train(raw_dir, "FD001")
    units = np.array(sorted(fd001["unit_number"].unique()))
    rng = np.random.default_rng(42)
    ref_units = rng.choice(units, size=80, replace=False)
    held_units = np.setdiff1d(units, ref_units)

    fd001_feats = feats(fd001)
    fd001_feats["rul"] = fd001.sort_values(["unit_number", "time_in_cycles"])["rul"].values
    reference = fd001_feats[fd001_feats["unit_number"].isin(ref_units)]
    held = fd001_feats[fd001_feats["unit_number"].isin(held_units)]

    test001, _ = load_test(raw_dir, "FD001")

    # Production-like window: consecutive rows from 5 unseen FD001 engines.
    def five_engine_window(t):
        u = np.random.default_rng(t).choice(held["unit_number"].unique(), 5, replace=False)
        return held[held["unit_number"].isin(u)].tail(n)

    scenarios = {
        "control_same_engines_as_reference": reference,
        "no_drift_heldout_fd001_engines": held,
        "no_drift_5_unseen_fd001_engines": five_engine_window,
        "fd001_official_test_set": feats(test001),
        "fd001_heldout_near_failure_rul_le_30": held[held["rul"] <= 30],
        "fd003_new_fault_mode": feats(load_train(raw_dir, "FD003")),
        "fd002_six_operating_conditions": feats(load_train(raw_dir, "FD002")),
    }

    out = {
        "threshold": threshold,
        "window_rows": n,
        "trials": DRIFT_TRIALS,
        "reference_rows": int(len(reference)),
        "scenarios": {},
    }
    for name, pool in scenarios.items():
        scores = []
        for t in range(DRIFT_TRIALS):
            if callable(pool):
                window = pool(t)
            else:
                window = pool.sample(n=min(n, len(pool)), random_state=t)
            scores.append(compute_drift_score(reference, window, cols)["share_of_drifted_columns"])
        scores = np.array(scores)
        out["scenarios"][name] = {
            "pool_rows": None if callable(pool) else int(len(pool)),
            "mean_drift_share": float(scores.mean()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "retrain_trigger_rate": float((scores > threshold).mean()),
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
            f"drift {name}: mean_share={s['mean_drift_share']:.2f} "
            f"trigger_rate={s['retrain_trigger_rate']:.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
