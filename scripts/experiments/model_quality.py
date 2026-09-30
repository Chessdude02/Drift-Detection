"""Model-quality experiment (docs/decisions.md D-26): does any cheap change beat the
current model on the numbers the gate uses, and does seed averaging shrink the
seed-to-seed noise that makes the gate unreliable (D-14)?

Variants, each trained 5 times (seeds 42, 0, 1, 2, 3) exactly as training does
(pdm.training.train.fit_bundle: frozen holdout excluded, interval, threshold chosen on
validation engines), then scored like the gate (pdm.evaluation.champion_challenger
.score_model):
- baseline:  config/training.yaml as is
- ensemble5: point model = average of 5 LightGBM seeds
- tuned_a/b/c: three hyperparameter sets. Which one "wins" is decided on INTERNAL
  VALIDATION RMSE only; NASA test numbers are reported for all but never used to choose.
- trend:     adds 14 trend features (5-cycle minus 20-cycle rolling mean per sensor).
  Changes the API (clients would have to send them), so it must win clearly to be worth it.
- *_cv variants: cross-validated calibration (see train.py _fit_bundle_cv).
  Here the seed changes which engines share a fold, so the spread across seeds shows how
  sensitive calibration still is to that assignment.

Reported per variant: mean and std over seeds of internal val RMSE, NASA test RMSE,
interval coverage/width, holdout cost per engine, holdout failures; and single-row
predict latency. Writes reports/model_quality.json. ~15 minutes.

Usage (from the repo root):
    python scripts/experiments/model_quality.py --raw-dir data/raw
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from pdm.common.config import load_yaml
from pdm.data.cmapss import load_test, load_train
from pdm.data.datasets import _cmapss_split, load_holdout_units
from pdm.data.features import add_rolling_features, cap_rul, feature_columns
from pdm.evaluation.champion_challenger import score_model
from pdm.training.train import fit_bundle

SEEDS = [42, 0, 1, 2, 3]
TUNED = {
    "tuned_a": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
    "tuned_b": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 100,
    },
    "tuned_c": {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
    },
}


def frames(raw_dir: Path, cfg: dict, trend: bool):
    """Training rows (holdout engines removed), holdout rows, and the NASA test set's
    last cycle per engine, with the same feature builder for all three."""
    sensors = cfg["features"]["sensor_columns"]
    windows = sorted(set(cfg["features"]["rolling_windows"]) | {5, 20})
    base_cols = feature_columns(sensors, 20)
    trend_cols = [f"{s}_trend" for s in sensors]
    cols = base_cols + (trend_cols if trend else [])

    def build(df):
        f = add_rolling_features(df, sensors, windows)
        for s in sensors:
            f[f"{s}_trend"] = f[f"{s}_roll_mean_5"] - f[f"{s}_roll_mean_20"]
        return f

    train = build(load_train(raw_dir, cfg["dataset"]["subset"]))
    train["true_rul"] = train["rul"]
    train["rul"] = cap_rul(train["rul"], cfg["dataset"]["rul_cap"])
    holdout_units = load_holdout_units(cfg["dataset"])
    training = train[~train["unit_number"].isin(holdout_units)].reset_index(drop=True)
    holdout = train[train["unit_number"].isin(holdout_units)].reset_index(drop=True)

    test_df, true_rul = load_test(raw_dir, cfg["dataset"]["subset"])
    last = build(test_df).groupby("unit_number").tail(1).set_index("unit_number").sort_index()
    y_test = cap_rul(true_rul.loc[last.index], cfg["dataset"]["rul_cap"])
    return training, {"holdout": holdout, "test_last": last, "y_test": y_test, "cols": cols}


class Served:
    """What the gate sees: .predict returns the bundle's output frame."""

    def __init__(self, bundle):
        self.bundle = bundle

    def predict(self, X):
        return self.bundle.predict_frame(X)


def latency_ms(bundle, row, n=200) -> float:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        for _ in range(5):
            bundle.predict_frame(row)
        start = time.perf_counter()
        for _ in range(n):
            bundle.predict_frame(row)
        return (time.perf_counter() - start) / n * 1000


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports/model_quality.json")
    parser.add_argument("--variants", nargs="*", help="Run only these variants")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    base_cfg = load_yaml("training.yaml")
    base_cfg["labels"] = {"enabled": False}
    decision_cfg = load_yaml(base_cfg["decision_config"])

    variants = {"baseline": ({}, False, None), "ensemble5": ({}, False, 5)}
    variants.update({name: (params, False, None) for name, params in TUNED.items()})
    variants["trend"] = ({}, True, None)
    # Cross-validated calibration (train.py _fit_bundle_cv): interval and threshold set
    # from all training engines' out-of-fold predictions, final model on all of them.
    variants["baseline_cv"] = ({"calibration": "cross_validation"}, False, None)
    variants["tuned_a_cv"] = ({**TUNED["tuned_a"], "calibration": "cross_validation"}, False, None)
    variants["tuned_c_cv"] = ({**TUNED["tuned_c"], "calibration": "cross_validation"}, False, None)
    variants["trend_cv"] = ({"calibration": "cross_validation"}, True, None)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    cache = {}
    results = {}
    for name, (params, trend, n_ens) in variants.items():
        if trend not in cache:
            cache[trend] = frames(raw_dir, base_cfg, trend)
        training, data = cache[trend]
        runs = []
        for seed in SEEDS:
            cfg = json.loads(json.dumps(base_cfg))
            model_params = {k: v for k, v in params.items() if k != "calibration"}
            cfg["model"]["params"].update({**model_params, "random_state": seed, "verbose": -1})
            if "calibration" in params:
                cfg["model"]["calibration"] = params["calibration"]
            if n_ens:
                cfg["model"]["ensemble_seeds"] = [seed * 100 + i for i in range(n_ens)]
            fitted = fit_bundle(training, data["cols"], cfg, _cmapss_split)
            bundle = fitted["bundle"]
            scores = score_model(Served(bundle), data, decision_cfg, bundle.maintenance_threshold)
            runs.append(
                {
                    "seed": seed,
                    "val_rmse": fitted["metrics"]["val_rmse"],
                    # internal (training-data-only) decision cost: the honest selection signal
                    "val_cost_per_engine": fitted["metrics"].get("val_cost_per_engine"),
                    **scores,
                }
            )
        summary = {}
        for key in (
            "val_rmse",
            "val_cost_per_engine",
            "test_rmse",
            "interval_coverage",
            "interval_mean_width",
            "cost_per_engine",
            "unplanned_failures",
            "mean_wasted_cycles",
        ):
            vals = np.array([r[key] for r in runs], dtype=float)
            summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        summary["predict_ms_single_row"] = latency_ms(
            bundle, data["test_last"][data["cols"]].head(1)
        )
        results[name] = {"summary": summary, "runs": runs}
        s = summary
        print(
            f"{name:10s} val {s['val_rmse']['mean']:5.2f}±{s['val_rmse']['std']:.2f}  "
            f"test {s['test_rmse']['mean']:5.2f}±{s['test_rmse']['std']:.2f}  "
            f"cov {s['interval_coverage']['mean']:.2f}  "
            f"width {s['interval_mean_width']['mean']:5.1f}  "
            f"cost {s['cost_per_engine']['mean']:6.2f}±{s['cost_per_engine']['std']:.2f}  "
            f"fail {s['unplanned_failures']['mean']:.1f}  {s['predict_ms_single_row']:.1f}ms",
            flush=True,
        )

    ran_tuned = [n for n in TUNED if n in results]
    tuned_by_val = (
        min(ran_tuned, key=lambda n: results[n]["summary"]["val_rmse"]["mean"])
        if ran_tuned
        else None
    )
    report = {"seeds": SEEDS, "tuned_chosen_on_validation": tuned_by_val, "variants": results}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"tuned set chosen on validation RMSE: {tuned_by_val}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
