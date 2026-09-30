"""Calibrate both drift-check thresholds on real C-MAPSS data and measure how well the
calibrated checks separate normal operation (including fleets that are younger or older
than the reference) from real shifts and single-sensor faults.

The two thresholds in config/drift.yaml it produces:
- drift.stattest_threshold: per-column normed Wasserstein threshold for the retrain
  trigger, applied against the life-stage-matched reference (pdm.drift.life_stage).
- sensor_check.threshold: per-sensor residual threshold for the sensor-fault alert
  (pdm.drift.sensor_check).

Method:
- FD001's 100 training engines are split into 5 folds of 20. For each fold, the other 80
  engines are the reference (with out-of-fold predicted RUL, exactly as
  scripts/seed_reference_data.py builds it) and a model trained on them produces the
  "served" predictions for the fold's engines, as the live service would.
- The retrain trigger fires when more than drift.threshold (half) of the columns exceed
  the per-column threshold, so per window the smallest non-firing threshold is the k-th
  largest column distance. The sensor alert fires when any sensor exceeds its threshold,
  so per window that number is the largest sensor score.
- Each threshold is the 99th percentile of that number over NORMAL windows (a large
  fleet and a 5-engine fleet) from folds 0-2 only. Folds 3-4 are never used to pick
  thresholds; every rate reported comes from them.
- Distances are computed with the production functions; the retrain rates are
  re-checked through run_drift_check.evaluate_window (Evidently) at the end.

Usage:
    python scripts/calibrate_drift.py --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, "src")

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.cmapss import load_test, load_train  # noqa: E402
from pdm.data.datasets import get_adapter  # noqa: E402
from pdm.data.features import build_feature_matrix, cap_rul  # noqa: E402
from pdm.drift.life_stage import add_oof_predictions, match_life_stage  # noqa: E402
from pdm.drift.run_drift_check import evaluate_window  # noqa: E402
from pdm.drift.sensor_check import sensor_fault_scores  # noqa: E402
from pdm.training.train import _fit_model  # noqa: E402

N_FOLDS = 5
CALIBRATION_FOLDS = [0, 1, 2]
EVALUATION_FOLDS = [3, 4]
WINDOWS_PER_FOLD = 60
TARGET_QUANTILE = 0.99
EVIDENTLY_CHECK_WINDOWS = 10
NORMAL = ["normal_large_fleet", "normal_small_fleet_5_engines"]


def normed_wasserstein(ref: np.ndarray, cur: np.ndarray) -> float:
    """Same formula as Evidently's 'wasserstein' stattest."""
    return wasserstein_distance(ref, cur) / max(np.std(ref), 0.001)


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports/drift_calibration.json")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    training_cfg = load_yaml("training.yaml")
    drift_cfg = load_yaml("drift.yaml")
    cols = drift_cfg["drift"]["columns"]
    share = drift_cfg["drift"]["threshold"]
    n = drift_cfg["current_window"]["lookback_rows"]
    matching = drift_cfg["life_stage_matching"]
    features_cfg, model_cfg = training_cfg["features"], training_cfg["model"]
    params = {**model_cfg["params"], "verbose": -1}
    model_cfg = {**model_cfg, "params": params}
    k_col = int(np.floor(len(cols) * share))  # index of the k-th largest distance

    adapter = get_adapter("cmapss")
    fd001, feat_cols = adapter["load"](
        raw_dir, {**training_cfg["dataset"], "subset": "FD001"}, features_cfg
    )
    fd001["life_frac"] = fd001["time_in_cycles"] / fd001.groupby("unit_number")[
        "time_in_cycles"
    ].transform("max")

    def raw_feats(df):
        sensors, windows = features_cfg["sensor_columns"], features_cfg["rolling_windows"]
        return build_feature_matrix(df, sensors, windows, max(windows))

    fd002 = raw_feats(load_train(raw_dir, "FD002"))
    fd003 = raw_feats(load_train(raw_dir, "FD003"))
    fd001_test = raw_feats(load_test(raw_dir, "FD001")[0])

    units = np.random.default_rng(42).permutation(fd001["unit_number"].unique())
    folds = np.array_split(units, N_FOLDS)

    def sample(pool, rng, rows=n):
        return pool.sample(n=min(rows, len(pool)), random_state=int(rng.integers(1 << 31)))

    def small_fleet(pool, rng):
        u = rng.choice(pool["unit_number"].unique(), 5, replace=False)
        return sample(pool[pool["unit_number"].isin(u)], rng)

    def young_fleet(pool, rng):  # every engine observed only up to a random point in life
        cut = pool["unit_number"].map(
            {u: rng.uniform(0.1, 0.9) for u in pool["unit_number"].unique()}
        )
        return sample(pool[pool["life_frac"] <= cut], rng)

    def scenarios(held, ref, rng, normal_only):
        yield "normal_large_fleet", sample(held, rng), None
        yield "normal_small_fleet_5_engines", small_fleet(held, rng), None
        if normal_only:
            return
        yield "ageing_young_fleet", young_fleet(held, rng), None
        yield "ageing_fd001_official_test_set", sample(fd001_test, rng), None
        yield "ageing_old_fleet_rul_le_30", sample(held[cap_rul(held["rul"]) <= 30], rng), None
        yield "shift_fd002_six_operating_conditions", sample(fd002, rng), None
        yield "shift_fd003_new_fault_mode", sample(fd003, rng), None
        base = sample(held, rng)
        col = cols[int(rng.integers(len(cols)))]
        for k in (0.5, 1.0, 2.0):
            w = base.copy()
            w[col] = w[col] + k * ref[col].std()
            yield f"sensor_offset_{k}std_one_sensor", w, col
        w = base.copy()
        w[col] = ref[col].median()
        yield "sensor_stuck_one_sensor", w, col

    records = []  # (fold, scenario, retrain_stat, sensor_stat, localized)
    fold_state = {}
    for k, fold_units in enumerate(folds):
        ref_rows = fd001[~fd001["unit_number"].isin(fold_units)]
        held = fd001[fd001["unit_number"].isin(fold_units)]
        ref = add_oof_predictions(ref_rows, feat_cols, model_cfg)
        served = _fit_model(model_cfg["algorithm"], params, ref[feat_cols], ref["rul"])
        fold_state[k] = (ref, held, served)
        rng = np.random.default_rng(1000 + k)
        normal_only = k in CALIBRATION_FOLDS
        for _ in range(WINDOWS_PER_FOLD):
            for name, w, broken in scenarios(held, ref, rng, normal_only):
                w = w.copy()
                w["prediction"] = served.predict(w[feat_cols])
                matched, _ = match_life_stage(
                    ref,
                    w["prediction"],
                    bin_width=matching["bin_width"],
                    max_rul=matching["max_rul"],
                    sample_rows=matching["sample_rows"],
                    seed=int(rng.integers(1 << 31)),
                )
                d = np.array([normed_wasserstein(matched[c].values, w[c].values) for c in cols])
                scores = sensor_fault_scores(ref, w, cols)
                top = max(scores, key=scores.get)
                records.append(
                    (k, name, float(np.sort(d)[::-1][k_col]), float(scores[top]), top == broken)
                )
        print(f"fold {k} done", flush=True)

    df = pd.DataFrame(
        records, columns=["fold", "scenario", "retrain_stat", "sensor_stat", "localized"]
    )
    calib = df[df["fold"].isin(CALIBRATION_FOLDS) & df["scenario"].isin(NORMAL)]
    t_retrain = float(np.ceil(np.quantile(calib["retrain_stat"], TARGET_QUANTILE) * 100) / 100)
    t_sensor = float(np.ceil(np.quantile(calib["sensor_stat"], TARGET_QUANTILE) * 100) / 100)

    ev = df[df["fold"].isin(EVALUATION_FOLDS)]
    rates = {}
    for name, g in ev.groupby("scenario", sort=False):
        sensor_fired = g["sensor_stat"] > t_sensor
        entry = {
            "windows": int(len(g)),
            "retrain_trigger_rate": float((g["retrain_stat"] > t_retrain).mean()),
            "sensor_alert_rate": float(sensor_fired.mean()),
        }
        if name.startswith("sensor_"):
            entry["alert_names_right_sensor_first"] = (
                float(g.loc[sensor_fired, "localized"].mean()) if sensor_fired.any() else None
            )
        rates[name] = entry

    # Cross-check the retrain rates through the production path (Evidently).
    config = {
        **drift_cfg,
        "drift": {**drift_cfg["drift"], "stattest_threshold": t_retrain},
        "sensor_check": {"enabled": True, "threshold": t_sensor},
    }
    ref, held, served = fold_state[EVALUATION_FOLDS[0]]
    rng = np.random.default_rng(99)
    crosscheck = {}
    for name, pool in (("normal_large_fleet", held), ("shift_fd003_new_fault_mode", fd003)):
        fired = []
        for _ in range(EVIDENTLY_CHECK_WINDOWS):
            w = sample(pool, rng).copy()
            w["prediction"] = served.predict(w[feat_cols])
            r = evaluate_window(ref, w, cols, config)
            fired.append(r["drift"]["share_of_drifted_columns"] > share)
        crosscheck[name] = float(np.mean(fired))

    report = {
        "thresholds": {"drift.stattest_threshold": t_retrain, "sensor_check.threshold": t_sensor},
        "target_quantile": TARGET_QUANTILE,
        "window_rows": n,
        "evaluation_folds_only": rates,
        "evidently_crosscheck_retrain_rate": crosscheck,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
