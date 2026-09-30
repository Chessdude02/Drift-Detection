"""Calibrate the drift check's per-column threshold on real C-MAPSS data, then measure
how well the calibrated check separates normal operation from real shifts.

Why: with Evidently's default per-column test (normed Wasserstein distance, threshold
0.1) a window of *unseen but normal* FD001 engines is flagged as drift 90-100% of the
time (see README "Measured results"), because engine-to-engine variation alone exceeds
0.1. The retrain trigger then fires on nearly every check.

Method:
- FD001's 100 training engines are split into 5 folds of 20. For each fold, the other
  80 engines are the reference and the fold's engines are "unseen normal" traffic.
- The drift check fires when share_of_drifted_columns > drift.threshold (0.5), i.e.
  when more than half of the columns exceed the per-column threshold t. So for each
  window, the smallest t that would NOT fire is a single number (the k-th largest
  column distance, k = floor(n_cols * share) + 1). t is set to a high quantile of that
  number over normal windows from folds 0-2 only.
- Folds 3-4 are never used to pick t; false-alarm rates are reported on them.
- Detection rates are reported for FD003 (new fault mode), FD002 (six operating
  conditions), unseen FD001 engines near failure, and synthetic offsets added to
  unseen FD001 windows (all columns, or 3 of 14).
- The final numbers are re-checked through the real pdm.drift.run_drift_check
  .compute_drift_score (Evidently) with the calibrated threshold.

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
from scipy.stats import wasserstein_distance

sys.path.insert(0, "src")

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.cmapss import load_train  # noqa: E402
from pdm.data.features import build_feature_matrix  # noqa: E402
from pdm.drift.run_drift_check import compute_drift_score  # noqa: E402

N_FOLDS = 5
CALIBRATION_FOLDS = [0, 1, 2]
EVALUATION_FOLDS = [3, 4]
WINDOWS_PER_FOLD = 200
TARGET_QUANTILE = 0.99
EVIDENTLY_CHECK_WINDOWS = 20
OFFSETS_STD = [0.25, 0.5, 1.0, 2.0]


def normed_wasserstein(ref: np.ndarray, cur: np.ndarray) -> float:
    """Same formula as Evidently's 'wasserstein' stattest."""
    return wasserstein_distance(ref, cur) / max(np.std(ref), 0.001)


def window_distances(ref_df, window_df, cols) -> np.ndarray:
    return np.array([normed_wasserstein(ref_df[c].values, window_df[c].values) for c in cols])


def min_nonfiring_threshold(dists: np.ndarray, share: float) -> float:
    """Smallest per-column threshold at which share_of_drifted_columns <= share."""
    k = int(np.floor(len(dists) * share)) + 1  # columns that must drift for share > share
    return float(np.sort(dists)[::-1][k - 1])


def fire_rate(dists_list, t: float, share: float) -> float:
    return float(np.mean([np.mean(d >= t) > share for d in dists_list]))


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports/drift_calibration.json")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    features_cfg = load_yaml("training.yaml")["features"]
    drift_cfg = load_yaml("drift.yaml")
    cols = drift_cfg["drift"]["columns"]
    share = drift_cfg["drift"]["threshold"]
    n = drift_cfg["current_window"]["lookback_rows"]
    sensors, windows = features_cfg["sensor_columns"], features_cfg["rolling_windows"]

    def feats(subset):
        df = load_train(raw_dir, subset)
        f = build_feature_matrix(df, sensors, windows, max(windows))
        f["rul"] = df.sort_values(["unit_number", "time_in_cycles"])["rul"].values
        return f

    fd001, fd002, fd003 = feats("FD001"), feats("FD002"), feats("FD003")
    units = np.random.default_rng(42).permutation(fd001["unit_number"].unique())
    folds = np.array_split(units, N_FOLDS)

    def fleet_window(pool, rng):  # many engines, random life stages
        return pool.sample(n=min(n, len(pool)), random_state=int(rng.integers(1 << 31)))

    def small_fleet_window(pool, rng):  # 5 engines, all their cycles
        u = rng.choice(pool["unit_number"].unique(), 5, replace=False)
        sub = pool[pool["unit_number"].isin(u)]
        return sub.sample(n=min(n, len(sub)), random_state=int(rng.integers(1 << 31)))

    window_types = {
        "fleet_500_random_rows": fleet_window,
        "small_fleet_5_engines": small_fleet_window,
    }

    # Per-fold reference and normal-traffic distances.
    per_fold = []
    for k, fold_units in enumerate(folds):
        ref = fd001[~fd001["unit_number"].isin(fold_units)]
        held = fd001[fd001["unit_number"].isin(fold_units)]
        rng = np.random.default_rng(1000 + k)
        normal = {
            name: [window_distances(ref, fn(held, rng), cols) for _ in range(WINDOWS_PER_FOLD)]
            for name, fn in window_types.items()
        }
        per_fold.append({"ref": ref, "held": held, "normal": normal})

    # Calibrate on the harder (higher) of the two window types, calibration folds only.
    calib = [
        min_nonfiring_threshold(d, share)
        for k in CALIBRATION_FOLDS
        for name in window_types
        for d in per_fold[k]["normal"][name]
    ]
    t = float(np.round(np.quantile(calib, TARGET_QUANTILE), 2))
    default_t = 0.1

    def rates(dists_list):
        return {
            "calibrated": fire_rate(dists_list, t, share),
            "default_0.1": fire_rate(dists_list, default_t, share),
        }

    report = {
        "calibrated_threshold": t,
        "target_quantile": TARGET_QUANTILE,
        "drift_share_threshold": share,
        "window_rows": n,
        "false_alarm_rate_heldout_folds": {},
        "detection_rate_heldout_folds": {},
    }
    for name in window_types:
        dl = [d for k in EVALUATION_FOLDS for d in per_fold[k]["normal"][name]]
        report["false_alarm_rate_heldout_folds"][name] = rates(dl)

    rng = np.random.default_rng(7)
    shifted = {}
    for k in EVALUATION_FOLDS:
        ref, held = per_fold[k]["ref"], per_fold[k]["held"]
        near = held[held["rul"] <= 30]
        for _ in range(WINDOWS_PER_FOLD // 2):
            shifted.setdefault("fd003_new_fault_mode_fleet", []).append(
                window_distances(ref, fleet_window(fd003, rng), cols)
            )
            shifted.setdefault("fd002_six_operating_conditions_fleet", []).append(
                window_distances(ref, fleet_window(fd002, rng), cols)
            )
            shifted.setdefault("fd001_unseen_near_failure_rul_le_30", []).append(
                window_distances(ref, fleet_window(near, rng), cols)
            )
            base = fleet_window(held, rng)
            for off in OFFSETS_STD:
                for label, target in (("all_14", cols), ("3_of_14", cols[:3])):
                    w = base.copy()
                    for c in target:
                        w[c] = w[c] + off * ref[c].std()
                    shifted.setdefault(f"offset_{off}std_{label}_columns", []).append(
                        window_distances(ref, w, cols)
                    )
    for name, dl in shifted.items():
        report["detection_rate_heldout_folds"][name] = rates(dl)

    # Cross-check through the production code path (Evidently), fold 3.
    ref, held = per_fold[3]["ref"], per_fold[3]["held"]
    rng = np.random.default_rng(99)
    check = {}
    for name, pool in (
        ("fd001_unseen_fleet", held),
        ("fd002_fleet", fd002),
        ("fd003_fleet", fd003),
    ):
        scores = [
            compute_drift_score(
                ref, fleet_window(pool, rng), cols, stattest="wasserstein", stattest_threshold=t
            )["share_of_drifted_columns"]
            for _ in range(EVIDENTLY_CHECK_WINDOWS)
        ]
        check[name] = float(np.mean(np.array(scores) > share))
    report["evidently_crosscheck_fire_rate_fold3"] = check

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
