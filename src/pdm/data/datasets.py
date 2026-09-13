"""Dataset adapters: each maps a dataset config to (feature_df, feature_columns) plus a
train/val split, so pdm.training.train.run_training stays dataset-agnostic. Add a new
dataset type by registering another adapter here — the rest of the training pipeline
(fit, log to MLflow, validation gate) does not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pdm.data.bearing_features import build_bearing_feature_matrix
from pdm.data.bearing_features import feature_columns as bearing_columns
from pdm.data.cmapss import load_train as load_cmapss_train
from pdm.data.features import build_feature_matrix as build_cmapss_matrix
from pdm.data.features import cap_rul
from pdm.data.features import feature_columns as cmapss_columns


def _cmapss_load(raw_dir, dataset_cfg: dict, features_cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = load_cmapss_train(raw_dir, subset=dataset_cfg["subset"])
    rul_cap = dataset_cfg.get("rul_cap")
    if rul_cap is not None:
        df["rul"] = cap_rul(df["rul"], cap=rul_cap)

    sensors = features_cfg["sensor_columns"]
    windows = features_cfg["rolling_windows"]
    primary_window = max(windows)

    feature_df = build_cmapss_matrix(df, sensors, windows, primary_window)
    feature_df["rul"] = df["rul"].values
    return feature_df, cmapss_columns(sensors, primary_window)


def _cmapss_split(
    df: pd.DataFrame, val_split: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware random split: each C-MAPSS engine (unit_number) is a group, so an
    engine's cycles never straddle both train and validation."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    return next(splitter.split(df, groups=df["unit_number"]))


def _bearing_load(raw_dir, dataset_cfg: dict, features_cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    feature_df = build_bearing_feature_matrix(
        raw_dir,
        channels=features_cfg["channels"],
        failure_channel=features_cfg["failure_channel"],
    )
    rul_cap = dataset_cfg.get("rul_cap")
    if rul_cap is not None:
        feature_df["rul"] = cap_rul(feature_df["rul"], cap=rul_cap)
    return feature_df, bearing_columns(features_cfg["channels"])


def _bearing_split(
    df: pd.DataFrame, val_split: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random split across the full RUL range — NOT a temporal tail holdout.

    An earlier version held out the chronological tail (most recent snapshots) for
    validation, reasoning that validating on a run's unseen future avoids leaking
    degradation state. In production that measurably backfired: tree-based regressors
    cannot predict outside the range of targets they were trained on, so permanently
    withholding every low-RUL (near-failure) example from training left the model
    structurally unable to ever predict low RUL — on ANY input, not just the tail it
    was validated on. Trained this way on the real 2nd_test data (RUL 0-983), holding
    out the last 20% meant training only ever saw RUL 197-983; scored against the real,
    genuinely separate 3rd_test holdout, predictions never dropped below ~242 even at
    the true failure point (true RUL=0) — F2 and precision were exactly 0.0, not just
    low, because the classification threshold (RUL<=5) was structurally unreachable.

    Now that a genuinely separate physical run (e.g. 3rd_test, `data/holdout/
    holdout_v1.csv`) provides the real generalization check, this split's only job is
    approximate internal model-selection/sanity-checking — it no longer needs to be
    leak-proof against autocorrelated neighboring snapshots. A random split lets
    training see the full RUL range down to 0, which is a precondition for the model
    being able to predict near-failure at all.
    """
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(len(df))
    n_val = max(1, int(round(len(df) * val_split)))
    return shuffled[n_val:], shuffled[:n_val]


_ADAPTERS = {
    "cmapss": {"load": _cmapss_load, "split": _cmapss_split},
    "ims_bearing": {"load": _bearing_load, "split": _bearing_split},
}


def get_adapter(dataset_type: str) -> dict:
    if dataset_type not in _ADAPTERS:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}. Known: {list(_ADAPTERS)}")
    return _ADAPTERS[dataset_type]
