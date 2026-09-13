"""Per-snapshot statistical vibration features for the IMS Bearing dataset, and the
run-to-failure RUL construction (remaining snapshots until the end of the run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from pdm.data.ims_bearing import list_snapshot_files, load_snapshot

CHANNEL_STATS = ["rms", "kurtosis", "skewness", "peak_to_peak", "crest_factor", "std", "mean_abs"]


def compute_channel_features(signal: np.ndarray) -> dict[str, float]:
    """Time-domain vibration health indicators for one channel's samples in one snapshot.

    These are the standard IMS-bearing-analysis features: RMS and std track overall
    energy, kurtosis/crest-factor are sensitive to the impulsive shock signature bearing
    faults produce (a healthy bearing's vibration is close to Gaussian; a spalled race or
    ball produces sharp periodic impulses that spike both), skewness/peak-to-peak/mean-abs
    round out the standard vibration-health feature set.
    """
    rms = float(np.sqrt(np.mean(signal**2)))
    is_constant = np.std(signal) == 0
    # A zero-variance signal has no distribution shape to speak of; scipy's kurtosis/skew
    # return nan for it (division by a zero central moment), which would otherwise leak
    # NaNs into the feature matrix and downstream drift/model code. 0.0 (no excess
    # kurtosis/skew) is the reasonable convention for "flat, featureless signal".
    return {
        "rms": rms,
        "kurtosis": 0.0 if is_constant else float(kurtosis(signal)),
        "skewness": 0.0 if is_constant else float(skew(signal)),
        "peak_to_peak": float(signal.max() - signal.min()),
        "crest_factor": float(np.max(np.abs(signal)) / rms) if rms > 0 else 0.0,
        "std": float(np.std(signal)),
        "mean_abs": float(np.mean(np.abs(signal))),
    }


def feature_columns(channels: list[str]) -> list[str]:
    return [f"{ch}_{stat}" for ch in channels for stat in CHANNEL_STATS]


def build_bearing_feature_matrix(
    raw_dir: Path | str, channels: list[str], failure_channel: str
) -> pd.DataFrame:
    """One row per snapshot file (in chronological/positional order), with per-channel
    vibration features plus a `rul` column counting remaining snapshots until the run
    ends — the standard run-to-failure framing: the last recorded snapshot is treated as
    the failure point of `failure_channel`.
    """
    if failure_channel not in channels:
        raise ValueError(f"failure_channel {failure_channel!r} not in channels {channels}")

    files = list_snapshot_files(raw_dir)
    n_channels = len(channels)

    rows: list[dict[str, float | str]] = []
    for idx, path in enumerate(files):
        signal = load_snapshot(path, n_channels)
        row: dict[str, float | str] = {"snapshot_index": idx, "timestamp": path.name}
        for ch_idx, ch_name in enumerate(channels):
            row.update(
                {
                    f"{ch_name}_{stat}": value
                    for stat, value in compute_channel_features(signal[:, ch_idx]).items()
                }
            )
        rows.append(row)

    # Already in chronological order (list_snapshot_files sorts by filename), so
    # positional (.iloc) order equals temporal order for downstream train/val splitting.
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["rul"] = (len(df) - 1) - df["snapshot_index"]
    return df
