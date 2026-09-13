"""Builds the fixed, versioned holdout set used for staging/production comparison.

This is a one-time (per version), deliberate action — NOT something the training
pipeline or CI ever regenerates automatically. The resulting CSV must come from a time
range that was never used to train any model version, and once built, `holdout_vN.csv`
is frozen: every future candidate-vs-production comparison scores against the exact same
rows, so F2/precision/PR-AUC deltas are meaningful across model versions and time.

The output stores raw features + `rul` (ground truth remaining useful life) but NOT a
precomputed binary failure label — the failure/healthy threshold (`failure_horizon`) is
a scoring-time config value (config/evaluation.yaml), not baked into the frozen holdout,
so tuning that horizon later doesn't require rebuilding the holdout file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdm.data.bearing_features import build_bearing_feature_matrix, feature_columns


def build_holdout_dataframe(
    raw_dir: Path | str, channels: list[str], failure_channel: str
) -> pd.DataFrame:
    """Feature-engineers every snapshot in `raw_dir` into one fixed holdout row each.

    Callers are responsible for pointing `raw_dir` at data reserved for holdout use only
    (e.g. the most recent, never-trained-on time range of a run-to-failure experiment) —
    see data/README.md's "Holdout set" section.
    """
    df = build_bearing_feature_matrix(raw_dir, channels=channels, failure_channel=failure_channel)
    cols = ["snapshot_index", "timestamp", *feature_columns(channels), "rul"]
    return df[cols]
