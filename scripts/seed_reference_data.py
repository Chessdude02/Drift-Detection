"""Writes the Evidently reference dataset (engineered features from the training set)
to config/drift.yaml's reference.path, so the drift-check job has a baseline to compare against.

Usage:
    python scripts/seed_reference_data.py --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.cmapss import load_train  # noqa: E402
from pdm.data.features import build_feature_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the drift-reference dataset from training data."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--training-config", default="training.yaml")
    parser.add_argument("--drift-config", default="drift.yaml")
    args = parser.parse_args()

    training_cfg = load_yaml(args.training_config)
    drift_cfg = load_yaml(args.drift_config)

    df = load_train(Path(args.raw_dir), subset=training_cfg["dataset"]["subset"])
    sensors = training_cfg["features"]["sensor_columns"]
    windows = training_cfg["features"]["rolling_windows"]
    primary_window = max(windows)

    feature_df = build_feature_matrix(df, sensors, windows, primary_window)

    out_path = Path(drift_cfg["reference"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(feature_df)} reference rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
