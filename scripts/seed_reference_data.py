"""Writes the drift-check reference dataset to config/drift.yaml's reference.path: the
engineered training features plus `predicted_rul`, an out-of-fold RUL prediction per row
(pdm.drift.life_stage.add_oof_predictions) that the drift check uses to match the
reference's life-stage mix to each window's.

Usage:
    python scripts/seed_reference_data.py --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.datasets import get_adapter  # noqa: E402
from pdm.drift.life_stage import add_oof_predictions  # noqa: E402


def build_reference(raw_dir: Path, training_cfg: dict):
    dataset_cfg = training_cfg["dataset"]
    adapter = get_adapter(dataset_cfg.get("type", "cmapss"))
    feature_df, cols = adapter["load"](raw_dir, dataset_cfg, training_cfg["features"])
    return add_oof_predictions(feature_df, cols, training_cfg["model"])


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

    reference = build_reference(Path(args.raw_dir), training_cfg)

    out_path = Path(drift_cfg["reference"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(out_path, index=False)
    print(f"Wrote {len(reference)} reference rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
