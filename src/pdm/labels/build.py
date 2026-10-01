"""Builds new training labels by joining logged predictions with what happened to each
engine (pdm.labels.outcomes), and writes them as a versioned dataset.

For each engine, its outcome events split its readings into lives: a reading at cycle c
belongs to the first event at or after c. Then:

- life ended in a failure at F: label = min(F - c, rul_cap), exact. `true_rul` = F - c
  (uncapped) is kept so the maintenance-decision scoring can use these engines too.
- life ended in maintenance at M (never failed): the true RUL is only known to exceed
  M - c. Because training caps RUL at `rul_cap`, a reading with M - c >= rul_cap still
  has an EXACT capped label (rul_cap). Readings closer to the maintenance than that are
  dropped - their capped label is unknown, and guessing would teach the model that
  engines die when they are maintained.
- no event yet (still running), no cycle logged, or missing a feature column: dropped
  and counted.

The output directory holds labels.parquet and manifest.json (counts, sha256, sources),
so a training run can record exactly which labels it used.

Usage:
    python -m pdm.labels.build --out data/labels
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.labels.outcomes import OutcomeStore
from pdm.serving.inference_log import InferenceLog

logger = logging.getLogger(__name__)

LABEL_FILE = "labels.parquet"
MANIFEST_FILE = "manifest.json"
# Label rows get synthetic unit numbers far above any C-MAPSS unit, so group splits
# keep each engine life on one side and parquet columns stay integer.
LABEL_UNIT_OFFSET = 1_000_000


def build_labels(
    inference_rows: list[dict], events: list[dict], feature_cols: list[str], rul_cap: float
) -> tuple[pd.DataFrame, dict]:
    stats = {
        "rows_seen": len(inference_rows),
        "dropped_shadow": 0,
        "dropped_no_asset_or_cycle": 0,
        "dropped_no_outcome_yet": 0,
        "dropped_censored_below_cap": 0,
        "dropped_missing_features": 0,
        "dropped_duplicate": 0,
    }
    by_asset: dict[str, list[dict]] = {}
    for e in events:
        by_asset.setdefault(e["asset_id"], []).append(e)
    for evs in by_asset.values():
        evs.sort(key=lambda e: (e["cycle"], e["event_type"] != "failure"))

    out = []
    seen = set()
    for row in inference_rows:  # newest first, so the first copy of (asset, cycle) wins
        if row.get("shadow"):
            stats["dropped_shadow"] += 1
            continue
        asset, cycle = row.get("asset_id"), row.get("cycle")
        if not asset or cycle is None:
            stats["dropped_no_asset_or_cycle"] += 1
            continue
        if (asset, cycle) in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add((asset, cycle))
        event = next((e for e in by_asset.get(asset, []) if e["cycle"] >= cycle), None)
        if event is None:
            stats["dropped_no_outcome_yet"] += 1
            continue
        if any(c not in row["features"] for c in feature_cols):
            stats["dropped_missing_features"] += 1
            continue
        remaining = event["cycle"] - cycle
        if event["event_type"] == "failure":
            label, true_rul, kind = min(remaining, rul_cap), float(remaining), "failure"
        elif remaining >= rul_cap:
            label, true_rul, kind = rul_cap, np.nan, "censored_at_cap"
        else:
            stats["dropped_censored_below_cap"] += 1
            continue
        out.append(
            {
                **{c: float(row["features"][c]) for c in feature_cols},
                "asset_id": asset,
                "life_end_cycle": int(event["cycle"]),
                "time_in_cycles": int(cycle),
                "rul": float(label),
                "true_rul": true_rul,
                "label_kind": kind,
                "model_version_at_prediction": row.get("model_version"),
            }
        )

    df = pd.DataFrame(out)
    if not df.empty:
        lives = df[["asset_id", "life_end_cycle"]].drop_duplicates().reset_index(drop=True)
        lives["unit_number"] = LABEL_UNIT_OFFSET + np.arange(len(lives))
        df = df.merge(lives, on=["asset_id", "life_end_cycle"]).sort_values(
            ["unit_number", "time_in_cycles"]
        )
        df = df.reset_index(drop=True)
    stats.update(
        {
            "rows_labelled": len(df),
            "rows_failure": int((df["label_kind"] == "failure").sum()) if len(df) else 0,
            "rows_censored_at_cap": (
                int((df["label_kind"] == "censored_at_cap").sum()) if len(df) else 0
            ),
            "engine_lives": int(df["unit_number"].nunique()) if len(df) else 0,
            "assets": int(df["asset_id"].nunique()) if len(df) else 0,
        }
    )
    return df, stats


def write_label_dataset(df: pd.DataFrame, stats: dict, out_dir: Path, sources: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LABEL_FILE
    df.to_parquet(path, index=False)
    manifest = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stats": stats,
        "sources": sources,
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


def build_and_write(
    inference_log_db: str, outcome_db: str, out_dir: Path, feature_cols: list[str], rul_cap
) -> dict:
    store = OutcomeStore(outcome_db)
    events = store.events()
    assets = sorted({e["asset_id"] for e in events})
    rows = InferenceLog(inference_log_db).read_for_assets(assets) if assets else []
    df, stats = build_labels(rows, events, feature_cols, rul_cap)
    manifest = write_label_dataset(
        df,
        stats,
        out_dir,
        {"inference_log_db": inference_log_db, "outcome_db": outcome_db, "events": len(events)},
    )
    logger.info("Built %d label rows: %s", len(df), stats)
    return manifest


def load_label_dataset(label_dir: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label rows ready to append to training data, plus the manifest. Empty frame if
    the directory holds no dataset yet."""
    path = Path(label_dir) / LABEL_FILE
    if not path.exists():
        return pd.DataFrame(), {}
    manifest = json.loads((Path(label_dir) / MANIFEST_FILE).read_text())
    df = pd.read_parquet(path)
    if df.empty:
        return df, manifest
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Label dataset lacks feature columns {missing}; it was built for a different "
            "feature set. Rebuild labels (logged readings only hold the features that "
            "were served at the time)."
        )
    return df, manifest


def main() -> int:
    from pdm.common.config import get_settings, load_yaml
    from pdm.common.logging import setup_logging
    from pdm.data.features import columns_from_config

    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build training labels from outcomes.")
    parser.add_argument("--config-name", default="training.yaml")
    parser.add_argument("--inference-log-db", default=settings.inference_log_db)
    parser.add_argument("--outcome-db", default=settings.outcome_db)
    parser.add_argument("--out", default=settings.labels_dir)
    args = parser.parse_args()

    cfg = load_yaml(args.config_name)
    features = cfg["features"]
    cols = columns_from_config(features)
    manifest = build_and_write(
        args.inference_log_db, args.outcome_db, Path(args.out), cols, cfg["dataset"]["rul_cap"]
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
