"""Where the drift job gets its reference data.

`source: model` (default in config/drift.yaml): download the reference that was logged
with the current Production model's training run (pdm.training.train logs it under
drift_reference/). The reference then always matches the model being monitored, and
the drift job needs nothing but MLFLOW_TRACKING_URI - no file baked into its image or
mounted into its pod.

`source: file`: read reference.path (seeded by scripts/seed_reference_data.py), for
local runs or a model trained before references were logged.

There is deliberately no silent fallback from `model` to `file`: comparing live
traffic against another model's training data is exactly the mismatch this avoids, so
a missing artifact is an error.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

DRIFT_REFERENCE_ARTIFACT_DIR = "drift_reference"
DRIFT_REFERENCE_FILE = "reference.parquet"

logger = logging.getLogger(__name__)


def load_reference(reference_cfg: dict) -> tuple[pd.DataFrame, str]:
    """Returns (reference, description of where it came from)."""
    source = reference_cfg.get("source", "file")
    if source == "file":
        path = reference_cfg["path"]
        return pd.read_parquet(path), f"file {path}"
    if source != "model":
        raise ValueError(f"Unknown reference.source {source!r} (expected 'model' or 'file')")

    import mlflow
    from mlflow import MlflowClient

    from pdm.evaluation.registry import get_production_version

    name = reference_cfg["model_name"]
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    prod = get_production_version(client, name)
    if prod is None:
        raise RuntimeError(f"No Production version of {name!r}; nothing to monitor yet")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(
                prod.run_id, f"{DRIFT_REFERENCE_ARTIFACT_DIR}/{DRIFT_REFERENCE_FILE}", tmp
            )
        except Exception as exc:
            raise RuntimeError(
                f"{name} v{prod.version} (run {prod.run_id}) has no logged drift reference. "
                "Retrain with drift_reference.enabled, or set reference.source: file."
            ) from exc
        reference = pd.read_parquet(Path(local))
    return reference, f"{name} v{prod.version} (run {prod.run_id})"
