"""The drift reference travels with the model: training logs it, the drift job loads
the Production model's copy (pdm.drift.reference)."""

from __future__ import annotations

import pandas as pd
import pytest

from pdm.data.cmapss import SENSOR_COLUMNS
from pdm.drift.reference import load_reference
from pdm.training.train import run_training

pytestmark = pytest.mark.integration

CONFIG = {
    "dataset": {"type": "cmapss", "subset": "FD001", "rul_cap": 125},
    "features": {"sensor_columns": SENSOR_COLUMNS[:4], "rolling_windows": [5]},
    "model": {
        "algorithm": "lightgbm",
        "params": {"n_estimators": 5, "min_child_samples": 1, "random_state": 42, "verbose": -1},
        "val_split": 0.5,
        "intervals": {"enabled": True, "coverage": 0.9},
    },
    "decision_config": "decision.yaml",
    "drift_reference": {"enabled": True},
    "mlflow": {"experiment_name": "test_ref", "registered_model_name": "test_ref"},
    "evaluation": {"max_rmse": 10_000.0},
}


def test_reference_is_logged_with_the_model_and_loaded_from_production(
    raw_data_dir, mlflow_tracking_uri
):
    import mlflow

    result = run_training(raw_data_dir, CONFIG, register=True)
    client = mlflow.MlflowClient()
    cfg = {"source": "model", "model_name": "test_ref"}

    with pytest.raises(RuntimeError, match="No Production version"):
        load_reference(cfg)

    client.transition_model_version_stage("test_ref", result["model_version"], "Production")
    reference, source = load_reference(cfg)
    assert f"v{result['model_version']}" in source
    assert {"predicted_rul", "rul", "unit_number"} <= set(reference.columns)
    assert reference["predicted_rul"].notna().all()
    assert result["maintenance_threshold"] >= 0


def test_file_source_and_bad_source(tmp_path):
    path = tmp_path / "ref.parquet"
    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(path)
    reference, source = load_reference({"source": "file", "path": str(path)})
    assert len(reference) == 2 and str(path) in source
    with pytest.raises(ValueError):
        load_reference({"source": "s3"})
