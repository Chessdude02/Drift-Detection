from __future__ import annotations

import pytest

from pdm.data.cmapss import SENSOR_COLUMNS
from pdm.training.train import run_training

pytestmark = pytest.mark.integration

TINY_CONFIG = {
    "dataset": {"type": "cmapss", "subset": "FD001", "rul_cap": 125},
    "features": {"sensor_columns": SENSOR_COLUMNS[:4], "rolling_windows": [5]},
    "model": {
        "algorithm": "lightgbm",
        "params": {
            "n_estimators": 5,
            "learning_rate": 0.3,
            "max_depth": 3,
            "num_leaves": 7,
            "random_state": 42,
            "min_child_samples": 1,
        },
        "val_split": 0.5,
    },
    "mlflow": {"experiment_name": "test_cmapss_rul", "registered_model_name": "test_cmapss_rul"},
    "evaluation": {"max_rmse": 10_000.0},  # generous so the smoke test always passes the gate
}


def test_run_training_end_to_end_logs_to_mlflow(raw_data_dir, mlflow_tracking_uri):
    result = run_training(raw_data_dir, TINY_CONFIG, register=True)

    assert "run_id" in result
    assert result["val_rmse"] >= 0
    assert result["gate_passed"] is True

    import mlflow

    run = mlflow.get_run(result["run_id"])
    assert run.data.metrics["val_rmse"] == result["val_rmse"]
    assert run.data.params["algorithm"] == "lightgbm"

    client = mlflow.MlflowClient()
    artifact_paths = [a.path for a in client.list_artifacts(result["run_id"])]
    assert "model" in artifact_paths, "expected a logged 'model' artifact directory"
