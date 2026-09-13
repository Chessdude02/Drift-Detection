"""Integration test: runs the full training pipeline (feature extraction -> fit ->
MLflow logging/registration -> validation gate) end-to-end against a small (~2%),
real-format sample of the NASA IMS Bearing run-to-failure dataset.

This intentionally does NOT assert real-world accuracy — the fixture is 24 snapshots
of 300 samples each vs. the real dataset's ~984 snapshots of 20480 samples, and the
validation split is the run's final (hardest-to-extrapolate) snapshots, so high accuracy
isn't a meaningful target here. The bar is "did the pipeline run correctly end-to-end and
produce a usable artifact": it completes without error, logs a real model artifact and
metrics to MLflow, and the holdout RMSE is finite and within a generous sanity ceiling
(the full RUL range for this fixture) rather than exploding or NaN-ing out.
"""

from __future__ import annotations

import math

import mlflow
import pytest

from pdm.training.train import run_training

pytestmark = pytest.mark.integration

CHANNELS = ["bearing1", "bearing2", "bearing3", "bearing4"]

# 24 fixture snapshots -> RUL ranges 0..23; a completely untrained/broken pipeline would
# produce errors far outside that range, so this ceiling is a correctness floor, not an
# accuracy target (see module docstring).
RUL_RANGE = 23
SANITY_RMSE_CEILING = float(RUL_RANGE)

BEARING_CONFIG = {
    "dataset": {"type": "ims_bearing", "rul_cap": None},
    "features": {"channels": CHANNELS, "failure_channel": "bearing4"},
    "model": {
        "algorithm": "lightgbm",
        "params": {
            "n_estimators": 20,
            "learning_rate": 0.2,
            "max_depth": 3,
            "num_leaves": 7,
            "random_state": 42,
            "min_child_samples": 2,
        },
        "val_split": 0.2,
    },
    "mlflow": {
        "experiment_name": "test_ims_bearing_rul",
        "registered_model_name": "test_ims_bearing_rul",
    },
    "evaluation": {"max_rmse": SANITY_RMSE_CEILING},
}


def test_bearing_pipeline_runs_end_to_end_and_logs_artifact(bearing_raw_dir, mlflow_tracking_uri):
    result = run_training(bearing_raw_dir, BEARING_CONFIG, register=True)

    # 1. Completed without error and returned a real MLflow run.
    assert "run_id" in result

    # 2. Metrics are sane: finite, non-negative, and below the correctness-floor ceiling
    # (not full accuracy — just "the model learned *something*, not garbage/NaN").
    assert math.isfinite(result["val_rmse"])
    assert result["val_rmse"] >= 0
    assert result["val_rmse"] <= SANITY_RMSE_CEILING
    assert result["gate_passed"] is True

    # 3. A real model artifact was produced and is loadable.
    client = mlflow.MlflowClient()
    artifact_paths = [a.path for a in client.list_artifacts(result["run_id"])]
    assert "model" in artifact_paths, "expected a logged 'model' artifact directory"

    loaded_model = mlflow.pyfunc.load_model(f"runs:/{result['run_id']}/model")
    assert loaded_model is not None

    # 4. Run metadata reflects the dataset actually used (catches config/adapter drift).
    run = client.get_run(result["run_id"])
    assert run.data.params["dataset_type"] == "ims_bearing"
    assert run.data.metrics["val_rmse"] == result["val_rmse"]


def test_bearing_pipeline_registers_a_loadable_model_version(bearing_raw_dir, mlflow_tracking_uri):
    result = run_training(bearing_raw_dir, BEARING_CONFIG, register=True)

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(
        f"name='{BEARING_CONFIG['mlflow']['registered_model_name']}'"
    )
    assert len(versions) >= 1
    assert any(v.run_id == result["run_id"] for v in versions)

    # result["model_version"] must be the actual registered version number, so CI can
    # immediately set image_tag on it without a separate registry lookup.
    assert result["model_version"] is not None
    matching = [v for v in versions if v.version == result["model_version"]]
    assert len(matching) == 1
    assert matching[0].run_id == result["run_id"]


def test_bearing_pipeline_model_version_is_none_without_register(
    bearing_raw_dir, mlflow_tracking_uri
):
    result = run_training(bearing_raw_dir, BEARING_CONFIG, register=False)
    assert result["model_version"] is None


def test_bearing_pipeline_sets_extra_tags_on_the_run(bearing_raw_dir, mlflow_tracking_uri):
    extra_tags = {"git_sha": "abc1234", "ci_run_id": "999"}
    result = run_training(bearing_raw_dir, BEARING_CONFIG, register=False, extra_tags=extra_tags)

    run = mlflow.MlflowClient().get_run(result["run_id"])
    assert run.data.tags["git_sha"] == "abc1234"
    assert run.data.tags["ci_run_id"] == "999"
