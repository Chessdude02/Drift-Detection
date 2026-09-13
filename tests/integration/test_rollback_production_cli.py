"""Integration test: invokes scripts/rollback_production.py exactly as the rollback
workflow will — as a subprocess, reading/writing the real MLflow tracking store.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from pdm.evaluation.registry import get_production_version, promote_version, set_image_tag

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "test_rollback_cli_model"


def _new_version(client: MlflowClient) -> str:
    mlflow.set_experiment("test_rollback_cli_candidates")
    with mlflow.start_run() as run:
        pass
    created = client.create_model_version(
        MODEL_NAME, source="file:///fake/model", run_id=run.info.run_id
    )
    return str(created.version)


def _run_cli() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/rollback_production.py", "--model-name", MODEL_NAME],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_rollback_reverts_to_tagged_version_and_outputs_its_image_tag(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)

    v1 = _new_version(client)
    set_image_tag(client, MODEL_NAME, v1, "sha-v1")
    promote_version(
        client, MODEL_NAME, v1, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}, "v1"
    )

    v2 = _new_version(client)
    set_image_tag(client, MODEL_NAME, v2, "sha-v2")
    promote_version(
        client, MODEL_NAME, v2, {"f2_score": 0.5, "precision": 0.55, "pr_auc": 0.65}, "v1"
    )

    result = _run_cli()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "sha-v1" in result.stdout
    prod = get_production_version(client, MODEL_NAME)
    assert str(prod.version) == v1


def test_rollback_with_nothing_to_roll_back_to_fails_clearly(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)

    v1 = _new_version(client)
    set_image_tag(client, MODEL_NAME, v1, "sha-v1")
    promote_version(
        client, MODEL_NAME, v1, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}, "v1"
    )

    result = _run_cli()

    assert result.returncode == 1
    assert "RUNBOOK.md" in result.stdout
    prod = get_production_version(client, MODEL_NAME)
    assert str(prod.version) == v1  # untouched
