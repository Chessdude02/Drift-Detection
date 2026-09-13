"""Integration test: invokes scripts/promote_to_production.py exactly as the
staging->prod GitHub Actions workflow will — as a subprocess, reading/writing the real
MLflow tracking store — covering both the bootstrap path and the ordinary gated path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from pdm.evaluation.registry import get_production_version, set_image_tag

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "test_promote_cli_model"
STAGING_EXPERIMENT = "ims_bearing_rul_staging_eval"


def _make_candidate_version(client: MlflowClient, scores: dict[str, float]) -> str:
    # Explicit set_experiment before start_run: mlflow's fluent API caches the "active
    # experiment id" globally, which otherwise leaks a stale id from whichever test ran
    # previously in this process (each test gets its own tmp tracking store via the
    # mlflow_tracking_uri fixture, but that cache doesn't know the store changed).
    mlflow.set_experiment("test_promote_cli_candidates")
    with mlflow.start_run() as run:
        pass
    created = client.create_model_version(
        MODEL_NAME, source="file:///fake/model", run_id=run.info.run_id
    )
    version = str(created.version)
    set_image_tag(client, MODEL_NAME, version, f"sha-{version}")

    mlflow.set_experiment(STAGING_EXPERIMENT)
    with mlflow.start_run(run_name="staging_eval"):
        mlflow.set_tag("evaluation_stage", "staging")
        mlflow.set_tag("candidate_run_id", run.info.run_id)
        mlflow.set_tag("holdout_version", "v1")
        mlflow.log_metric("f2_score", scores["f2_score"])
        mlflow.log_metric("precision", scores["precision"])
        mlflow.log_metric("pr_auc", scores["pr_auc"])

    return version


def _run_cli(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "scripts/promote_to_production.py",
            "--model-name",
            MODEL_NAME,
            *extra_args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_first_promotion_without_confirm_bootstrap_is_rejected(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)
    version = _make_candidate_version(client, {"f2_score": 0.9, "precision": 0.9, "pr_auc": 0.9})

    result = _run_cli("--version", version)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "bootstrap_confirmed" in result.stdout
    assert get_production_version(client, MODEL_NAME) is None


def test_first_promotion_with_confirm_bootstrap_succeeds(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)
    version = _make_candidate_version(client, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6})

    result = _run_cli("--version", version, "--confirm-bootstrap")

    assert result.returncode == 0, result.stdout + result.stderr
    prod = get_production_version(client, MODEL_NAME)
    assert prod is not None
    assert str(prod.version) == version
    assert prod.tags["baseline_f2_score"] == "0.4"


def test_second_promotion_rejected_when_it_regresses_f2(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)

    v1 = _make_candidate_version(client, {"f2_score": 0.5, "precision": 0.5, "pr_auc": 0.6})
    assert _run_cli("--version", v1, "--confirm-bootstrap").returncode == 0

    v2 = _make_candidate_version(client, {"f2_score": 0.1, "precision": 0.9, "pr_auc": 0.9})
    result = _run_cli("--version", v2)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout
    # Production must be untouched: still v1.
    prod = get_production_version(client, MODEL_NAME)
    assert str(prod.version) == v1


def test_second_promotion_approved_when_it_clears_the_gate(mlflow_tracking_uri):
    client = MlflowClient()
    client.create_registered_model(MODEL_NAME)

    v1 = _make_candidate_version(client, {"f2_score": 0.3, "precision": 0.4, "pr_auc": 0.5})
    assert _run_cli("--version", v1, "--confirm-bootstrap").returncode == 0

    v2 = _make_candidate_version(client, {"f2_score": 0.35, "precision": 0.42, "pr_auc": 0.6})
    result = _run_cli("--version", v2)

    assert result.returncode == 0, result.stdout + result.stderr
    prod = get_production_version(client, MODEL_NAME)
    assert str(prod.version) == v2
    assert prod.tags["image_tag"] == f"sha-{v2}"
