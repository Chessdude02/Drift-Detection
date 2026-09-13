"""Shared config loading: YAML files + environment variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from the config/ directory, e.g. load_yaml('training.yaml')."""
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    # None (default) = let MLflow pick its own local default for new experiments.
    # Set to a gs:// (or any MLflow-supported) URI to give every newly-created
    # experiment that artifact root explicitly - see set_experiment_with_artifact_root:
    # MLflow's server-only MLFLOW_DEFAULT_ARTIFACT_ROOT env var has NO effect on a bare
    # client-side tracking store (verified empirically), so this has to be threaded
    # through explicitly at experiment-creation time instead of just set as an env var.
    mlflow_default_artifact_root: str | None = None
    model_name: str = "cmapss_rul"
    model_stage: str = "Production"

    data_dir: str = "./data"

    serving_port: int = 8000
    model_refresh_seconds: int = 300
    inference_log_db: str = "./inference_log.db"

    drift_threshold: float = 0.5
    drift_reference_path: str = "./data/processed/reference.parquet"
    prometheus_pushgateway_url: str = "http://localhost:9091"

    kube_namespace: str = "pdm"
    retrain_cronjob_name: str = "pdm-retrain"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_mlflow_env() -> None:
    """Ensure MLFLOW_TRACKING_URI is set to a real value before importing mlflow elsewhere.

    Guards against the specific failure mode that once broke CI silently: a workflow
    step referencing an unset/misnamed secret via `${{ secrets.X }}` resolves to an
    EMPTY STRING, not "unset". `os.environ.setdefault` never overrides an already-present
    (even empty) key, and MLflow treats an empty-string tracking URI as "use the local
    ./mlruns default" - so nothing would ever raise, it would just silently write every
    CI run's data to a directory that's destroyed when the ephemeral runner exits. If the
    env var is present but empty, that's essentially always a misconfigured CI secret,
    never intentional local-dev usage (leaving it fully unset is how you opt into the
    local default on purpose) - fail loudly instead of guessing.
    """
    existing = os.environ.get("MLFLOW_TRACKING_URI")
    if existing == "":
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is set but empty. This almost always means it was "
            "sourced from a CI secret/variable that doesn't exist or was never set "
            "(e.g. ${{ secrets.SOME_NAME }} where SOME_NAME isn't configured on this "
            "repo) - refusing to silently fall back to a local ./mlruns store that "
            "disappears with the runner. Set MLFLOW_TRACKING_URI to a real value, or "
            "unset the env var entirely to use the local default intentionally."
        )
    settings = get_settings()
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)


def _normalize_artifact_root(root: str) -> str:
    """A bare local filesystem path (no `scheme://`) has to become a proper `file://`
    URI before MLflow will accept it as an artifact_location - passing e.g. the raw
    Windows path `C:/Users/.../mlruns` fails, because `C:` parses as a URI scheme
    ("C"), not a drive letter, and MLflow's artifact-repository registry doesn't
    recognize that scheme. A real remote URI like `gs://...`/`s3://...` already has a
    scheme and is returned unchanged.
    """
    if "://" in root:
        return root.rstrip("/")
    return Path(root).resolve().as_uri()


def set_experiment_with_artifact_root(name: str) -> None:
    """Like `mlflow.set_experiment(name)`, but if the experiment doesn't exist yet and
    `Settings.mlflow_default_artifact_root` is set, creates it with that artifact root
    explicitly. Existing experiments are left exactly as they were first created - only
    matters the very first time a given experiment name is used against a given
    tracking store (e.g. right after a fresh mlflow.db is pulled from GCS with no
    matching experiment in it yet).
    """
    settings = get_settings()
    client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        mlflow.set_experiment(experiment_id=existing.experiment_id)
        return

    if settings.mlflow_default_artifact_root:
        root = _normalize_artifact_root(settings.mlflow_default_artifact_root)
        exp_id = client.create_experiment(name, artifact_location=f"{root}/{name}")
    else:
        exp_id = client.create_experiment(name)
    mlflow.set_experiment(experiment_id=exp_id)
