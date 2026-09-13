"""Shared config loading: YAML files + environment variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    mlflow_default_artifact_root: str = "./mlruns"
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
    """Ensure MLflow env vars are set from Settings before importing mlflow elsewhere."""
    settings = get_settings()
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
