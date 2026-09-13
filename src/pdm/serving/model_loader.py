"""Loads and periodically refreshes the current Production model from the MLflow registry."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    name: str
    version: str


class ModelLoader:
    """Thread-safe holder for the currently-loaded model, with a background refresh loop.

    Kept deliberately simple: refresh_seconds polling against the MLflow registry rather than
    a push-based mechanism, since a demo/reference serving app doesn't need sub-second promotion
    latency and this avoids needing a message bus.
    """

    def __init__(self, model_name: str, stage: str, refresh_seconds: int = 300):
        self.model_name = model_name
        self.stage = stage
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        with self._lock:
            return self._loaded

    def load_once(self) -> LoadedModel:
        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        versions = client.get_latest_versions(self.model_name, stages=[self.stage])
        if not versions:
            raise RuntimeError(
                f"No model version found for '{self.model_name}' in stage '{self.stage}'"
            )
        latest = versions[0]
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.stage}")
        loaded = LoadedModel(model=model, name=self.model_name, version=latest.version)
        with self._lock:
            self._loaded = loaded
        logger.info(
            "Loaded model %s version %s (stage=%s)", self.model_name, latest.version, self.stage
        )
        return loaded

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            try:
                self.load_once()
            except Exception:
                logger.exception("Model refresh failed; keeping previously loaded model")

    def start(self) -> None:
        try:
            self.load_once()
        except Exception:
            logger.exception(
                "Initial model load failed; serving will report not-ready until a model loads"
            )
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
