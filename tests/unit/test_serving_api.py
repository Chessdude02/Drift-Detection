from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pdm.serving.model_loader import LoadedModel

pytestmark = pytest.mark.unit


class DummyModel:
    def predict(self, df):
        return [42.0] * len(df)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "inference_log.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    from pdm.serving.app import app

    with TestClient(app) as c:
        # Startup couldn't find a real registered model (none exists in this tmp tracking
        # store) — inject a dummy one directly, the same seam a real promotion would use.
        c.app.state.loader._loaded = LoadedModel(model=DummyModel(), name="test_model", version="1")
        yield c


def _sample_features(client) -> dict:
    required = client.app.state.required_columns
    return {c: 1.0 for c in required}


def test_healthz_always_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_true_once_model_loaded(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["model_name"] == "test_model"


def test_readyz_false_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "inference_log.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    from pdm.serving.app import app

    with TestClient(app) as c:
        resp = c.get("/readyz")
        assert resp.json()["ready"] is False


def test_predict_happy_path(client):
    resp = client.post("/predict", json={"features": _sample_features(client)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_rul"] == 42.0
    assert body["model_name"] == "test_model"


def test_predict_missing_features_returns_422(client):
    resp = client.post("/predict", json={"features": {}})
    assert resp.status_code == 422


def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "inference_log.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    from pdm.serving.app import app

    with TestClient(app) as c:
        required = c.app.state.required_columns
        resp = c.post("/predict", json={"features": {col: 1.0 for col in required}})
        assert resp.status_code == 503


def test_predict_shadow_header_marks_response(client):
    resp = client.post(
        "/predict", json={"features": _sample_features(client)}, headers={"X-Shadow": "true"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-shadow") == "true"
