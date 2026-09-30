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
    resp = client.post(
        "/predict", json={"asset_id": "engine-1", "features": _sample_features(client)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_rul"] == 42.0
    assert body["model_name"] == "test_model"


def test_predict_missing_features_returns_422(client):
    resp = client.post("/predict", json={"asset_id": "engine-1", "features": {}})
    assert resp.status_code == 422


def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "inference_log.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    from pdm.serving.app import app

    with TestClient(app) as c:
        required = c.app.state.required_columns
        resp = c.post(
            "/predict", json={"asset_id": "engine-1", "features": {col: 1.0 for col in required}}
        )
        assert resp.status_code == 503


def test_predict_shadow_header_marks_response(client):
    resp = client.post(
        "/predict",
        json={"asset_id": "engine-1", "features": _sample_features(client)},
        headers={"X-Shadow": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-shadow") == "true"


def test_predict_without_asset_id_is_rejected(client):
    resp = client.post("/predict", json={"features": _sample_features(client)})
    assert resp.status_code == 422
    assert "asset_id" in resp.text


def test_predict_non_finite_feature_is_rejected(client):
    features = _sample_features(client)
    first = next(iter(features))
    body = '{"asset_id": "engine-1", "features": {%s}}' % ", ".join(
        f'"{k}": {"NaN" if k == first else v}' for k, v in features.items()
    )
    resp = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert first in resp.text


def test_out_of_range_feature_is_flagged_not_rejected(client):
    features = _sample_features(client)
    first = next(iter(features))
    loaded = client.app.state.loader._loaded
    loaded.feature_ranges = {first: (0.0, 0.5)}
    resp = client.post("/predict", json={"asset_id": "engine-1", "features": features})
    assert resp.status_code == 200
    assert resp.json()["input_warnings"] == [f"{first} outside training range"]


def test_asset_cycle_and_warnings_reach_the_inference_log(client):
    features = _sample_features(client)
    client.post(
        "/predict",
        json={
            "asset_id": "engine-7",
            "cycle": 42,
            "observed_at": "2026-01-01T00:00:00Z",
            "features": features,
        },
    )
    row = client.app.state.inference_log.read_recent(1)[0]
    assert row["asset_id"] == "engine-7" and row["cycle"] == 42
    assert row["observed_at"] == 1767225600.0
    assert row["model_version"] == "1"
