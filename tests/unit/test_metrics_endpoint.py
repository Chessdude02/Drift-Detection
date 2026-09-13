from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pdm.serving.model_loader import LoadedModel

pytestmark = pytest.mark.unit


class DummyModel:
    def predict(self, df):
        return [10.0] * len(df)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "inference_log.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    from pdm.serving.app import app

    with TestClient(app) as c:
        c.app.state.loader._loaded = LoadedModel(model=DummyModel(), name="test_model", version="1")
        yield c


def test_metrics_endpoint_exposes_expected_names(client):
    required = client.app.state.required_columns
    client.post("/predict", json={"features": {c: 1.0 for c in required}})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    for expected in [
        "pdm_predictions_total",
        "pdm_request_latency_seconds",
        "pdm_prediction_value",
        "pdm_feature_mean",
        "pdm_feature_stddev",
        "pdm_model_version_info",
    ]:
        assert expected in body, f"missing metric {expected}"
