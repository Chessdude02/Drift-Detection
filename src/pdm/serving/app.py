"""FastAPI serving app: /predict, /healthz, /readyz, /metrics.

Model loading is startup-best-effort: if the MLflow registry has no Production model yet
(fresh environment, registry unreachable), the app still starts so /healthz succeeds and
Kubernetes doesn't crash-loop it — but /readyz and /predict report not-ready until a model
loads, which also naturally gates Argo Rollouts' readiness/analysis checks.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pdm.common.config import configure_mlflow_env, get_settings, load_yaml
from pdm.common.logging import setup_logging
from pdm.serving.inference import predict_row, validate_features
from pdm.serving.inference_log import InferenceLog
from pdm.serving.metrics import (
    MODEL_INFO,
    PREDICTION_VALUE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    feature_stats,
)
from pdm.serving.model_loader import ModelLoader
from pdm.serving.schemas import HealthResponse, PredictRequest, PredictResponse, ReadyResponse

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow_env()
    settings = get_settings()
    serving_cfg = load_yaml("serving.yaml")

    app.state.required_columns = serving_cfg["feature_schema"]["required_columns"]
    app.state.inference_log = InferenceLog(settings.inference_log_db)
    app.state.loader = ModelLoader(
        model_name=serving_cfg["model"]["name"],
        stage=serving_cfg["model"]["stage"],
        refresh_seconds=serving_cfg["model"]["refresh_seconds"],
    )
    app.state.loader.start()
    yield
    app.state.loader.stop()


app = FastAPI(title="Predictive Maintenance Serving API", lifespan=lifespan)


def _get_loaded_model(request: Request):
    loader: ModelLoader = request.app.state.loader
    return loader.loaded


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness: process is up. Does not require a model to be loaded."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=ReadyResponse)
def readyz(request: Request) -> ReadyResponse:
    """Readiness: only true once a Production model is loaded. Gates k8s/Argo Rollouts traffic."""
    loaded = _get_loaded_model(request)
    if loaded is None:
        return ReadyResponse(ready=False, detail="no model loaded yet")
    return ReadyResponse(ready=True, model_name=loaded.name, model_version=loaded.version)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, response: Response) -> PredictResponse:
    loaded = _get_loaded_model(request)
    shadow = request.headers.get("x-shadow", "").lower() == "true"
    shadow_label = "true" if shadow else "false"

    if loaded is None:
        REQUEST_COUNT.labels(status="error", shadow=shadow_label).inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    required = request.app.state.required_columns
    missing = validate_features(req.features, required)
    if missing:
        REQUEST_COUNT.labels(status="error", shadow=shadow_label).inc()
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    start = time.perf_counter()
    try:
        prediction = predict_row(loaded.model, req.features, required)
    except Exception:
        REQUEST_COUNT.labels(status="error", shadow=shadow_label).inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        REQUEST_LATENCY.labels(shadow=shadow_label).observe(time.perf_counter() - start)

    REQUEST_COUNT.labels(status="ok", shadow=shadow_label).inc()
    PREDICTION_VALUE.observe(prediction)
    feature_stats.update({c: req.features[c] for c in required})
    request.app.state.inference_log.record(req.features, prediction, shadow=shadow)
    MODEL_INFO.labels(model_name=loaded.name, version=loaded.version).set(1)

    if shadow:
        # Shadow/mirrored traffic: log for comparison but the caller (the mirror mechanism)
        # never surfaces this response to a real user, so we still return it honestly to
        # whatever called us directly (e.g. this endpoint under test) rather than faking 202 —
        # the "don't affect real users" guarantee lives in the Ingress mirror config, not here.
        response.headers["X-Shadow"] = "true"

    return PredictResponse(
        predicted_rul=prediction, model_name=loaded.name, model_version=loaded.version
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
