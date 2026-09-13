from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One row of engineered features, keyed by column name.

    Must contain every column in config/serving.yaml's feature_schema.required_columns.
    Extra columns are ignored; missing ones are rejected in app.py before scoring.
    """

    features: dict[str, float] = Field(..., description="feature_name -> value")


class PredictResponse(BaseModel):
    predicted_rul: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None
