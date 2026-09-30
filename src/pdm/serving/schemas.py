from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One row of engineered features, keyed by column name, for one engine.

    `features` must contain every column in config/serving.yaml's
    feature_schema.required_columns; extra columns are ignored.

    `asset_id` identifies the engine. It is required by default
    (serving.yaml input_validation.require_asset_id): without it the drift check cannot
    count engines per window and outcomes (failures, maintenance) cannot be matched back
    to predictions to create new training labels.
    """

    features: dict[str, float] = Field(..., description="feature_name -> value")
    asset_id: str | None = Field(None, min_length=1, max_length=128, description="engine id")
    cycle: int | None = Field(
        None, ge=0, description="engine operating cycle of this reading (for labelling)"
    )
    observed_at: datetime | None = Field(
        None, description="when the reading was taken; defaults to when it was received"
    )


class PredictResponse(BaseModel):
    predicted_rul: float
    model_name: str
    model_version: str
    # Calibrated interval (about `coverage` of true RULs fall inside; see
    # pdm.training.rul_model) and the model's maintenance rule applied to its lower
    # bound. None when the served model has no interval / rule.
    rul_lower: float | None = None
    rul_upper: float | None = None
    maintenance_recommended: bool | None = None
    # Features outside the range seen in training. The prediction is still returned
    # (the model holds its edge value flat), but treat it with suspicion.
    input_warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None
