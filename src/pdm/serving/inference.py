"""Pure model-inference helpers, decoupled from FastAPI so they're unit-testable without
spinning up the app, MLflow, or a real registered model.
"""

from __future__ import annotations

import pandas as pd


def validate_features(features: dict[str, float], required_columns: list[str]) -> list[str]:
    """Returns the required columns missing from `features` (empty list if all present)."""
    return [c for c in required_columns if c not in features]


def predict_row(model, features: dict[str, float], required_columns: list[str]) -> float:
    """Builds the single-row model input in the required column order and returns a
    plain float prediction. Assumes `validate_features` has already been checked.
    """
    row = pd.DataFrame([{c: features[c] for c in required_columns}])
    return float(model.predict(row)[0])
