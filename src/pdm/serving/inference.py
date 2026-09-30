"""Pure model-inference helpers, decoupled from FastAPI so they're unit-testable without
spinning up the app, MLflow, or a real registered model.
"""

from __future__ import annotations

import math

import pandas as pd


def validate_features(features: dict[str, float], required_columns: list[str]) -> list[str]:
    """Returns the required columns missing from `features` (empty list if all present)."""
    return [c for c in required_columns if c not in features]


def predict_details(model, features: dict[str, float], required_columns: list[str]) -> dict:
    """Builds the single-row model input in the required column order and returns
    {"rul", "rul_lower", "rul_upper", "maintenance_recommended"}. Models that return a
    plain number (no interval, no maintenance rule) get None for the extra fields.
    Assumes `validate_features` has already been checked.
    """
    row = pd.DataFrame([{c: features[c] for c in required_columns}])
    output = model.predict(row)
    if isinstance(output, pd.DataFrame):
        first = output.iloc[0]

        def optional(name: str) -> float | None:
            value = first.get(name)
            return None if value is None or math.isnan(float(value)) else float(value)

        maintain = optional("maintenance_recommended")
        return {
            "rul": float(first["rul"]),
            "rul_lower": optional("rul_lower"),
            "rul_upper": optional("rul_upper"),
            "maintenance_recommended": None if maintain is None else bool(maintain),
        }
    return {
        "rul": float(output[0]),
        "rul_lower": None,
        "rul_upper": None,
        "maintenance_recommended": None,
    }


def predict_row(model, features: dict[str, float], required_columns: list[str]) -> float:
    """Point RUL only, as a plain float."""
    return predict_details(model, features, required_columns)["rul"]
