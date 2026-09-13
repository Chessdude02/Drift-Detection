"""Evaluation metrics and the CI/CD model-validation gate."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def passes_validation_gate(rmse_value: float, max_rmse: float) -> bool:
    """The gate CI (and train.py itself) uses to decide whether a model may be promoted."""
    return rmse_value <= max_rmse
