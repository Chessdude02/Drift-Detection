"""Per-sensor fault check, separate from the retrain trigger.

The retrain trigger needs more than half the columns to drift, so one broken sensor is
invisible to it. Checking each sensor's own distribution does not work well either:
engine-to-engine variation and wear move every sensor, so the threshold has to be loose.

Instead, each sensor is predicted from the other sensors with a linear model fitted on
the reference. Wear and engine-to-engine differences move sensors *together*, so the
prediction error (residual) stays stable; a biased or stuck sensor breaks that
relationship and its residual distribution shifts. Measured on real FD001 (see
scripts/calibrate_drift.py): a single-sensor offset of 0.5 std is caught ~100% of the
time with ~0-2% false alarms on normal and ageing fleets.

A fired sensor alert means "inspect this sensor", NOT "retrain": retraining on data from
a broken sensor would bake the fault into the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression


def _normed_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Same formula as Evidently's 'wasserstein' stattest."""
    return float(wasserstein_distance(reference, current) / max(np.std(reference), 0.001))


def fit_residual_models(reference: pd.DataFrame, columns: list[str]) -> dict:
    return {
        col: LinearRegression().fit(reference[[c for c in columns if c != col]], reference[col])
        for col in columns
    }


def residuals(models: dict, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: df[col].to_numpy() - model.predict(df[[c for c in columns if c != col]])
            for col, model in models.items()
        }
    )


def sensor_fault_scores(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    """Normed Wasserstein distance between reference and current residuals, per column."""
    models = fit_residual_models(reference, columns)
    ref_res = residuals(models, reference, columns)
    cur_res = residuals(models, current, columns)
    return {
        col: _normed_wasserstein(ref_res[col].to_numpy(), cur_res[col].to_numpy())
        for col in columns
    }


def faulty_sensors(scores: dict[str, float], threshold: float) -> list[str]:
    """Columns above threshold, worst first. An offset on one sensor also nudges the
    residuals of sensors that use it as a predictor, so the first entry is the most
    likely culprit (see scripts/calibrate_drift.py for how often it is right).
    """
    return sorted((c for c, s in scores.items() if s > threshold), key=lambda c: -scores[c])
