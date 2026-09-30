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

Telling a broken sensor from a fleet-wide shift (diagnose): a broken sensor also shifts
the residuals of every sensor that uses it as a predictor, so a large offset can flag
most of the 14 sensors - and so does a real fleet-wide shift (a new fault mode flags
13-14). Counting flagged sensors cannot separate them. Removing the top-scoring sensor
and re-scoring the rest can: on FD001, after removing it, 0 sensors stay flagged for
every single-sensor offset (0.5-3 std) and stuck sensor tested, and 12-13 stay flagged
for the FD003/FD002 fleet shifts. So a fault is reported only when removing at most
`max_culprits` sensors explains every flag; otherwise it is a system-wide shift, which is
the retrain check's job, not a sensor alert.
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


def diagnose(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    threshold: float,
    max_culprits: int = 2,
) -> dict:
    """Returns {"verdict", "culprits", "scores"}. verdict is:
    - "ok": no sensor over threshold;
    - "sensor_fault": removing the `culprits` (at most max_culprits, worst first)
      leaves no other sensor flagged;
    - "system_wide_shift": it does not - the whole relationship between sensors changed.
    `scores` are the first-pass per-sensor scores (all columns)."""
    scores = sensor_fault_scores(reference, current, columns)
    if not faulty_sensors(scores, threshold):
        return {"verdict": "ok", "culprits": [], "scores": scores}
    culprits: list[str] = []
    remaining, current_scores = list(columns), scores
    for _ in range(max_culprits):
        top = max(current_scores, key=current_scores.get)
        culprits.append(top)
        remaining = [c for c in remaining if c != top]
        if len(remaining) < 2:
            break
        current_scores = sensor_fault_scores(reference, current, remaining)
        if not faulty_sensors(current_scores, threshold):
            return {"verdict": "sensor_fault", "culprits": culprits, "scores": scores}
    return {"verdict": "system_wide_shift", "culprits": [], "scores": scores}
