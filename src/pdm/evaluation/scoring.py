"""Scores a trained RUL regressor on the fixed holdout set as a binary failure
classifier: F2-score (beta=2, recall-weighted) and precision on the failure class, plus
PR-AUC as a threshold-independent ranking metric.

The underlying model is still a RUL regressor (unchanged) — a snapshot is labeled/
predicted "failure" when its true/predicted RUL falls at or under `failure_horizon`
(config/evaluation.yaml), and `-predicted_rul` is used as the continuous risk score for
PR-AUC (lower predicted RUL = higher predicted failure risk = higher score).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score

logger = logging.getLogger(__name__)


def compute_holdout_scores(
    model, holdout_df: pd.DataFrame, feature_columns: list[str], failure_horizon: int
) -> dict[str, float]:
    """Returns {"f2_score", "precision", "pr_auc", "n_holdout", "n_positive"}.

    `model` is anything exposing `.predict(X) -> array-like` (an MLflow pyfunc model or
    a raw sklearn/lightgbm estimator both satisfy this).
    """
    y_true = (holdout_df["rul"] <= failure_horizon).astype(int).to_numpy()
    predicted_rul = np.asarray(model.predict(holdout_df[feature_columns]))
    y_pred = (predicted_rul <= failure_horizon).astype(int)
    risk_score = -predicted_rul

    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        logger.warning(
            "Holdout set has %d/%d positive (failure) rows; PR-AUC is undefined for a "
            "single-class holdout and is reported as NaN.",
            n_positive,
            len(y_true),
        )
        pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, risk_score))

    return {
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
        "n_holdout": float(len(y_true)),
        "n_positive": float(n_positive),
    }
