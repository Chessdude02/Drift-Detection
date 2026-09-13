from __future__ import annotations

import math

import pandas as pd
import pytest

from pdm.evaluation.scoring import compute_holdout_scores

pytestmark = pytest.mark.unit

FEATURE_COLUMNS = ["x"]


class ThresholdModel:
    """Predicts a fixed RUL per row, given directly as the 'x' feature column value."""

    def predict(self, df: pd.DataFrame):
        return df["x"].to_numpy()


def _holdout(rul_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x": rul_values, "rul": rul_values})


def test_perfect_predictions_score_maximally():
    # true rul == predicted rul (model just echoes 'x'), horizon=5 -> rows 0..2 are
    # failures (rul<=5), rows 10,20 are healthy. Perfect echo -> perfect classification.
    holdout = _holdout([0, 3, 5, 10, 20])
    scores = compute_holdout_scores(ThresholdModel(), holdout, FEATURE_COLUMNS, failure_horizon=5)
    assert scores["f2_score"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["pr_auc"] == 1.0
    assert scores["n_holdout"] == 5
    assert scores["n_positive"] == 3


def test_worst_case_predictions_score_at_floor():
    # Model predicts the true rul plus a huge offset -> never predicts "failure" even
    # though 2 rows truly are failures. Zero recall/precision.
    class OffsetModel:
        def predict(self, df):
            return df["x"].to_numpy() + 1000

    holdout = _holdout([0, 3, 10, 20])
    scores = compute_holdout_scores(OffsetModel(), holdout, FEATURE_COLUMNS, failure_horizon=5)
    assert scores["f2_score"] == 0.0
    assert scores["precision"] == 0.0


def test_single_class_holdout_reports_nan_pr_auc(caplog):
    # All rows healthy (rul > horizon) -> no positives -> PR-AUC undefined.
    holdout = _holdout([10, 20, 30])
    scores = compute_holdout_scores(ThresholdModel(), holdout, FEATURE_COLUMNS, failure_horizon=5)
    assert math.isnan(scores["pr_auc"])
    assert scores["n_positive"] == 0


def test_scores_are_plain_finite_floats_for_mixed_predictions():
    holdout = _holdout([0, 2, 4, 6, 8, 10, 15, 20])
    scores = compute_holdout_scores(ThresholdModel(), holdout, FEATURE_COLUMNS, failure_horizon=5)
    for key in ("f2_score", "precision", "pr_auc"):
        assert isinstance(scores[key], float)
        assert math.isfinite(scores[key])
        assert 0.0 <= scores[key] <= 1.0
