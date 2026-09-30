"""Serving and drift configs must name the exact columns the C-MAPSS model is trained on.

A mismatch here is silent until runtime: MLflow's schema enforcement rejects every
/predict call, and the drift check finds no overlapping columns.
"""

import pytest

from pdm.common.config import load_yaml
from pdm.data.features import feature_columns

pytestmark = pytest.mark.unit


def _training_columns() -> list[str]:
    features = load_yaml("training.yaml")["features"]
    return feature_columns(features["sensor_columns"], max(features["rolling_windows"]))


def test_serving_required_columns_match_training_features():
    required = load_yaml("serving.yaml")["feature_schema"]["required_columns"]
    assert sorted(required) == sorted(_training_columns())


def test_drift_columns_are_training_features():
    drift_columns = load_yaml("drift.yaml")["drift"]["columns"]
    assert set(drift_columns) <= set(_training_columns())
