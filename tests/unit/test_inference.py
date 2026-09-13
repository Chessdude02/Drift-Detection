from __future__ import annotations

import pandas as pd
import pytest

from pdm.serving.inference import predict_row, validate_features

pytestmark = pytest.mark.unit

REQUIRED = ["a", "b", "c"]


class RecordingModel:
    """Records the exact DataFrame it was called with, so tests can assert column order
    and that no extra/missing columns leak through."""

    def __init__(self, return_value: float = 1.0):
        self.return_value = return_value
        self.last_input: pd.DataFrame | None = None

    def predict(self, df: pd.DataFrame):
        self.last_input = df
        return [self.return_value] * len(df)


def test_validate_features_all_present_returns_empty():
    assert validate_features({"a": 1, "b": 2, "c": 3}, REQUIRED) == []


def test_validate_features_reports_missing():
    missing = validate_features({"a": 1}, REQUIRED)
    assert missing == ["b", "c"]


def test_validate_features_ignores_extra_columns():
    assert validate_features({"a": 1, "b": 2, "c": 3, "extra": 99}, REQUIRED) == []


def test_predict_row_returns_plain_float():
    model = RecordingModel(return_value=42.5)
    result = predict_row(model, {"a": 1, "b": 2, "c": 3}, REQUIRED)
    assert result == 42.5
    assert isinstance(result, float)


def test_predict_row_builds_single_row_in_required_order():
    model = RecordingModel()
    predict_row(model, {"c": 3, "a": 1, "b": 2, "unused": 0}, REQUIRED)
    assert model.last_input is not None
    assert list(model.last_input.columns) == REQUIRED
    assert model.last_input.iloc[0].tolist() == [1, 2, 3]
    assert len(model.last_input) == 1
