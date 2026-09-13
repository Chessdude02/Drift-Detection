from __future__ import annotations

import pandas as pd
import pytest

from pdm.data.cmapss import load_train
from pdm.data.features import build_feature_matrix, feature_columns
from pdm.drift.run_drift_check import compute_drift_score, trigger_retrain_if_needed

pytestmark = pytest.mark.unit

SENSORS = ["sensor_2", "sensor_3", "sensor_7"]
WINDOW = 5


@pytest.fixture
def feature_frames(raw_data_dir):
    df = load_train(raw_data_dir, subset="FD001")
    features = build_feature_matrix(df, SENSORS, [WINDOW], primary_window=WINDOW)
    return features, feature_columns(SENSORS, primary_window=WINDOW)


def test_compute_drift_score_identical_data_is_low(feature_frames):
    features, columns = feature_frames
    result = compute_drift_score(features, features, columns)
    assert result["share_of_drifted_columns"] == 0.0
    assert result["dataset_drift"] is False


def test_compute_drift_score_shifted_data_detects_drift(feature_frames):
    features, columns = feature_frames
    shifted = features.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(shifted[col]):
            shifted[col] = shifted[col] * 5 + 1000  # large synthetic shift

    result = compute_drift_score(features, shifted, columns)
    assert result["share_of_drifted_columns"] > 0.0


def test_trigger_retrain_below_threshold_is_noop():
    triggered = trigger_retrain_if_needed(score=0.1, threshold=0.5, config={}, dry_run=True)
    assert triggered is False


def test_trigger_retrain_above_threshold_dry_run_does_not_call_k8s():
    triggered = trigger_retrain_if_needed(
        score=0.9,
        threshold=0.5,
        config={
            "retrain_trigger": {"namespace": "pdm", "source_cronjob": "x", "job_name_prefix": "y"}
        },
        dry_run=True,
    )
    assert triggered is True
