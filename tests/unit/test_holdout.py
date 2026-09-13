from __future__ import annotations

import pandas.testing as pdt
import pytest

from pdm.data.bearing_features import feature_columns
from pdm.evaluation.holdout import build_holdout_dataframe

pytestmark = pytest.mark.unit

CHANNELS = ["bearing1", "bearing2", "bearing3", "bearing4"]


def test_build_holdout_dataframe_has_features_and_rul_but_no_label(bearing_raw_dir):
    df = build_holdout_dataframe(bearing_raw_dir, channels=CHANNELS, failure_channel="bearing4")

    assert set(feature_columns(CHANNELS)).issubset(df.columns)
    assert "rul" in df.columns
    # Deliberately no baked-in binary label: the failure horizon is a scoring-time
    # config value, not frozen into the holdout file.
    assert "failure_label" not in df.columns
    assert "snapshot_index" in df.columns and "timestamp" in df.columns


def test_build_holdout_dataframe_is_deterministic(bearing_raw_dir):
    df1 = build_holdout_dataframe(bearing_raw_dir, channels=CHANNELS, failure_channel="bearing4")
    df2 = build_holdout_dataframe(bearing_raw_dir, channels=CHANNELS, failure_channel="bearing4")
    pdt.assert_frame_equal(df1, df2)
