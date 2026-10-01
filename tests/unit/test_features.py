from __future__ import annotations

import numpy as np
import pytest

from pdm.data.cmapss import SENSOR_COLUMNS, load_test, load_train
from pdm.data.features import add_rolling_features, build_feature_matrix, cap_rul, feature_columns

pytestmark = pytest.mark.unit


def test_load_train_computes_rul(raw_data_dir):
    df = load_train(raw_data_dir, subset="FD001")
    assert "rul" in df.columns
    # RUL at each unit's last cycle must be 0.
    last_rows = df.sort_values("time_in_cycles").groupby("unit_number").tail(1)
    assert (last_rows["rul"] == 0).all()
    # RUL decreases monotonically within a unit as cycles increase.
    for _, grp in df.groupby("unit_number"):
        grp = grp.sort_values("time_in_cycles")
        assert (grp["rul"].diff().dropna() <= 0).all()


def test_load_test_and_rul(raw_data_dir):
    test_df, true_rul = load_test(raw_data_dir, subset="FD001")
    assert set(test_df["unit_number"].unique()) == set(true_rul.index)
    assert (true_rul >= 0).all()


def test_cap_rul():
    import pandas as pd

    rul = pd.Series([0, 50, 130, 200])
    capped = cap_rul(rul, cap=125)
    assert capped.tolist() == [0, 50, 125, 125]


def test_add_rolling_features_no_cross_unit_leakage(raw_data_dir):
    df = load_train(raw_data_dir, subset="FD001")
    windows = [5, 10]
    featured = add_rolling_features(df, SENSOR_COLUMNS[:2], windows)
    for col in SENSOR_COLUMNS[:2]:
        assert f"{col}_roll_mean_10" in featured.columns
    # First row of each unit's rolling mean should equal its own raw value (min_periods=1).
    for unit, grp in featured.groupby("unit_number"):
        grp = grp.sort_values("time_in_cycles")
        first = grp.iloc[0]
        assert abs(first[f"{SENSOR_COLUMNS[0]}_roll_mean_5"] - first[SENSOR_COLUMNS[0]]) < 1e-6


def test_build_feature_matrix_matches_feature_columns(raw_data_dir):
    df = load_train(raw_data_dir, subset="FD001")
    sensors = SENSOR_COLUMNS[:3]
    windows = [10]
    matrix = build_feature_matrix(df, sensors, windows, primary_window=10)
    expected = feature_columns(sensors, primary_window=10)
    assert set(expected).issubset(matrix.columns)
    assert matrix.shape[0] == df.shape[0]


def test_trend_features_are_short_minus_long_mean(raw_data_dir):
    from pdm.data.features import columns_from_config, features_from_config

    df = load_train(raw_data_dir, subset="FD001")
    cfg = {
        "sensor_columns": SENSOR_COLUMNS[:2],
        "rolling_windows": [5, 10],
        "trend": {"short_window": 2, "long_window": 5},
    }
    out = features_from_config(df, cfg)
    cols = columns_from_config(cfg)
    assert list(out.columns) == ["unit_number", "time_in_cycles", *cols]
    trend_col = f"{SENSOR_COLUMNS[0]}_trend_2_5"
    assert trend_col in cols
    rolled = add_rolling_features(df, SENSOR_COLUMNS[:1], [2, 5])
    expected = (
        rolled[f"{SENSOR_COLUMNS[0]}_roll_mean_2"] - rolled[f"{SENSOR_COLUMNS[0]}_roll_mean_5"]
    )
    np.testing.assert_allclose(out[trend_col].to_numpy(), expected.to_numpy())

    without = features_from_config(df, {k: v for k, v in cfg.items() if k != "trend"})
    assert not any("trend" in c for c in without.columns)
