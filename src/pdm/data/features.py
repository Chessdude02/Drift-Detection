"""Feature engineering shared by training and serving: rolling sensor stats + RUL capping."""

from __future__ import annotations

import pandas as pd

from pdm.data.cmapss import OP_SETTING_COLUMNS


def cap_rul(rul: pd.Series, cap: int = 125) -> pd.Series:
    """Clip RUL at `cap` — standard C-MAPSS practice since RUL is flat/unknown far from failure."""
    return rul.clip(upper=cap)


def add_rolling_features(
    df: pd.DataFrame, sensor_columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add per-unit rolling mean features for each sensor column and window size.

    Rolling stats are grouped by unit_number and ordered by time_in_cycles so windows
    never leak across engines. Only rolling *mean* at the largest window is kept as the
    model feature set (matches config/serving.yaml's required_columns), computed here
    generically so callers can add more windows/statistics without touching serving code.
    """
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    grouped = df.groupby("unit_number", group_keys=False)
    for window in windows:
        for col in sensor_columns:
            out_col = f"{col}_roll_mean_{window}"
            df[out_col] = grouped[col].apply(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
    return df


def build_feature_matrix(
    df: pd.DataFrame, sensor_columns: list[str], windows: list[int], primary_window: int
) -> pd.DataFrame:
    """Produce the exact feature matrix the model trains/predicts on.

    `primary_window` selects which rolling window's columns become the canonical
    `sensor_X_roll_mean_<primary_window>` features (must match config/serving.yaml).
    """
    featured = add_rolling_features(df, sensor_columns, windows)
    feature_cols = OP_SETTING_COLUMNS + [
        f"{col}_roll_mean_{primary_window}" for col in sensor_columns
    ]
    return featured[["unit_number", "time_in_cycles"] + feature_cols].copy()


def feature_columns(sensor_columns: list[str], primary_window: int) -> list[str]:
    return OP_SETTING_COLUMNS + [f"{col}_roll_mean_{primary_window}" for col in sensor_columns]
