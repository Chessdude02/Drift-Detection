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
    df: pd.DataFrame,
    sensor_columns: list[str],
    windows: list[int],
    primary_window: int,
    trend: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Produce the exact feature matrix the model trains/predicts on.

    `primary_window` selects which rolling window's columns become the canonical
    `sensor_X_roll_mean_<primary_window>` features (must match config/serving.yaml).
    `trend=(short, long)` adds `sensor_X_trend_<short>_<long>` = short-window mean minus
    long-window mean per sensor: how fast the sensor is moving (docs/decisions.md D-27,
    D-30). Off by default; only the `cmapss_rul_trend` model uses it.
    """
    needed = sorted(set(windows) | set(trend or ()))
    featured = add_rolling_features(df, sensor_columns, needed)
    for col in sensor_columns:
        if trend:
            short, long = trend
            featured[_trend_name(col, trend)] = (
                featured[f"{col}_roll_mean_{short}"] - featured[f"{col}_roll_mean_{long}"]
            )
    cols = feature_columns(sensor_columns, primary_window, trend)
    return featured[["unit_number", "time_in_cycles"] + cols].copy()


def _trend_name(col: str, trend: tuple[int, int]) -> str:
    return f"{col}_trend_{trend[0]}_{trend[1]}"


def feature_columns(
    sensor_columns: list[str], primary_window: int, trend: tuple[int, int] | None = None
) -> list[str]:
    cols = OP_SETTING_COLUMNS + [f"{col}_roll_mean_{primary_window}" for col in sensor_columns]
    if trend:
        cols += [_trend_name(col, trend) for col in sensor_columns]
    return cols


def trend_from_config(features_cfg: dict) -> tuple[int, int] | None:
    trend = features_cfg.get("trend")
    return (int(trend["short_window"]), int(trend["long_window"])) if trend else None


def features_from_config(df: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    """build_feature_matrix with everything taken from a training config's `features`."""
    windows = features_cfg["rolling_windows"]
    return build_feature_matrix(
        df, features_cfg["sensor_columns"], windows, max(windows), trend_from_config(features_cfg)
    )


def columns_from_config(features_cfg: dict) -> list[str]:
    """feature_columns for a training config's `features`."""
    return feature_columns(
        features_cfg["sensor_columns"],
        max(features_cfg["rolling_windows"]),
        trend_from_config(features_cfg),
    )
