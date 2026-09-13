from __future__ import annotations

import numpy as np
import pytest

from pdm.data.bearing_features import (
    CHANNEL_STATS,
    build_bearing_feature_matrix,
    compute_channel_features,
    feature_columns,
)

pytestmark = pytest.mark.unit

CHANNELS = ["bearing1", "bearing2", "bearing3", "bearing4"]


def test_compute_channel_features_keys_and_types():
    signal = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    features = compute_channel_features(signal)
    assert set(features.keys()) == set(CHANNEL_STATS)
    assert all(isinstance(v, float) for v in features.values())


def test_compute_channel_features_constant_signal_is_degenerate_but_finite():
    # A constant signal has zero variance: kurtosis/skew are degenerate (scipy returns
    # 0.0 for a zero-variance input rather than NaN/inf), and crest factor is exactly 1.
    signal = np.full(50, 0.25)
    features = compute_channel_features(signal)
    assert features["rms"] == 0.25
    assert features["std"] == 0.0
    assert features["crest_factor"] == 1.0
    assert all(np.isfinite(v) for v in features.values())


def test_compute_channel_features_higher_amplitude_increases_rms():
    quiet = compute_channel_features(np.random.default_rng(0).normal(0, 0.05, 1000))
    loud = compute_channel_features(np.random.default_rng(0).normal(0, 0.5, 1000))
    assert loud["rms"] > quiet["rms"]


def test_feature_columns_shape():
    cols = feature_columns(CHANNELS)
    assert len(cols) == len(CHANNELS) * len(CHANNEL_STATS)
    assert "bearing1_rms" in cols
    assert "bearing4_kurtosis" in cols


def test_build_bearing_feature_matrix_shape_and_rul(bearing_raw_dir):
    df = build_bearing_feature_matrix(
        bearing_raw_dir, channels=CHANNELS, failure_channel="bearing4"
    )

    n_files = len(list(bearing_raw_dir.iterdir()))
    assert len(df) == n_files
    assert set(feature_columns(CHANNELS)).issubset(df.columns)

    # RUL counts down to exactly 0 at the last (failure) snapshot and is monotonically
    # decreasing since a run-to-failure series has one snapshot per timestep.
    assert df["rul"].iloc[-1] == 0
    assert (df["rul"].diff().dropna() == -1).all()


def test_build_bearing_feature_matrix_detects_degradation_trend(bearing_raw_dir):
    """The fixture's failure channel (bearing4) ramps up in amplitude/impulsiveness over
    the run — the extracted RMS should reflect that, i.e. this is a real signal a model
    could learn from, not just noise."""
    df = build_bearing_feature_matrix(
        bearing_raw_dir, channels=CHANNELS, failure_channel="bearing4"
    )
    early_rms = df["bearing4_rms"].iloc[: len(df) // 4].mean()
    late_rms = df["bearing4_rms"].iloc[-len(df) // 4 :].mean()
    assert late_rms > early_rms

    # A healthy (non-failure) channel should show no comparable trend.
    early_healthy = df["bearing1_rms"].iloc[: len(df) // 4].mean()
    late_healthy = df["bearing1_rms"].iloc[-len(df) // 4 :].mean()
    assert late_rms - early_rms > late_healthy - early_healthy


def test_build_bearing_feature_matrix_rejects_unknown_failure_channel(bearing_raw_dir):
    with pytest.raises(ValueError):
        build_bearing_feature_matrix(
            bearing_raw_dir, channels=CHANNELS, failure_channel="bearing99"
        )
