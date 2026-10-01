"""model.calibration: cross_validation (pdm.training.train._fit_bundle_cv)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_cmapss import make_fleet  # noqa: E402

from pdm.common.config import load_yaml  # noqa: E402
from pdm.data.cmapss import ALL_COLUMNS  # noqa: E402
from pdm.data.datasets import _cmapss_split  # noqa: E402
from pdm.data.features import build_feature_matrix, cap_rul, feature_columns  # noqa: E402
from pdm.training.train import fit_bundle  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def fleet():
    raw = make_fleet(15, seed=5, min_life=40, max_life=70)
    raw.columns = ALL_COLUMNS
    raw = raw.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    cfg = load_yaml("training.yaml")
    sensors, windows = cfg["features"]["sensor_columns"], cfg["features"]["rolling_windows"]
    df = build_feature_matrix(raw, sensors, windows, max(windows))
    life = raw.groupby("unit_number")["time_in_cycles"].transform("max")
    df["true_rul"] = (life - raw["time_in_cycles"]).to_numpy()
    df["rul"] = cap_rul(df["true_rul"], 125)
    cfg["model"]["params"].update({"n_estimators": 15, "min_child_samples": 5})
    return df, feature_columns(sensors, max(windows)), cfg


def test_cv_calibration_uses_every_engine(fleet):
    df, cols, cfg = fleet
    cfg["model"]["calibration"] = "cross_validation"
    fitted = fit_bundle(df, cols, cfg, _cmapss_split)
    bundle, metrics = fitted["bundle"], fitted["metrics"]

    assert len(fitted["train_df"]) == len(df)  # final model fitted on all engines
    assert metrics["cv_folds"] == 5
    assert 0.8 <= metrics["val_interval_coverage"] <= 1.0
    assert bundle.conformal_correction == metrics["conformal_correction"]
    assert bundle.maintenance_threshold == metrics["maintenance_threshold"]
    frame = bundle.predict_frame(df[cols].head(50))
    assert (frame["rul_lower"] <= frame["rul"]).all() and (frame["rul"] <= frame["rul_upper"]).all()


def test_split_calibration_still_available(fleet):
    df, cols, cfg = fleet
    cfg["model"]["calibration"] = "split"
    fitted = fit_bundle(df, cols, cfg, _cmapss_split)
    assert len(fitted["train_df"]) < len(df)
    assert "cv_folds" not in fitted["metrics"]
