"""Life-stage matching, the per-sensor residual check, and how evaluate_window combines
them. Uses small synthetic frames; the real-data behaviour is measured by
scripts/calibrate_drift.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.common.config import load_yaml
from pdm.drift.life_stage import add_oof_predictions, match_life_stage
from pdm.drift.run_drift_check import evaluate_window
from pdm.drift.sensor_check import faulty_sensors, sensor_fault_scores

pytestmark = pytest.mark.unit

COLUMNS = ["s1", "s2", "s3", "s4"]


@pytest.fixture
def reference() -> pd.DataFrame:
    """Four sensors driven by one shared 'wear' signal plus small noise, like C-MAPSS."""
    rng = np.random.default_rng(0)
    n = 3000
    wear = rng.uniform(0, 1, n)
    df = pd.DataFrame(
        {c: wear * (i + 1) + rng.normal(scale=0.05, size=n) for i, c in enumerate(COLUMNS)}
    )
    df["predicted_rul"] = 125 * (1 - wear)
    return df


def test_match_life_stage_follows_the_current_mix(reference):
    young = np.full(500, 115.0)  # every current engine early in life
    matched, coverage = match_life_stage(reference, young, sample_rows=2000)
    assert coverage == 1.0
    assert len(matched) == 2000
    assert matched["predicted_rul"].between(110, 130).all()


def test_match_life_stage_reports_partial_coverage(reference):
    ref = reference[reference["predicted_rul"] < 60]
    current = np.r_[np.full(300, 20.0), np.full(100, 100.0)]  # 25% in a bin ref lacks
    matched, coverage = match_life_stage(ref, current, sample_rows=1000)
    assert coverage == pytest.approx(0.75)
    assert matched["predicted_rul"].between(20, 30).all()


def test_match_life_stage_without_overlap_returns_reference(reference):
    ref = reference[reference["predicted_rul"] < 60]
    matched, coverage = match_life_stage(ref, np.full(100, 100.0))
    assert coverage == 0.0
    assert matched is ref


def test_ageing_alone_does_not_flag_sensors(reference):
    old_fleet = reference[reference["predicted_rul"] < 20]
    scores = sensor_fault_scores(reference, old_fleet, COLUMNS)
    assert faulty_sensors(scores, threshold=0.5) == []


def test_single_sensor_offset_is_flagged_first(reference):
    window = reference.sample(500, random_state=1).copy()
    window["s3"] = window["s3"] + reference["s3"].std()
    scores = sensor_fault_scores(reference, window, COLUMNS)
    flagged = faulty_sensors(scores, threshold=0.5)
    assert flagged and flagged[0] == "s3"


def _config(matching: bool, sensor: bool) -> dict:
    cfg = load_yaml("drift.yaml")
    cfg["life_stage_matching"] = {**cfg["life_stage_matching"], "enabled": matching}
    cfg["sensor_check"] = {**cfg["sensor_check"], "enabled": sensor}
    return cfg


def test_evaluate_window_matches_life_stage_when_predictions_present(reference):
    window = reference[reference["predicted_rul"] > 100].sample(500, random_state=2).copy()
    window["prediction"] = window["predicted_rul"]
    matched = evaluate_window(reference, window, COLUMNS, _config(True, False))
    plain = evaluate_window(reference, window, COLUMNS, _config(False, False))
    assert matched["life_stage_matched"] is True
    assert matched["drift"]["share_of_drifted_columns"] == 0.0
    assert plain["drift"]["share_of_drifted_columns"] == 1.0
    assert matched["sensor_scores"] == {} and matched["faulty_sensors"] == []
    assert matched["sensor_verdict"] is None


def test_evaluate_window_falls_back_without_predictions(reference):
    window = reference.sample(500, random_state=3).drop(columns="predicted_rul")
    result = evaluate_window(reference, window, COLUMNS, _config(True, True))
    assert result["life_stage_matched"] is False
    assert set(result["sensor_scores"]) == set(COLUMNS)


def test_add_oof_predictions_and_seeded_reference(raw_data_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from seed_reference_data import build_reference

    training_cfg = load_yaml("training.yaml")
    reference = build_reference(raw_data_dir, training_cfg)
    assert "predicted_rul" in reference.columns
    assert reference["predicted_rul"].notna().all()

    features = reference.drop(columns="predicted_rul")
    cols = [c for c in features.columns if c not in ("unit_number", "time_in_cycles", "rul")]
    again = add_oof_predictions(features, cols, training_cfg["model"])
    np.testing.assert_allclose(again["predicted_rul"], reference["predicted_rul"])


def test_diagnose_single_sensor_fault_vs_system_wide_shift(reference):
    from pdm.drift.sensor_check import diagnose

    window = reference.sample(500, random_state=4).copy()
    assert diagnose(reference, window, COLUMNS, threshold=0.5)["verdict"] == "ok"

    broken = window.copy()
    broken["s2"] = broken["s2"] + 2 * reference["s2"].std()
    result = diagnose(reference, broken, COLUMNS, threshold=0.5)
    assert result["verdict"] == "sensor_fault" and result["culprits"] == ["s2"]

    # every sensor's relationship to the others changes: not a sensor fault
    shifted = window.copy()
    for i, col in enumerate(COLUMNS):
        shifted[col] = shifted[col] * (1 + 0.8 * (i % 2)) + i
    assert diagnose(reference, shifted, COLUMNS, threshold=0.5)["verdict"] == "system_wide_shift"
