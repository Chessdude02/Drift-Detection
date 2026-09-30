"""The drift job's decision policy and window handling (pdm.drift.run_drift_check)."""

from __future__ import annotations

import time

import pytest

from pdm.common.config import load_yaml
from pdm.drift.run_drift_check import count_engines, decide_action, read_window, run_check
from pdm.serving.inference_log import InferenceLog

pytestmark = pytest.mark.unit

CFG = {
    "current_window": {"min_engines": 5, "lookback_rows": 500},
    "drift": {"threshold": 0.5},
    "sensor_check": {"hold_retrain_on_fault": True},
}


@pytest.mark.parametrize(
    "n_engines,share,faulty,expected",
    [
        (3, 0.9, [], "skip_too_few_engines"),
        (3, None, [], "skip_too_few_engines"),
        (None, 0.9, [], "retrain"),  # legacy rows without asset ids: cannot enforce
        (10, 0.2, [], "none"),
        (10, 0.2, ["s1"], "none"),  # sensor alert alone never retrains
        (10, 0.9, [], "retrain"),
        (10, 0.9, ["s1"], "hold_for_sensor_fault"),
    ],
)
def test_decide_action(n_engines, share, faulty, expected):
    assert decide_action(n_engines, share, faulty, CFG) == expected


def test_hold_can_be_switched_off():
    cfg = {**CFG, "sensor_check": {"hold_retrain_on_fault": False}}
    assert decide_action(10, 0.9, ["s1"], cfg) == "retrain"


def test_count_engines_ignores_missing_ids():
    assert count_engines([{"asset_id": "a"}, {"asset_id": "b"}, {"asset_id": None}]) == 2
    assert count_engines([{"asset_id": None}]) is None


def test_read_window_by_time_drops_shadow(tmp_path):
    log = InferenceLog(tmp_path / "log.db")
    log.record({"x": 1.0}, 1.0, asset_id="a")
    log.record({"x": 1.0}, 1.0, asset_id="b", shadow=True)
    rows = read_window(log, {"lookback_hours": 1, "lookback_rows": 10})
    assert [r["asset_id"] for r in rows] == ["a"]
    assert read_window(log, {"lookback_hours": None, "lookback_rows": 10})[0]["asset_id"] == "a"
    # read_since boundary: a window that starts in the future sees nothing
    assert log.read_since(time.time() + 3600) == []


def test_run_check_skips_small_windows_without_evaluating():
    rows = [{"asset_id": f"e{i % 2}", "features": {}, "prediction": 1.0} for i in range(50)]
    outcome = run_check(reference_df=None, rows=rows, config=load_yaml("drift.yaml"))
    assert outcome["action"] == "skip_too_few_engines"
    assert outcome["evaluation"] is None and outcome["n_engines"] == 2


def test_run_check_with_no_rows():
    assert run_check(None, [], load_yaml("drift.yaml"))["action"] == "skip_no_data"
