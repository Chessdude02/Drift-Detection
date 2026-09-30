"""Maintenance-decision scoring, the interval model, the champion/challenger decision and
the serving helper that unpacks interval output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.evaluation.champion_challenger import decide
from pdm.evaluation.decision import choose_threshold, simulate_policy
from pdm.serving.inference import predict_details, predict_row
from pdm.training.rul_model import RULIntervalModel, RULPyfunc, point_predictions

pytestmark = pytest.mark.unit

COSTS = {"unplanned_failure": 100.0, "planned_maintenance": 10.0, "per_wasted_cycle": 0.1}


def _trajectories(lengths):
    rows = []
    for unit, n in enumerate(lengths, start=1):
        for cycle in range(1, n + 1):
            rows.append({"unit_number": unit, "time_in_cycles": cycle, "true_rul": n - cycle})
    return pd.DataFrame(rows)


def test_perfect_signal_plans_every_engine_with_no_waste():
    traj = _trajectories([50, 80])
    result = simulate_policy(traj, traj["true_rul"].to_numpy(), 10, lead_time=10, costs=COSTS)
    assert result["planned"] == 2 and result["unplanned_failures"] == 0
    assert result["mean_wasted_cycles"] == 0.0
    assert result["cost_per_engine"] == pytest.approx(10.0)


def test_late_and_missed_count_as_failures():
    traj = _trajectories([50, 50])
    signal = np.where(traj["unit_number"] == 1, traj["true_rul"], 999.0)  # unit 2 never fires
    result = simulate_policy(traj, signal, 5, lead_time=10, costs=COSTS)  # unit 1 fires too late
    assert result["late"] == 1 and result["missed"] == 1
    assert result["unplanned_failures"] == 2
    assert result["cost_per_engine"] == pytest.approx(100.0)


def test_choose_threshold_trades_waste_against_failures():
    traj = _trajectories([60, 60, 60])
    signal = traj["true_rul"].to_numpy() + 5.0  # model is 5 cycles optimistic
    best = choose_threshold(traj, signal, range(5, 40), lead_time=10, costs=COSTS)
    assert best["unplanned_failures"] == 0
    assert best["threshold"] == 15  # fires at true RUL 10: in time, nothing wasted


def test_interval_model_reaches_target_coverage():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 10, 4000)})
    y = pd.Series(3 * X["x"] + rng.normal(scale=1 + X["x"] / 2, size=4000))
    fit, cal, test = slice(0, 2000), slice(2000, 3000), slice(3000, 4000)
    params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "verbose": -1}
    model = RULIntervalModel.fit("lightgbm", params, X[fit], y[fit], X[cal], y[cal], coverage=0.9)
    frame = model.predict_frame(X[test])
    inside = (y[test] >= frame["rul_lower"]) & (y[test] <= frame["rul_upper"])
    assert 0.85 <= inside.mean() <= 0.97
    assert (frame["rul_lower"] <= frame["rul"]).all() and (frame["rul"] <= frame["rul_upper"]).all()
    assert frame["maintenance_recommended"].isna().all()  # no threshold set yet

    model.maintenance_threshold = 5.0
    flagged = model.predict_frame(X[test])["maintenance_recommended"]
    assert set(flagged.unique()) <= {0.0, 1.0}
    np.testing.assert_allclose(point_predictions(frame), frame["rul"])


def test_serving_helpers_unpack_interval_output():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(size=300), "b": rng.uniform(size=300)})
    y = pd.Series(10 * X["a"])
    params = {"n_estimators": 20, "verbose": -1}
    bundle = RULIntervalModel.fit("lightgbm", params, X[:200], y[:200], X[200:], y[200:], 0.9)
    bundle.maintenance_threshold = 100.0
    model = RULPyfunc(bundle)

    class Wrapped:  # the shape mlflow.pyfunc gives the serving app
        def predict(self, df):
            return model.predict(None, df)

    details = predict_details(Wrapped(), {"a": 0.5, "b": 0.1}, ["a", "b"])
    assert details["rul_lower"] <= details["rul"] <= details["rul_upper"]
    assert details["maintenance_recommended"] is True
    assert predict_row(Wrapped(), {"a": 0.5, "b": 0.1}, ["a", "b"]) == details["rul"]


GATE = {
    "max_cost_increase": 0.05,
    "max_extra_failures": 0,
    "max_rmse_increase": 1.0,
    "max_test_rmse": 30.0,
    "max_coverage_shortfall": 0.1,
}
CHAMPION = {
    "test_rmse": 22.0,
    "interval_coverage": 0.9,
    "cost_per_engine": 12.0,
    "unplanned_failures": 0.0,
}


def test_gate_approves_a_no_worse_challenger():
    challenger = {**CHAMPION, "cost_per_engine": 12.5, "test_rmse": 22.8}
    assert decide(challenger, CHAMPION, GATE, 0.9).approved


@pytest.mark.parametrize(
    "change",
    [
        {"unplanned_failures": 1.0},
        {"cost_per_engine": 13.0},
        {"test_rmse": 23.5},
        {"interval_coverage": 0.75},
    ],
)
def test_gate_rejects_any_regression(change):
    assert not decide({**CHAMPION, **change}, CHAMPION, GATE, 0.9).approved


def test_gate_absolute_floor_applies_even_when_champion_is_worse():
    bad = {**CHAMPION, "test_rmse": 35.0}
    assert not decide(bad, {**bad, "test_rmse": 36.0}, GATE, 0.9).approved


def test_first_promotion_needs_a_human():
    assert not decide(CHAMPION, None, GATE, 0.9).approved
    assert decide(CHAMPION, None, GATE, 0.9, bootstrap_confirmed=True).approved
