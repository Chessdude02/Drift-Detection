"""The whole loop, in process, on a synthetic fleet (no Docker/Kubernetes needed):

train v1 -> first promotion needs a human -> serve traffic with engine ids -> drift check
(normal fleet: nothing; new fleet: retrain; broken sensor: hold) -> outcomes imported ->
labels built -> retrain (train on base + labels, champion/challenger gate) -> rollback.

Every stage is the production code path; only the data is synthetic and small, so the
test checks the plumbing and the decisions between stages, not model quality (that is
measured on real NASA data, see docs/run_log.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_cmapss import make_fleet, write_raw_dir  # noqa: E402

pytestmark = pytest.mark.integration

MODEL = "cmapss_rul"  # the name serving.yaml loads; the tracking store is a temp dir


@pytest.fixture
def env(tmp_path, monkeypatch, mlflow_tracking_uri):
    from pdm.common.config import get_settings, load_yaml

    monkeypatch.setenv("INFERENCE_LOG_DB", str(tmp_path / "log.db"))
    monkeypatch.setenv("OUTCOME_DB", str(tmp_path / "outcomes.db"))
    monkeypatch.setenv("LABELS_DIR", str(tmp_path / "labels"))
    get_settings.cache_clear()

    train = make_fleet(40, seed=1)
    write_raw_dir(tmp_path / "raw", train, make_fleet(15, seed=2))
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"units": list(range(33, 41))}))

    cfg = load_yaml("training.yaml")
    cfg["dataset"]["holdout_units_file"] = str(holdout)
    cfg["model"]["params"].update({"n_estimators": 40, "min_child_samples": 5})
    cfg["labels"]["dir"] = str(tmp_path / "labels")
    cfg["mlflow"] = {"experiment_name": "e2e", "registered_model_name": MODEL}

    drift = load_yaml("drift.yaml")
    drift["reference"]["model_name"] = MODEL
    yield {"tmp": tmp_path, "raw": tmp_path / "raw", "cfg": cfg, "drift": drift}
    get_settings.cache_clear()


def _serve(fleet: pd.DataFrame, prefix: str, stop: dict | None = None) -> dict:
    """Replays run-to-failure histories through the real API, interleaved by cycle."""
    from fastapi.testclient import TestClient

    from pdm.common.config import load_yaml
    from pdm.data.cmapss import ALL_COLUMNS
    from pdm.data.features import build_feature_matrix
    from pdm.serving.app import app

    fleet = fleet.copy()
    fleet.columns = ALL_COLUMNS
    feats_cfg = load_yaml("training.yaml")["features"]
    windows = feats_cfg["rolling_windows"]
    feats = build_feature_matrix(fleet, feats_cfg["sensor_columns"], windows, max(windows))
    if stop:
        feats = feats[feats["time_in_cycles"] <= feats["unit_number"].map(stop)]
    cols = load_yaml("serving.yaml")["feature_schema"]["required_columns"]
    statuses = []
    with TestClient(app) as http:
        for row in feats.sort_values(["time_in_cycles", "unit_number"]).itertuples(index=False):
            resp = http.post(
                "/predict",
                json={
                    "asset_id": f"{prefix}-{row.unit_number}",
                    "cycle": int(row.time_in_cycles),
                    "features": {c: float(getattr(row, c)) for c in cols},
                },
            )
            statuses.append(resp.status_code)
            last = resp.json()
    return {"statuses": statuses, "last": last}


def _check(env) -> dict:
    from pdm.common.config import get_settings
    from pdm.drift.reference import load_reference
    from pdm.drift.run_drift_check import read_window, run_check
    from pdm.serving.inference_log import InferenceLog

    reference, source = load_reference(env["drift"]["reference"])
    rows = read_window(
        InferenceLog(get_settings().inference_log_db), env["drift"]["current_window"]
    )
    return {"source": source, **run_check(reference, rows, env["drift"])}


def test_full_loop(env):
    import mlflow

    from pdm.evaluation.champion_challenger import run_gate
    from pdm.evaluation.registry import (
        get_production_version,
        promote_version_with_metrics,
        rollback_to_previous,
    )
    from pdm.labels.outcomes import OutcomeStore
    from pdm.training.retrain import retrain
    from pdm.training.train import run_training

    client = mlflow.MlflowClient()
    raw, cfg = env["raw"], env["cfg"]

    # 1. first model: the gate refuses without a human, then promotes with one
    v1 = run_training(raw, cfg)["model_version"]
    assert not run_gate(v1, raw, cfg, promote=True)["promoted"]
    assert run_gate(v1, raw, cfg, promote=True, confirm_bootstrap=True)["promoted"]
    assert str(get_production_version(client, MODEL).version) == str(v1)

    # 2. traffic from the same kind of fleet: served, logged, no drift action
    same = _serve(make_fleet(8, seed=3, first_unit=101), "same")
    assert set(same["statuses"]) == {200}
    body = same["last"]
    assert body["rul_lower"] <= body["predicted_rul"] <= body["rul_upper"]
    assert isinstance(body["maintenance_recommended"], bool)
    normal = _check(env)
    assert f"{MODEL} v{v1}" in normal["source"]
    assert normal["n_engines"] >= 5
    assert normal["action"] == "none"

    # 3. a new fleet with a different fault mode and operating level: retrain, no hold
    other = np.random.default_rng(99).normal(size=21)
    new_fleet = make_fleet(12, seed=4, wear_direction=other, level_shift=1.5, first_unit=201)
    lengths = new_fleet.groupby(0)[1].max().to_dict()
    stop = {u: (n if u % 3 else int(n * 0.5)) for u, n in lengths.items()}  # every 3rd maintained
    _serve(new_fleet, "new", stop=stop)
    shifted = _check(env)
    assert shifted["action"] == "retrain"
    assert shifted["evaluation"]["sensor_verdict"] in ("system_wide_shift", "ok")

    # 4. outcomes arrive from the maintenance system; retrain uses them as labels
    csv = env["tmp"] / "cmms.csv"
    csv.write_text(
        "asset_id,event_type,cycle\n"
        + "".join(f"new-{u},{'failure' if u % 3 else 'maintenance'},{stop[u]}\n" for u in stop)
    )
    assert OutcomeStore(env["tmp"] / "outcomes.db").import_csv(csv)["added"] == len(stop)
    result = retrain(raw, cfg, promote=True)
    assert result["labels"]["stats"]["rows_labelled"] > 0
    assert result["labels"]["stats"]["rows_failure"] > 0
    run = client.get_run(result["training"]["run_id"])
    assert int(run.data.params["outcome_label_rows"]) == result["labels"]["stats"]["rows_labelled"]
    gate = result["gate"]
    assert gate["champion_version"] == str(v1) and gate["reasons"]

    # 5. rollback puts v1 back. If the gate kept v1, promote the candidate by hand first
    # so the rollback path is exercised either way.
    candidate = gate["candidate_version"]
    if not gate["promoted"]:
        promote_version_with_metrics(client, MODEL, candidate, gate["challenger"], "e2e")
    assert str(get_production_version(client, MODEL).version) == candidate
    assert str(rollback_to_previous(client, MODEL).version) == str(v1)
    assert str(get_production_version(client, MODEL).version) == str(v1)


def test_broken_sensor_holds_retrain_and_names_the_sensor(env):
    from pdm.drift.reference import load_reference
    from pdm.drift.run_drift_check import decide_action, evaluate_window
    from pdm.evaluation.champion_challenger import run_gate
    from pdm.training.train import run_training

    raw, cfg = env["raw"], env["cfg"]
    v1 = run_training(raw, cfg)["model_version"]
    run_gate(v1, raw, cfg, promote=True, confirm_bootstrap=True)
    reference, _ = load_reference(env["drift"]["reference"])
    cols = env["drift"]["drift"]["columns"]

    window = reference.sample(500, random_state=0).copy()
    window["prediction"] = window["predicted_rul"]
    broken = cols[3]
    window[broken] = window[broken] + 2 * reference[broken].std()
    result = evaluate_window(reference, window, cols, env["drift"])
    assert result["sensor_verdict"] == "sensor_fault"
    assert result["faulty_sensors"][0] == broken
    # even if drift were over the threshold, a named culprit holds the retrain
    assert decide_action(10, 0.9, result["faulty_sensors"], env["drift"]) == "hold_for_sensor_fault"
