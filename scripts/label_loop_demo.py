"""Runs the whole feedback loop on real NASA data, with FD003 (a fleet with a second
fault mode the FD001 model never saw) arriving as live traffic:

1. score the current Production model on FD003 engines kept aside for evaluation;
2. replay other FD003 engines through the real serving API (pdm.serving.app), a reading
   per engine per cycle, interleaved like a fleet: some run to failure, some are
   maintained early, some are still running;
3. run the real drift check on the latest window;
4. write their outcomes as a maintenance-system CSV and import it (pdm.labels.outcomes);
5. retrain (pdm.training.retrain): build labels, train on FD001 + labels, gate;
6. score the new candidate on the same kept-aside FD003 engines.

Needs a Production model in MLFLOW_TRACKING_URI. Writes to INFERENCE_LOG_DB,
OUTCOME_DB and LABELS_DIR (set them to scratch paths). Report: --json-out.

Usage:
    python scripts/label_loop_demo.py --raw-dir data/raw --json-out reports/label_loop_fd003.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from pdm.common.config import configure_mlflow_env, get_settings, load_yaml  # noqa: E402
from pdm.data.cmapss import load_train  # noqa: E402
from pdm.data.features import build_feature_matrix, cap_rul  # noqa: E402
from pdm.evaluation.decision import simulate_policy  # noqa: E402
from pdm.training.evaluate import rmse  # noqa: E402


def fleet_features(raw_dir: Path, subset: str, features_cfg: dict) -> pd.DataFrame:
    df = load_train(raw_dir, subset).sort_values(["unit_number", "time_in_cycles"])
    windows = features_cfg["rolling_windows"]
    feats = build_feature_matrix(df, features_cfg["sensor_columns"], windows, max(windows))
    feats["true_rul"] = df["rul"].to_numpy()
    return feats


def score_on_engines(model, engines: pd.DataFrame, cols, decision_cfg, threshold, rul_cap):
    out = model.predict(engines[cols])
    signal = out["rul_lower"].where(out["rul_lower"].notna(), out["rul"]).to_numpy()
    policy = simulate_policy(
        engines, signal, threshold, decision_cfg["lead_time_cycles"], decision_cfg["costs"]
    )
    return {
        "rmse_all_cycles": rmse(cap_rul(engines["true_rul"], rul_cap), out["rul"]),
        "rmse_last_60_cycles": rmse(
            cap_rul(engines["true_rul"], rul_cap)[engines["true_rul"] < 60],
            out["rul"][engines["true_rul"].to_numpy() < 60],
        ),
        "cost_per_engine": policy["cost_per_engine"],
        "unplanned_failures": policy["unplanned_failures"],
        "missed": policy["missed"],
        "late": policy["late"],
        "mean_wasted_cycles": policy["mean_wasted_cycles"],
        "n_engines": policy["n_engines"],
    }


def plan_fleet(units, rng, lengths) -> dict:
    """Each live engine's fate: fail (replayed to its last cycle), maintain early, or
    still running (replayed to a random point, no outcome yet)."""
    plan = {}
    for u in units:
        fate = rng.choice(["failure", "maintenance", "running"], p=[0.5, 0.3, 0.2])
        last = lengths[u]
        stop = last if fate == "failure" else int(rng.uniform(0.3, 0.9) * last)
        plan[int(u)] = {
            "fate": str(fate),
            "stop_cycle": int(stop),
            "offset": int(rng.integers(0, 150)),
        }
    return plan


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--subset", default="FD003")
    parser.add_argument("--n-live", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    configure_mlflow_env()
    import mlflow
    from fastapi.testclient import TestClient
    from mlflow import MlflowClient

    from pdm.drift.reference import load_reference
    from pdm.drift.run_drift_check import read_window, run_check
    from pdm.evaluation.registry import get_production_version
    from pdm.labels.outcomes import OutcomeStore
    from pdm.serving.app import app
    from pdm.serving.inference_log import InferenceLog
    from pdm.training.retrain import retrain

    settings = get_settings()
    training_cfg = load_yaml("training.yaml")
    drift_cfg = load_yaml("drift.yaml")
    decision_cfg = load_yaml(training_cfg["decision_config"])
    rul_cap = training_cfg["dataset"]["rul_cap"]
    name = training_cfg["mlflow"]["registered_model_name"]
    cols = load_yaml("serving.yaml")["feature_schema"]["required_columns"]
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    fleet = fleet_features(Path(args.raw_dir), args.subset, training_cfg["features"])
    units = np.random.default_rng(args.seed).permutation(fleet["unit_number"].unique())
    live_units, eval_units = units[: args.n_live], units[args.n_live :]
    eval_engines = fleet[fleet["unit_number"].isin(eval_units)]

    def score_version(version) -> dict:
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        threshold = client.get_run(client.get_model_version(name, version).run_id).data.metrics[
            "maintenance_threshold"
        ]
        return score_on_engines(model, eval_engines, cols, decision_cfg, threshold, rul_cap)

    prod_before = get_production_version(client, name)
    report = {
        "subset": args.subset,
        "live_engines": len(live_units),
        "eval_engines": len(eval_units),
        "before": {"version": str(prod_before.version), **score_version(prod_before.version)},
    }

    # 2. replay the live fleet through the serving API
    rng = np.random.default_rng(args.seed + 1)
    lengths = fleet.groupby("unit_number")["time_in_cycles"].max().to_dict()
    plan = plan_fleet(live_units, rng, lengths)
    live = fleet[fleet["unit_number"].isin(live_units)].copy()
    live["stop"] = live["unit_number"].map(lambda u: plan[u]["stop_cycle"])
    live = live[live["time_in_cycles"] <= live["stop"]]
    live["clock"] = live["time_in_cycles"] + live["unit_number"].map(lambda u: plan[u]["offset"])
    live = live.sort_values(["clock", "unit_number"])
    sent = rejected = 0
    with TestClient(app) as http:
        for row in live.itertuples(index=False):
            body = {
                "asset_id": f"{args.subset.lower()}-{row.unit_number}",
                "cycle": int(row.time_in_cycles),
                "features": {c: float(getattr(row, c)) for c in cols},
            }
            resp = http.post("/predict", json=body)
            sent += 1
            rejected += resp.status_code != 200
    report["replay"] = {
        "requests": sent,
        "rejected": rejected,
        "fates": pd.Series([p["fate"] for p in plan.values()]).value_counts().to_dict(),
    }

    # 3. drift check on the latest window, exactly as the CronJob does it
    reference, _ = load_reference(drift_cfg["reference"])
    window = read_window(InferenceLog(settings.inference_log_db), drift_cfg["current_window"])
    outcome = run_check(reference, window, drift_cfg)
    evaluation = outcome["evaluation"] or {}
    report["drift_check"] = {
        "action": outcome["action"],
        "engines_in_window": outcome["n_engines"],
        "drift_share": (evaluation.get("drift") or {}).get("share_of_drifted_columns"),
        "faulty_sensors": evaluation.get("faulty_sensors"),
        "sensor_verdict": evaluation.get("sensor_verdict"),
    }

    # 4. outcomes arrive as a maintenance-system export
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp) / "cmms_export.csv"
        with open(export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["asset_id", "event_type", "cycle"])
            for unit, p in plan.items():
                if p["fate"] != "running":
                    w.writerow([f"{args.subset.lower()}-{unit}", p["fate"], p["stop_cycle"]])
        report["outcome_import"] = OutcomeStore(settings.outcome_db).import_csv(export)

    # 5. retrain on FD001 + outcome labels, gate against Production
    result = retrain(Path(args.raw_dir), training_cfg, promote=True)
    report["labels"] = result["labels"]["stats"] if result["labels"] else None
    report["gate"] = {
        k: result["gate"][k] for k in ("approved", "promoted", "reasons", "candidate_version")
    }
    report["after"] = {
        "version": result["gate"]["candidate_version"],
        **score_version(result["gate"]["candidate_version"]),
    }

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
