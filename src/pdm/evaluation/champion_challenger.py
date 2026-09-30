"""C-MAPSS champion/challenger gate: re-score the Production model and a candidate on the
same frozen data and promote the candidate only if it is no worse on every check in
config/champion_challenger.yaml.

Why re-score instead of comparing against numbers saved at the last promotion: both
models are judged on identical data by identical code, so a change in the scoring code
or the data can never make an old baseline incomparable.

Usage:
    python -m pdm.evaluation.champion_challenger --candidate-version 7 [--promote]
    python -m pdm.evaluation.champion_challenger --candidate-version 1 --promote --confirm-bootstrap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from pdm.common.config import configure_mlflow_env, load_yaml
from pdm.common.logging import setup_logging
from pdm.data.cmapss import load_test
from pdm.data.datasets import load_cmapss_holdout
from pdm.data.features import build_feature_matrix, cap_rul
from pdm.evaluation.decision import simulate_policy
from pdm.evaluation.registry import get_production_version, promote_version_with_metrics
from pdm.training.evaluate import rmse

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    approved: bool
    reasons: list[str]


def load_evaluation_data(raw_dir: Path, training_cfg: dict) -> dict:
    dataset_cfg, features_cfg = training_cfg["dataset"], training_cfg["features"]
    holdout, cols = load_cmapss_holdout(raw_dir, dataset_cfg, features_cfg)
    test_df, true_rul = load_test(raw_dir, dataset_cfg["subset"])
    windows = features_cfg["rolling_windows"]
    feats = build_feature_matrix(test_df, features_cfg["sensor_columns"], windows, max(windows))
    last = feats.groupby("unit_number").tail(1).set_index("unit_number").sort_index()
    y_test = true_rul.loc[last.index]
    if dataset_cfg.get("rul_cap") is not None:
        y_test = cap_rul(y_test, dataset_cfg["rul_cap"])
    return {"holdout": holdout, "test_last": last, "y_test": y_test, "cols": cols}


def score_model(model, data: dict, decision_cfg: dict, maintenance_threshold: float) -> dict:
    """`model.predict` must return a DataFrame with rul / rul_lower / rul_upper
    (pdm.training.rul_model.RULPyfunc)."""
    cols = data["cols"]
    test = model.predict(data["test_last"][cols])
    y = data["y_test"].to_numpy(dtype=float)
    inside = (y >= test["rul_lower"].to_numpy()) & (y <= test["rul_upper"].to_numpy())

    hold = model.predict(data["holdout"][cols])
    signal = hold["rul_lower"].where(hold["rul_lower"].notna(), hold["rul"]).to_numpy()
    policy = simulate_policy(
        data["holdout"],
        signal,
        maintenance_threshold,
        decision_cfg["lead_time_cycles"],
        decision_cfg["costs"],
    )
    return {
        "test_rmse": rmse(y, test["rul"]),
        "interval_coverage": float(np.mean(inside)),
        "interval_mean_width": float((test["rul_upper"] - test["rul_lower"]).mean()),
        "cost_per_engine": policy["cost_per_engine"],
        "unplanned_failures": float(policy["unplanned_failures"]),
        "mean_wasted_cycles": policy["mean_wasted_cycles"],
        "maintenance_threshold": float(maintenance_threshold),
    }


def decide(
    challenger: dict,
    champion: dict | None,
    gate_cfg: dict,
    coverage_target: float,
    bootstrap_confirmed: bool = False,
) -> GateDecision:
    reasons = []
    floor_ok = challenger["test_rmse"] <= gate_cfg["max_test_rmse"]
    reasons.append(
        f"absolute floor: test_rmse {challenger['test_rmse']:.2f} <= "
        f"{gate_cfg['max_test_rmse']:.2f} -> {'PASS' if floor_ok else 'FAIL'}"
    )
    required_cov = coverage_target - gate_cfg["max_coverage_shortfall"]
    cov_ok = challenger["interval_coverage"] >= required_cov
    reasons.append(
        f"interval coverage {challenger['interval_coverage']:.3f} >= {required_cov:.3f} "
        f"-> {'PASS' if cov_ok else 'FAIL'}"
    )
    if champion is None:
        ok = floor_ok and cov_ok and bootstrap_confirmed
        reasons.append(
            "no Production champion to compare against (first promotion): "
            + (
                "confirmed by a human (--confirm-bootstrap)"
                if bootstrap_confirmed
                else "needs --confirm-bootstrap after a human reviews the scores -> FAIL"
            )
        )
        return GateDecision(ok, reasons)

    max_cost = champion["cost_per_engine"] * (1 + gate_cfg["max_cost_increase"])
    max_fail = champion["unplanned_failures"] + gate_cfg["max_extra_failures"]
    max_rmse = champion["test_rmse"] + gate_cfg["max_rmse_increase"]
    checks = [
        ("cost_per_engine", challenger["cost_per_engine"] <= max_cost, max_cost),
        ("unplanned_failures", challenger["unplanned_failures"] <= max_fail, max_fail),
        ("test_rmse", challenger["test_rmse"] <= max_rmse, max_rmse),
    ]
    for name, ok, limit in checks:
        reasons.append(
            f"{name}: challenger {challenger[name]:.3f} vs champion {champion[name]:.3f} "
            f"(limit {limit:.3f}) -> {'PASS' if ok else 'FAIL'}"
        )
    approved = floor_ok and cov_ok and all(ok for _, ok, _ in checks)
    return GateDecision(approved, reasons)


def _threshold_of(client: MlflowClient, version) -> float:
    return float(client.get_run(version.run_id).data.metrics["maintenance_threshold"])


def run_gate(
    candidate_version: str,
    raw_dir: Path,
    training_cfg: dict,
    promote: bool = False,
    confirm_bootstrap: bool = False,
) -> dict:
    """Scores the candidate and the current Production model, decides, and promotes the
    candidate if approved and `promote` is set. Returns a JSON-able report."""
    gate_cfg = load_yaml("champion_challenger.yaml")
    decision_cfg = load_yaml(training_cfg["decision_config"])
    name = training_cfg["mlflow"]["registered_model_name"]
    coverage_target = training_cfg["model"]["intervals"]["coverage"]
    data = load_evaluation_data(Path(raw_dir), training_cfg)

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    candidate = client.get_model_version(name, str(candidate_version))
    challenger = score_model(
        mlflow.pyfunc.load_model(f"models:/{name}/{candidate.version}"),
        data,
        decision_cfg,
        _threshold_of(client, candidate),
    )
    prod = get_production_version(client, name)
    champion = None
    if prod is not None and str(prod.version) != str(candidate.version):
        champion = score_model(
            mlflow.pyfunc.load_model(f"models:/{name}/{prod.version}"),
            data,
            decision_cfg,
            _threshold_of(client, prod),
        )
    decision = decide(challenger, champion, gate_cfg, coverage_target, confirm_bootstrap)

    promoted = False
    if decision.approved and promote:
        promote_version_with_metrics(
            client, name, candidate.version, challenger, gate_cfg["holdout_version"]
        )
        promoted = True
    report = {
        "approved": decision.approved,
        "promoted": promoted,
        "reasons": decision.reasons,
        "candidate_version": str(candidate.version),
        "champion_version": str(prod.version) if champion else None,
        "challenger": challenger,
        "champion": champion,
    }
    logger.info(
        "Gate %s candidate v%s vs champion %s (promoted=%s): %s",
        "APPROVED" if decision.approved else "REJECTED",
        candidate.version,
        report["champion_version"],
        promoted,
        decision.reasons,
    )
    return report


def main() -> int:
    setup_logging()
    configure_mlflow_env()
    parser = argparse.ArgumentParser(description="C-MAPSS champion/challenger gate")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config-name", default="training.yaml")
    parser.add_argument("--promote", action="store_true", help="Promote if approved")
    parser.add_argument("--confirm-bootstrap", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    report = run_gate(
        args.candidate_version,
        Path(args.raw_dir),
        load_yaml(args.config_name),
        promote=args.promote,
        confirm_bootstrap=args.confirm_bootstrap,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0 if report["approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
