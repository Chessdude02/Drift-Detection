"""Drift-check entrypoint: compare recent inference inputs against the training reference
distribution, push the results to Prometheus Pushgateway, and trigger retraining when the
configured threshold is exceeded.

Two separate checks run on each window (see evaluate_window):
- Retrain trigger: share of drifted columns (Evidently), against a reference resampled to
  the window's life-stage mix (pdm.drift.life_stage), so a fleet that is simply younger
  or older than the reference does not look like drift.
- Sensor-fault alert: per-sensor residual check (pdm.drift.sensor_check). It only raises
  an alert (pdm_sensor_fault gauge); it never triggers retraining.

Usage:
    python -m pdm.drift.run_drift_check --config-name drift.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from pdm.common.config import get_settings, load_yaml
from pdm.common.logging import setup_logging
from pdm.drift.life_stage import REFERENCE_PREDICTION_COLUMN, match_life_stage
from pdm.drift.sensor_check import faulty_sensors, sensor_fault_scores
from pdm.serving.inference_log import InferenceLog

logger = logging.getLogger(__name__)

CURRENT_PREDICTION_COLUMN = "prediction"


def compute_drift_score(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    stattest: str | None = None,
    stattest_threshold: float | None = None,
) -> dict:
    """`stattest`/`stattest_threshold` are passed to Evidently's per-column drift test;
    None keeps Evidently's defaults. config/drift.yaml sets calibrated values - see
    scripts/calibrate_drift.py for why the defaults are unusable on C-MAPSS.
    """
    ref = reference_df[columns]
    cur = current_df[columns]

    report = Report(
        metrics=[DataDriftPreset(stattest=stattest, stattest_threshold=stattest_threshold)]
    )
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    drift_metric = result["metrics"][0]["result"]
    return {
        "share_of_drifted_columns": float(drift_metric.get("share_of_drifted_columns", 0.0)),
        "dataset_drift": bool(drift_metric.get("dataset_drift", False)),
        "n_reference_rows": len(ref),
        "n_current_rows": len(cur),
    }


def evaluate_window(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str], config: dict
) -> dict:
    """Runs both checks on one window. `current_df` needs a `prediction` column for
    life-stage matching and `reference_df` a `predicted_rul` column (written by
    scripts/seed_reference_data.py); if either is missing, matching is skipped with a
    warning and the plain reference is used.
    """
    drift_cfg = config["drift"]
    matching_cfg = config.get("life_stage_matching", {})
    sensor_cfg = config.get("sensor_check", {})

    drift_reference = reference_df
    matched, coverage = False, None
    if matching_cfg.get("enabled"):
        if (
            REFERENCE_PREDICTION_COLUMN in reference_df.columns
            and CURRENT_PREDICTION_COLUMN in current_df.columns
        ):
            drift_reference, coverage = match_life_stage(
                reference_df,
                current_df[CURRENT_PREDICTION_COLUMN],
                bin_width=matching_cfg.get("bin_width", 10),
                max_rul=matching_cfg.get("max_rul", 130),
                sample_rows=matching_cfg.get("sample_rows", 5000),
                seed=matching_cfg.get("seed", 0),
            )
            matched = coverage > 0
        else:
            logger.warning(
                "Life-stage matching enabled but reference lacks %r or current window lacks "
                "%r; using the unmatched reference (expect false alarms if the fleet's age "
                "mix differs). Re-seed the reference with scripts/seed_reference_data.py.",
                REFERENCE_PREDICTION_COLUMN,
                CURRENT_PREDICTION_COLUMN,
            )

    drift = compute_drift_score(
        drift_reference,
        current_df,
        columns,
        stattest=drift_cfg.get("stattest"),
        stattest_threshold=drift_cfg.get("stattest_threshold"),
    )

    scores: dict[str, float] = {}
    flagged: list[str] = []
    if sensor_cfg.get("enabled"):
        scores = sensor_fault_scores(reference_df, current_df, columns)
        flagged = faulty_sensors(scores, sensor_cfg["threshold"])

    return {
        "drift": drift,
        "life_stage_matched": matched,
        "life_stage_coverage": coverage,
        "sensor_scores": scores,
        "faulty_sensors": flagged,
    }


def push_metrics(
    drift_score: float | None,
    sensor_scores: dict[str, float],
    flagged_sensors: list[str],
    pushgateway_url: str,
    action: str,
    n_engines: int | None,
    job: str = "pdm_drift_check",
) -> None:
    """`action` is the outcome of decide_action; it is pushed as one-hot gauges so
    alerts can fire on a held retrain or a check that keeps being skipped."""
    registry = CollectorRegistry()
    if drift_score is not None:
        Gauge(
            "pdm_drift_score",
            "Share of drifted columns from the latest drift check",
            registry=registry,
        ).set(drift_score)
    if sensor_scores:
        score_gauge = Gauge(
            "pdm_sensor_fault_score",
            "Per-sensor residual drift score from the latest drift check",
            ["sensor"],
            registry=registry,
        )
        fault_gauge = Gauge(
            "pdm_sensor_fault",
            "1 if the sensor's residual score exceeded sensor_check.threshold, else 0",
            ["sensor"],
            registry=registry,
        )
        for sensor, value in sensor_scores.items():
            score_gauge.labels(sensor=sensor).set(value)
            fault_gauge.labels(sensor=sensor).set(1 if sensor in flagged_sensors else 0)
    action_gauge = Gauge(
        "pdm_drift_check_action",
        "1 for the latest drift check's outcome: none, retrain, hold_for_sensor_fault, "
        "skip_too_few_engines or skip_no_data",
        ["action"],
        registry=registry,
    )
    for name in ACTIONS:
        action_gauge.labels(action=name).set(1 if name == action else 0)
    if n_engines is not None:
        Gauge(
            "pdm_drift_window_engines",
            "Distinct engines in the latest drift-check window",
            registry=registry,
        ).set(n_engines)
    try:
        push_to_gateway(pushgateway_url, job=job, registry=registry)
    except Exception:
        logger.exception("Failed to push drift metrics to Pushgateway at %s", pushgateway_url)


ACTIONS = ("none", "retrain", "hold_for_sensor_fault", "skip_too_few_engines", "skip_no_data")


def count_engines(rows: list[dict]) -> int | None:
    """Distinct asset ids in the window, or None if no row carries one (legacy clients),
    in which case the minimum cannot be enforced."""
    ids = {r["asset_id"] for r in rows if r.get("asset_id")}
    return len(ids) if ids else None


def decide_action(
    n_engines: int | None,
    drift_share: float | None,
    faulty: list[str],
    config: dict,
) -> str:
    """What the drift job should do. Pure, so the policy is testable:
    - skip_too_few_engines: under current_window.min_engines, both checks give 15-60%
      false alarms (decisions.md D8), so no decision is made at all.
    - hold_for_sensor_fault: drift says retrain, but a sensor also looks broken.
      Retraining on a broken sensor's data would bake the fault in; a human decides.
    """
    min_engines = config["current_window"].get("min_engines")
    if min_engines and n_engines is not None and n_engines < min_engines:
        return "skip_too_few_engines"
    if drift_share is None or drift_share <= config["drift"]["threshold"]:
        return "none"
    if faulty and config.get("sensor_check", {}).get("hold_retrain_on_fault", True):
        return "hold_for_sensor_fault"
    return "retrain"


def trigger_retrain_if_needed(
    score: float, threshold: float, config: dict, dry_run: bool = False
) -> bool:
    if score <= threshold:
        logger.info("Drift score %.3f <= threshold %.3f; no retrain triggered", score, threshold)
        return False

    logger.warning("Drift score %.3f exceeds threshold %.3f; triggering retrain", score, threshold)
    if dry_run:
        logger.info("Dry run: would create retrain Job (not calling Kubernetes API)")
        return True

    from pdm.drift.trigger_retrain import trigger_retrain_job

    trigger_retrain_job(
        namespace=config["retrain_trigger"]["namespace"],
        source_cronjob=config["retrain_trigger"]["source_cronjob"],
        job_name_prefix=config["retrain_trigger"]["job_name_prefix"],
    )
    return True


def read_window(inference_log: InferenceLog, window_cfg: dict) -> list[dict]:
    """Non-shadow rows of the current window: the last `lookback_hours` (capped at
    `lookback_rows`) when lookback_hours is set, else the last `lookback_rows` rows.

    Shadow rows are mirrored copies of live requests scored by a candidate model:
    counting them would double-count inputs and mix another model's predictions into
    life-stage matching.
    """
    hours = window_cfg.get("lookback_hours")
    limit = window_cfg["lookback_rows"]
    if hours:
        rows = inference_log.read_since(time.time() - hours * 3600, limit=limit)
    else:
        rows = inference_log.read_recent(limit=limit)
    return [r for r in rows if not r["shadow"]]


def run_check(reference_df: pd.DataFrame, rows: list[dict], config: dict) -> dict:
    """One drift check over already-read rows. Returns the evaluation plus the action."""
    n_engines = count_engines(rows)
    if not rows:
        return {"action": "skip_no_data", "n_engines": 0, "evaluation": None}
    if n_engines is None:
        logger.warning(
            "No asset_id on any row in the window; cannot enforce min_engines. Clients "
            "should send asset_id (serving.yaml input_validation.require_asset_id)."
        )
    action = decide_action(n_engines, None, [], config)
    if action == "skip_too_few_engines":
        logger.warning(
            "Only %s engines in the window (< min_engines=%s); skipping drift check",
            n_engines,
            config["current_window"]["min_engines"],
        )
        return {"action": action, "n_engines": n_engines, "evaluation": None}

    current_df = pd.DataFrame(
        [{**r["features"], CURRENT_PREDICTION_COLUMN: r["prediction"]} for r in rows]
    )
    columns = [
        c
        for c in config["drift"]["columns"]
        if c in reference_df.columns and c in current_df.columns
    ]
    if not columns:
        raise ValueError("No overlapping drift columns between reference and current data")
    evaluation = evaluate_window(reference_df, current_df, columns, config)
    action = decide_action(
        n_engines,
        evaluation["drift"]["share_of_drifted_columns"],
        evaluation["faulty_sensors"],
        config,
    )
    return {"action": action, "n_engines": n_engines, "evaluation": evaluation}


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run an Evidently drift check against recent inference data."
    )
    parser.add_argument("--config-name", default="drift.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't call the Kubernetes API or Pushgateway"
    )
    args = parser.parse_args()

    config = load_yaml(args.config_name)
    settings = get_settings()

    reference_df = pd.read_parquet(config["reference"]["path"])
    if len(reference_df) < config["reference"]["min_reference_rows"]:
        logger.error(
            "Reference dataset too small (%d rows); aborting drift check", len(reference_df)
        )
        return 1

    rows = read_window(InferenceLog(settings.inference_log_db), config["current_window"])
    try:
        outcome = run_check(reference_df, rows, config)
    except ValueError:
        logger.exception("Drift check failed")
        return 1

    action, evaluation = outcome["action"], outcome["evaluation"]
    score = evaluation["drift"]["share_of_drifted_columns"] if evaluation else None
    sensor_scores = evaluation["sensor_scores"] if evaluation else {}
    faulty = evaluation["faulty_sensors"] if evaluation else []
    logger.info(
        "Drift check: action=%s engines=%s drift=%s",
        action,
        outcome["n_engines"],
        evaluation["drift"] if evaluation else None,
    )
    if faulty:
        logger.warning(
            "Suspected sensor fault (inspect, do not retrain): %s",
            {s: round(sensor_scores[s], 3) for s in faulty},
        )
    if action == "hold_for_sensor_fault":
        logger.warning(
            "Drift %.3f is over the retrain threshold, but retraining is HELD because a "
            "sensor looks faulty. Fix or rule out the sensor, then trigger the retrain by hand.",
            score,
        )

    if not args.dry_run:
        push_metrics(
            score,
            sensor_scores,
            faulty,
            settings.prometheus_pushgateway_url,
            action=action,
            n_engines=outcome["n_engines"],
        )
    if action == "retrain":
        trigger_retrain_if_needed(score, config["drift"]["threshold"], config, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
