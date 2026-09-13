"""Drift-check entrypoint: compare recent inference inputs against the training reference
distribution with Evidently, push the drift score to Prometheus Pushgateway, and trigger
retraining when the configured threshold is exceeded.

Usage:
    python -m pdm.drift.run_drift_check --config-name drift.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from pdm.common.config import get_settings, load_yaml
from pdm.common.logging import setup_logging
from pdm.serving.inference_log import InferenceLog

logger = logging.getLogger(__name__)


def compute_drift_score(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]
) -> dict:
    ref = reference_df[columns]
    cur = current_df[columns]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    drift_metric = result["metrics"][0]["result"]
    return {
        "share_of_drifted_columns": float(drift_metric.get("share_of_drifted_columns", 0.0)),
        "dataset_drift": bool(drift_metric.get("dataset_drift", False)),
        "n_reference_rows": len(ref),
        "n_current_rows": len(cur),
    }


def push_drift_metric(score: float, pushgateway_url: str, job: str = "pdm_drift_check") -> None:
    registry = CollectorRegistry()
    gauge = Gauge(
        "pdm_drift_score", "Share of drifted columns from the latest drift check", registry=registry
    )
    gauge.set(score)
    try:
        push_to_gateway(pushgateway_url, job=job, registry=registry)
    except Exception:
        logger.exception("Failed to push drift metric to Pushgateway at %s", pushgateway_url)


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

    inference_log = InferenceLog(settings.inference_log_db)
    rows = inference_log.read_recent(limit=config["current_window"]["lookback_rows"])
    if not rows:
        logger.warning("No recent inference rows found; skipping drift check")
        return 0
    current_df = pd.DataFrame([r["features"] for r in rows])

    columns = [
        c
        for c in config["drift"]["columns"]
        if c in reference_df.columns and c in current_df.columns
    ]
    if not columns:
        logger.error("No overlapping drift columns between reference and current data")
        return 1

    result = compute_drift_score(reference_df, current_df, columns)
    score = result["share_of_drifted_columns"]
    logger.info("Drift check result: %s", result)

    if not args.dry_run:
        push_drift_metric(score, settings.prometheus_pushgateway_url)

    trigger_retrain_if_needed(score, config["drift"]["threshold"], config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
