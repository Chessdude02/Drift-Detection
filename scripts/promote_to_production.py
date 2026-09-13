"""Applies the staging -> production promotion gate to a candidate model version and,
if approved, promotes it: MLflow stage transition to Production, frozen baseline_*
tags for the next comparison, and a reaffirmed image_tag for the K8s deploy step.

On rejection: prints the reasons, writes approved=false to $GITHUB_OUTPUT if present,
and exits 1. Production is never touched on rejection.

Usage:
    python scripts/promote_to_production.py --model-name ims_bearing_rul --version 7
    python scripts/promote_to_production.py --model-name ims_bearing_rul --version 1 \
        --confirm-bootstrap
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

sys.path.insert(0, "src")
from pdm.common.config import configure_mlflow_env, load_yaml  # noqa: E402
from pdm.evaluation.promotion import evaluate_promotion_gate  # noqa: E402
from pdm.evaluation.registry import (  # noqa: E402
    get_baseline_scores,
    get_image_tag,
    get_production_version,
    promote_version,
    set_image_tag,
)


def _find_latest_staging_eval(
    client: MlflowClient, experiment_name: str, candidate_run_id: str
) -> Run:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(
            f"No staging-evaluation experiment {experiment_name!r} found yet. "
            "Run the dev->staging workflow for this candidate first."
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.candidate_run_id = '{candidate_run_id}' and tags.evaluation_stage = 'staging'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(
            f"No staging evaluation found for candidate run {candidate_run_id!r} in "
            f"experiment {experiment_name!r}. Run the dev->staging workflow first."
        )
    return runs[0]


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def main() -> int:
    configure_mlflow_env()
    eval_cfg = load_yaml("evaluation.yaml")
    gate_config = load_yaml("promotion_gate.yaml")

    parser = argparse.ArgumentParser(description="Apply the staging->production promotion gate.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--version", required=True, help="Candidate's registered model version number"
    )
    parser.add_argument(
        "--confirm-bootstrap",
        action="store_true",
        help=(
            "Required to approve the FIRST-EVER promotion of this model (no existing "
            "Production baseline to compare against). Has no effect on any later "
            "promotion, which always goes through the automatic F2/precision gate."
        ),
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    candidate_version = client.get_model_version(args.model_name, args.version)

    if not candidate_version.run_id:
        raise SystemExit(
            f"Model version {args.model_name} v{args.version} has no associated run_id; "
            "cannot look up its staging evaluation."
        )

    staging_run = _find_latest_staging_eval(
        client, eval_cfg["holdout"]["staging_eval_mlflow_experiment"], candidate_version.run_id
    )
    candidate_scores = {
        "f2_score": staging_run.data.metrics["f2_score"],
        "precision": staging_run.data.metrics["precision"],
        "pr_auc": staging_run.data.metrics["pr_auc"],
    }
    holdout_version = staging_run.data.tags.get("holdout_version", "unknown")
    if holdout_version != eval_cfg["holdout"]["version"]:
        print(
            f"WARNING: candidate was staging-evaluated against holdout {holdout_version!r}, "
            f"but the currently configured holdout is {eval_cfg['holdout']['version']!r}. "
            "Scores may not be comparable to a baseline set against a different holdout."
        )

    prod_version = get_production_version(client, args.model_name)
    prod_baseline = get_baseline_scores(prod_version) if prod_version is not None else None

    decision = evaluate_promotion_gate(
        candidate_scores, prod_baseline, gate_config, bootstrap_confirmed=args.confirm_bootstrap
    )

    print(f"Candidate: {args.model_name} v{args.version} (run {candidate_version.run_id})")
    print(f"Candidate staging scores: {candidate_scores}")
    print(f"Production baseline: {prod_baseline}")
    print(f"Gate decision: {'APPROVED' if decision.approved else 'REJECTED'}")
    for reason in decision.reasons:
        print(f"  - {reason}")

    if not decision.approved:
        _write_github_output("approved", "false")
        return 1

    image_tag = get_image_tag(candidate_version)
    if not image_tag:
        raise SystemExit(
            f"Model version {args.model_name} v{args.version} has no image_tag set. "
            "The dev build/push workflow must tag it (pdm.evaluation.registry.set_image_tag) "
            "before it can be promoted."
        )

    promote_version(client, args.model_name, args.version, candidate_scores, holdout_version)
    set_image_tag(client, args.model_name, args.version, image_tag)  # reaffirm, idempotent

    print(f"Promoted {args.model_name} v{args.version} to Production (image_tag={image_tag})")
    _write_github_output("approved", "true")
    _write_github_output("image_tag", image_tag)
    _write_github_output("promoted_version", args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
