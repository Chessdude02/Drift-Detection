"""Scores a candidate model against the fixed holdout set and logs F2/precision/PR-AUC
to MLflow as a new run tagged with the candidate's identity — this is what the
dev->staging workflow calls after deploying a candidate to staging.

Usage:
    python scripts/score_on_holdout.py --model-uri runs:/<run_id>/model --candidate-run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys

import mlflow

sys.path.insert(0, "src")
from pdm.common.config import configure_mlflow_env, load_yaml  # noqa: E402
from pdm.data.bearing_features import feature_columns  # noqa: E402
from pdm.evaluation.scoring import compute_holdout_scores  # noqa: E402


def main() -> int:
    configure_mlflow_env()
    eval_cfg = load_yaml("evaluation.yaml")
    training_cfg = load_yaml("training_bearing.yaml")

    parser = argparse.ArgumentParser(
        description="Score a candidate model on the fixed holdout set."
    )
    parser.add_argument(
        "--model-uri", required=True, help="e.g. runs:/<run_id>/model or models:/name/3"
    )
    parser.add_argument(
        "--candidate-run-id", required=True, help="MLflow run id of the candidate's training run"
    )
    parser.add_argument("--holdout-path", default=eval_cfg["holdout"]["path"])
    parser.add_argument("--holdout-version", default=eval_cfg["holdout"]["version"])
    parser.add_argument(
        "--failure-horizon", type=int, default=eval_cfg["failure"]["horizon_snapshots"]
    )
    parser.add_argument(
        "--json-out", help="Optional path to also write the scores as JSON (for a CI step to read)"
    )
    args = parser.parse_args()

    import pandas as pd

    holdout_df = pd.read_csv(args.holdout_path)
    model = mlflow.pyfunc.load_model(args.model_uri)
    cols = feature_columns(training_cfg["features"]["channels"])

    scores = compute_holdout_scores(model, holdout_df, cols, args.failure_horizon)

    # A named experiment (not the implicit "Default") so
    # scripts/promote_to_production.py can reliably search for this candidate's latest
    # staging evaluation by tag.
    mlflow.set_experiment(eval_cfg["holdout"]["staging_eval_mlflow_experiment"])
    with mlflow.start_run(run_name=f"staging_eval_{args.candidate_run_id[:8]}") as run:
        mlflow.set_tag("evaluation_stage", "staging")
        mlflow.set_tag("candidate_run_id", args.candidate_run_id)
        mlflow.set_tag("candidate_model_uri", args.model_uri)
        mlflow.set_tag("holdout_version", args.holdout_version)
        mlflow.log_param("failure_horizon_snapshots", args.failure_horizon)
        mlflow.log_metric("f2_score", scores["f2_score"])
        mlflow.log_metric("precision", scores["precision"])
        mlflow.log_metric("pr_auc", scores["pr_auc"])
        mlflow.log_metric("n_holdout", scores["n_holdout"])
        mlflow.log_metric("n_positive", scores["n_positive"])
        print(f"Logged staging evaluation to MLflow run {run.info.run_id}")

    print(json.dumps(scores, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({**scores, "staging_eval_run_id": run.info.run_id}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
