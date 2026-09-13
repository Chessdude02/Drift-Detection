"""Promote an MLflow model version from Staging to Production (or any stage transition).

Usage:
    python scripts/promote_model.py --name cmapss_rul --version 3 --stage Production
"""

from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, "src")
from pdm.common.config import configure_mlflow_env  # noqa: E402


def main() -> int:
    configure_mlflow_env()
    parser = argparse.ArgumentParser(description="Promote an MLflow registered model version.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--stage", default="Production")
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        default=True,
        help="Archive existing versions currently in the target stage (default: on)",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    client.transition_model_version_stage(
        name=args.name,
        version=args.version,
        stage=args.stage,
        archive_existing_versions=args.archive_existing,
    )
    print(f"Promoted {args.name} v{args.version} -> {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
