"""Rolls Production back to the version tagged `rollback_production=true` (one step
back — never a full history walk; see RUNBOOK.md for what to do if that target is also
bad). Used by both the manual and the auto-triggered (failed health check) rollback
workflow.

Usage:
    python scripts/rollback_production.py --model-name ims_bearing_rul
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, "src")
from pdm.common.config import configure_mlflow_env  # noqa: E402
from pdm.evaluation.registry import get_image_tag, rollback_to_previous  # noqa: E402


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def main() -> int:
    configure_mlflow_env()
    parser = argparse.ArgumentParser(description="Roll Production back one step.")
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    try:
        restored = rollback_to_previous(client, args.model_name)
    except ValueError as e:
        print(f"Rollback failed: {e}")
        print(
            "See RUNBOOK.md: if this happens, the fix is a manual redeployment of a "
            "specific known-good MLflow model version - not automated."
        )
        return 1

    image_tag = get_image_tag(restored)
    if not image_tag:
        print(
            f"Rolled back to {args.model_name} v{restored.version}, but it has no "
            "image_tag set - cannot determine which image to redeploy. Fix this "
            "version's image_tag tag manually before updating the K8s deployment."
        )
        return 1

    print(f"Production for {args.model_name} is now v{restored.version} (image_tag={image_tag})")
    _write_github_output("restored_version", restored.version)
    _write_github_output("image_tag", image_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
