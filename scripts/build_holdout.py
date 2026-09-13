"""Builds the fixed, versioned holdout set (config/evaluation.yaml's holdout.path) from
a directory of IMS Bearing snapshot files, and logs it as an MLflow artifact for
provenance. This is a deliberate, manual, one-time-per-version action — never run
automatically by CI or the training pipeline. Re-running it with the SAME --out path
overwrites the frozen benchmark, silently invalidating every prior baseline comparison;
bump --version/--out (e.g. holdout_v2.csv) instead if the holdout genuinely needs to change.

Usage:
    python scripts/build_holdout.py --raw-dir data/raw_bearing_holdout \
        --out data/holdout/holdout_v1.csv --version v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, "src")
from pdm.common.config import configure_mlflow_env, load_yaml  # noqa: E402
from pdm.evaluation.holdout import build_holdout_dataframe  # noqa: E402


def main() -> int:
    configure_mlflow_env()
    eval_cfg = load_yaml("evaluation.yaml")

    parser = argparse.ArgumentParser(
        description="Build the fixed holdout set for staging/prod gating."
    )
    parser.add_argument(
        "--raw-dir", required=True, help="Snapshot files reserved for holdout use only"
    )
    parser.add_argument(
        "--channels", nargs="+", default=["bearing1", "bearing2", "bearing3", "bearing4"]
    )
    parser.add_argument("--failure-channel", default="bearing4")
    parser.add_argument("--out", default=eval_cfg["holdout"]["path"])
    parser.add_argument("--version", default=eval_cfg["holdout"]["version"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists (default: refuse, to protect a frozen holdout)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(
            f"Refusing to overwrite existing holdout file {out_path} without --force. "
            "A holdout set is meant to be frozen once built; use a new --version/--out "
            "if you deliberately need a new one.",
            file=sys.stderr,
        )
        return 1

    df = build_holdout_dataframe(
        args.raw_dir, channels=args.channels, failure_channel=args.failure_channel
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} holdout rows to {out_path}")

    mlflow.set_experiment(eval_cfg["holdout"]["mlflow_experiment"])
    with mlflow.start_run(run_name=f"build_holdout_{args.version}") as run:
        mlflow.log_param("holdout_version", args.version)
        mlflow.log_param("raw_dir", str(args.raw_dir))
        mlflow.log_param("channels", args.channels)
        mlflow.log_param("failure_channel", args.failure_channel)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_artifact(str(out_path), artifact_path="holdout")
        mlflow.set_tag("holdout_version", args.version)
        print(
            f"Logged holdout artifact to MLflow run {run.info.run_id} (experiment "
            f"{eval_cfg['holdout']['mlflow_experiment']!r})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
