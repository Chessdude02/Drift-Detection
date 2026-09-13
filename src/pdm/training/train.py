"""Training entrypoint: load a dataset, engineer features, fit a model, log+register to
MLflow. Dataset-agnostic — `config["dataset"]["type"]` (default "cmapss") selects the
loader/feature/split adapter from pdm.data.datasets; see config/training.yaml (C-MAPSS)
and config/training_bearing.yaml (IMS Bearing) for the two supported configs.

Usage:
    python -m pdm.training.train --raw-dir data/raw --config-name training.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow

from pdm.common.config import configure_mlflow_env, load_yaml
from pdm.common.logging import setup_logging
from pdm.data.datasets import get_adapter
from pdm.training.evaluate import passes_validation_gate, rmse

logger = logging.getLogger(__name__)


def _fit_model(algorithm: str, params: dict, X_train, y_train):
    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**params)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    model.fit(X_train, y_train)
    return model


def run_training(
    raw_dir: Path, config: dict, register: bool = True, extra_tags: dict[str, str] | None = None
) -> dict:
    """Runs one full train+eval+log cycle. Returns a dict with rmse, run_id, model_version
    (the registered model version number, or None if register=False), and gate result.
    `extra_tags` (e.g. CI provenance: git_sha, git_ref, ci_run_id) are set on the run.
    """
    dataset_cfg = config["dataset"]
    features_cfg = config["features"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]
    eval_cfg = config["evaluation"]

    dataset_type = dataset_cfg.get("type", "cmapss")
    adapter = get_adapter(dataset_type)

    feature_df, cols = adapter["load"](raw_dir, dataset_cfg, features_cfg)
    random_state = model_cfg["params"].get("random_state", 42)
    train_idx, val_idx = adapter["split"](feature_df, model_cfg["val_split"], random_state)
    train_df, val_df = feature_df.iloc[train_idx], feature_df.iloc[val_idx]

    X_train, y_train = train_df[cols], train_df["rul"]
    X_val, y_val = val_df[cols], val_df["rul"]

    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run() as run:
        if extra_tags:
            mlflow.set_tags(extra_tags)
        mlflow.log_params({"algorithm": model_cfg["algorithm"], **model_cfg["params"]})
        mlflow.log_param("dataset_type", dataset_type)
        if "subset" in dataset_cfg:
            mlflow.log_param("subset", dataset_cfg["subset"])
        if dataset_cfg.get("rul_cap") is not None:
            mlflow.log_param("rul_cap", dataset_cfg["rul_cap"])

        model = _fit_model(model_cfg["algorithm"], model_cfg["params"], X_train, y_train)
        val_preds = model.predict(X_val)
        val_rmse = rmse(y_val, val_preds)
        mlflow.log_metric("val_rmse", val_rmse)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        flavor = mlflow.lightgbm if model_cfg["algorithm"] == "lightgbm" else mlflow.xgboost
        model_info = flavor.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=mlflow_cfg["registered_model_name"] if register else None,
        )
        model_version = getattr(model_info, "registered_model_version", None)

        gate_passed = passes_validation_gate(val_rmse, eval_cfg["max_rmse"])
        mlflow.log_metric("gate_passed", int(gate_passed))

        logger.info(
            "run_id=%s model_version=%s val_rmse=%.3f max_rmse=%.3f gate_passed=%s",
            run.info.run_id,
            model_version,
            val_rmse,
            eval_cfg["max_rmse"],
            gate_passed,
        )
        return {
            "run_id": run.info.run_id,
            "model_version": model_version,
            "val_rmse": val_rmse,
            "max_rmse": eval_cfg["max_rmse"],
            "gate_passed": gate_passed,
        }


def main() -> int:
    setup_logging()
    configure_mlflow_env()
    parser = argparse.ArgumentParser(description="Train an RUL model and log it to MLflow.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config-name", default="training.yaml")
    parser.add_argument(
        "--no-register", action="store_true", help="Log the run without registering a model version"
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit non-zero if the validation gate (RMSE threshold) fails (used in CI)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set an MLflow run tag (repeatable), e.g. --tag git_sha=abc123 --tag ci_run_id=456",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write {run_id, model_version, val_rmse, gate_passed} as JSON",
    )
    args = parser.parse_args()

    extra_tags = dict(tag.split("=", 1) for tag in args.tag)

    config = load_yaml(args.config_name)
    result = run_training(
        Path(args.raw_dir), config, register=not args.no_register, extra_tags=extra_tags
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.fail_on_gate and not result["gate_passed"]:
        logger.error(
            "Model failed validation gate: val_rmse=%.3f > max_rmse=%.3f",
            result["val_rmse"],
            result["max_rmse"],
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
