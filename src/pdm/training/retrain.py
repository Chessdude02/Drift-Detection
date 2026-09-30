"""Retrain entrypoint used by the pdm-retrain CronJob and by drift-triggered Jobs:
train a candidate, then let the champion/challenger gate decide whether it replaces
Production. A candidate that fails the gate stays registered (for inspection) but is
never promoted; Production is untouched.

Exit code: 0 if the candidate was promoted or correctly rejected by the gate, 1 if
training itself failed. A rejection is a normal outcome, not a job failure - the gate's
reasons are in the log and in --json-out.

Usage:
    python -m pdm.training.retrain --raw-dir /data/raw --config-name training.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pdm.common.config import configure_mlflow_env, load_yaml
from pdm.common.logging import setup_logging
from pdm.evaluation.champion_challenger import run_gate
from pdm.training.train import run_training

logger = logging.getLogger(__name__)


def retrain(raw_dir: Path, config: dict, promote: bool = True, extra_tags=None) -> dict:
    trained = run_training(raw_dir, config, register=True, extra_tags=extra_tags)
    gate = run_gate(trained["model_version"], raw_dir, config, promote=promote)
    return {"training": trained, "gate": gate}


def main() -> int:
    setup_logging()
    configure_mlflow_env()
    parser = argparse.ArgumentParser(description="Train a candidate and gate it.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config-name", default="training.yaml")
    parser.add_argument("--no-promote", action="store_true", help="Score only, never promote")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    result = retrain(Path(args.raw_dir), load_yaml(args.config_name), promote=not args.no_promote)
    gate = result["gate"]
    logger.info(
        "Candidate v%s %s",
        gate["candidate_version"],
        "promoted to Production" if gate["promoted"] else "NOT promoted (see gate reasons)",
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
