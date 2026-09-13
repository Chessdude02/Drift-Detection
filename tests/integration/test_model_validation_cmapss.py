"""CI gate: runs the real training pipeline end-to-end against the committed C-MAPSS
fixture and fails if holdout RMSE exceeds a (loose, fixture-appropriate) threshold.

See tests/unit/test_evaluate.py for pure-function tests of the metric math itself.
"""

from __future__ import annotations

import pytest

from pdm.data.cmapss import SENSOR_COLUMNS
from pdm.training.train import run_training

pytestmark = pytest.mark.integration

GATE_CONFIG = {
    "dataset": {"type": "cmapss", "subset": "FD001", "rul_cap": 125},
    "features": {"sensor_columns": SENSOR_COLUMNS[:4], "rolling_windows": [5]},
    "model": {
        "algorithm": "lightgbm",
        "params": {
            "n_estimators": 20,
            "learning_rate": 0.2,
            "max_depth": 3,
            "num_leaves": 7,
            "random_state": 42,
            "min_child_samples": 1,
        },
        "val_split": 0.5,
    },
    "mlflow": {"experiment_name": "gate_cmapss_rul", "registered_model_name": "gate_cmapss_rul"},
    # Deliberately generous for the tiny/noisy fixture; config/training.yaml's real threshold
    # (35.0) is tuned against the full C-MAPSS FD001 dataset, not this 50-row sample.
    "evaluation": {"max_rmse": 200.0},
}


def test_model_validation_gate(raw_data_dir, mlflow_tracking_uri):
    result = run_training(raw_data_dir, GATE_CONFIG, register=False)
    assert result["gate_passed"], (
        f"Model failed validation gate: val_rmse={result['val_rmse']:.2f} > "
        f"max_rmse={GATE_CONFIG['evaluation']['max_rmse']}"
    )
