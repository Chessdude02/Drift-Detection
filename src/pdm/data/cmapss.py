"""NASA C-MAPSS turbofan degradation dataset: schema + loaders.

Raw files (train_FD00x.txt / test_FD00x.txt / RUL_FD00x.txt) are whitespace-delimited,
no header, 26 columns: unit_number, time_in_cycles, 3 operational settings, 21 sensors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
OP_SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS


def _read_whitespace_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some distributions have two trailing all-NaN columns from double spaces.
    df = df.iloc[:, : len(ALL_COLUMNS)]
    df.columns = ALL_COLUMNS
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)
    return df


def load_train(raw_dir: Path | str, subset: str = "FD001") -> pd.DataFrame:
    """Load a training subset and compute per-row RUL (cycles until that unit's last cycle)."""
    path = Path(raw_dir) / f"train_{subset}.txt"
    df = _read_whitespace_file(path)
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["rul"] = max_cycle - df["time_in_cycles"]
    return df


def load_test(raw_dir: Path | str, subset: str = "FD001") -> tuple[pd.DataFrame, pd.Series]:
    """Load a test subset plus the true final-cycle RUL for each unit (RUL_FD00x.txt)."""
    test_path = Path(raw_dir) / f"test_{subset}.txt"
    rul_path = Path(raw_dir) / f"RUL_{subset}.txt"
    test_df = _read_whitespace_file(test_path)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")[0]
    true_rul.index = range(1, len(true_rul) + 1)
    true_rul.index.name = "unit_number"
    return test_df, true_rul
