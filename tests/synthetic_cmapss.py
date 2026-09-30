"""Small synthetic C-MAPSS-format fleets for end-to-end tests: engines whose sensors drift
with wear toward failure, written in NASA's train/test/RUL file layout.

Not a model of real turbofans - just enough structure (a shared wear signal driving
correlated sensors, engine-to-engine offsets, noise) for every stage of the pipeline to
run for real: training, intervals, drift and sensor checks, labels, gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_SENSORS = 21


def make_fleet(
    n_engines: int,
    seed: int,
    min_life: int = 60,
    max_life: int = 120,
    wear_direction: np.ndarray | None = None,
    first_unit: int = 1,
    level_shift: float = 0.0,
) -> pd.DataFrame:
    """Run-to-failure histories, 26 columns like train_FD00x.txt. `wear_direction`
    changes how sensors respond to wear (a different 'fault mode'); `level_shift` moves
    every sensor (e.g. different operating conditions)."""
    rng = np.random.default_rng(seed)
    direction = (
        wear_direction
        if wear_direction is not None
        else np.random.default_rng(0).normal(size=N_SENSORS)
    )
    rows = []
    for i in range(n_engines):
        unit = first_unit + i
        life = int(rng.integers(min_life, max_life))
        offset = rng.normal(scale=0.3, size=N_SENSORS)
        for cycle in range(1, life + 1):
            wear = (cycle / life) ** 2
            sensors = (
                10
                + level_shift
                + offset
                + direction * 3 * wear
                + rng.normal(scale=0.2, size=N_SENSORS)
            )
            rows.append([unit, cycle, 0.0, 0.0, 100.0, *sensors])
    return pd.DataFrame(rows)


def write_raw_dir(raw_dir: Path, train: pd.DataFrame, test_engines: pd.DataFrame, seed: int = 1):
    """train_FD001 = `train`; test_FD001 = `test_engines` cut at a random point, with the
    cycles left written to RUL_FD001, like NASA's test set."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(raw_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    rng = np.random.default_rng(seed)
    kept, rul = [], []
    for _, g in test_engines.groupby(0):
        cut = int(rng.integers(max(len(g) // 3, 1), len(g)))
        kept.append(g.iloc[:cut])
        rul.append(len(g) - cut)
    pd.concat(kept).to_csv(raw_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(rul).to_csv(raw_dir / "RUL_FD001.txt", sep=" ", header=False, index=False)
