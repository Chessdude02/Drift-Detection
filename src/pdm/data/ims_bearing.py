"""NASA IMS Bearing run-to-failure vibration dataset: raw snapshot file I/O.

Each experiment is a directory of whitespace-delimited snapshot files, one per sampling
interval (every ~10 minutes in the real dataset), named as a zero-padded timestamp
string 'YYYY.MM.DD.HH.MM.SS' — lexicographic filename order is chronological order.
Each file holds one row per vibration sample and one column per accelerometer channel
(one channel per bearing in the 2nd/3rd IMS test sets used here).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_SNAPSHOT_NAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def list_snapshot_files(raw_dir: Path | str) -> list[Path]:
    """Snapshot files in chronological order (filenames are zero-padded timestamps).

    Only files whose name matches the real dataset's 'YYYY.MM.DD.HH.MM.SS' timestamp
    format are included — anything else (`.gitkeep`, `.DS_Store`, a stray readme dropped
    into the same folder, etc.) is silently skipped rather than passed to `load_snapshot`
    and blown up on as if it were vibration data.
    """
    raw_dir = Path(raw_dir)
    files = [p for p in raw_dir.iterdir() if p.is_file() and _SNAPSHOT_NAME_RE.match(p.name)]
    if not files:
        raise ValueError(f"No snapshot files found in {raw_dir}")
    return sorted(files, key=lambda p: p.name)


def load_snapshot(path: Path, n_channels: int) -> np.ndarray:
    """Load one snapshot file into an (n_samples, n_channels) array."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, n_channels)
    return data
