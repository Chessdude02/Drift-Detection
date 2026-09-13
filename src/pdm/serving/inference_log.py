"""Append-only SQLite log of inference inputs/outputs, consumed later by the drift-check job."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    features TEXT NOT NULL,
    prediction REAL NOT NULL,
    shadow INTEGER NOT NULL DEFAULT 0
);
"""


class InferenceLog:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(_SCHEMA)

    def record(self, features: dict[str, float], prediction: float, shadow: bool = False) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO inference_log (ts, features, prediction, shadow) VALUES (?, ?, ?, ?)",
                (time.time(), json.dumps(features), float(prediction), int(shadow)),
            )

    def read_recent(self, limit: int = 500) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, features, prediction, shadow FROM inference_log "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": r["ts"],
                "features": json.loads(r["features"]),
                "prediction": r["prediction"],
                "shadow": bool(r["shadow"]),
            }
            for r in rows
        ]
