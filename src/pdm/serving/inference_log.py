"""Append-only SQLite log of inference inputs/outputs, consumed later by the drift-check job
and by the label builder (pdm.labels) that turns outcomes into new training rows."""

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

# Columns added after the first release. Existing databases are migrated in place on
# open (ALTER TABLE ADD COLUMN), so old rows simply have NULLs.
_ADDED_COLUMNS = {
    "asset_id": "TEXT",
    "cycle": "INTEGER",
    "observed_at": "REAL",
    "input_warnings": "TEXT",
    "model_version": "TEXT",
}


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
            existing = {row[1] for row in conn.execute("PRAGMA table_info(inference_log)")}
            for name, sql_type in _ADDED_COLUMNS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE inference_log ADD COLUMN {name} {sql_type}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_inference_log_asset ON inference_log(asset_id)"
            )

    def record(
        self,
        features: dict[str, float],
        prediction: float,
        shadow: bool = False,
        asset_id: str | None = None,
        cycle: int | None = None,
        observed_at: float | None = None,
        input_warnings: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO inference_log (ts, features, prediction, shadow, asset_id, cycle, "
                "observed_at, input_warnings, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now,
                    json.dumps(features),
                    float(prediction),
                    int(shadow),
                    asset_id,
                    cycle,
                    observed_at if observed_at is not None else now,
                    json.dumps(input_warnings or []),
                    model_version,
                ),
            )

    def _rows(self, where: str, params: tuple, limit: int | None) -> list[dict]:
        sql = (
            "SELECT ts, features, prediction, shadow, asset_id, cycle, observed_at, "
            f"input_warnings, model_version FROM inference_log {where} ORDER BY id DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "ts": r["ts"],
                "features": json.loads(r["features"]),
                "prediction": r["prediction"],
                "shadow": bool(r["shadow"]),
                "asset_id": r["asset_id"],
                "cycle": r["cycle"],
                "observed_at": r["observed_at"],
                "input_warnings": json.loads(r["input_warnings"]) if r["input_warnings"] else [],
                "model_version": r["model_version"],
            }
            for r in rows
        ]

    def read_recent(self, limit: int = 500) -> list[dict]:
        return self._rows("", (), limit)

    def read_since(self, since_ts: float, limit: int | None = None) -> list[dict]:
        """Rows received at or after `since_ts` (unix seconds), newest first."""
        return self._rows("WHERE ts >= ?", (since_ts,), limit)

    def read_for_assets(self, asset_ids: list[str]) -> list[dict]:
        """Every non-shadow row for the given engines, newest first (label building)."""
        if not asset_ids:
            return []
        marks = ",".join("?" * len(asset_ids))
        return self._rows(
            f"WHERE shadow = 0 AND asset_id IN ({marks})", tuple(asset_ids), limit=None
        )
