from __future__ import annotations

import sqlite3

import pytest

from pdm.serving.inference_log import InferenceLog

pytestmark = pytest.mark.unit


def test_old_database_is_migrated_in_place(tmp_path):
    db = tmp_path / "log.db"
    with sqlite3.connect(db) as conn:  # the original schema, with one old row
        conn.execute(
            "CREATE TABLE inference_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
            "features TEXT NOT NULL, prediction REAL NOT NULL, shadow INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO inference_log (ts, features, prediction) VALUES (1.0, '{\"a\": 1}', 5.0)"
        )
    log = InferenceLog(db)
    log.record({"a": 2.0}, 6.0, asset_id="e1", cycle=3)
    rows = log.read_recent(10)
    assert rows[0]["asset_id"] == "e1" and rows[0]["cycle"] == 3
    assert rows[1]["asset_id"] is None and rows[1]["prediction"] == 5.0


def test_read_since_and_read_for_assets(tmp_path):
    log = InferenceLog(tmp_path / "log.db")
    log.record({"a": 1.0}, 1.0, asset_id="e1")
    log.record({"a": 1.0}, 2.0, asset_id="e2")
    log.record({"a": 1.0}, 3.0, asset_id="e1", shadow=True)
    assert len(log.read_since(0)) == 3
    assert log.read_since(10**12) == []
    rows = log.read_for_assets(["e1"])
    assert [r["prediction"] for r in rows] == [1.0]  # shadow row excluded
