"""What actually happened to each engine: the ground truth that turns logged predictions
into training labels (pdm.labels.build).

Two event types:
- failure: the engine failed at `cycle`. Every earlier reading of that life gets an
  exact label: RUL = failure cycle - reading cycle.
- maintenance: the engine was overhauled/replaced at `cycle` BEFORE failing. Its true
  RUL is unknown ("censored"): we only know it would have lasted longer than that.

In practice these come from the maintenance system (CMMS) as a periodic export, so the
main entry point is a CSV import; `add` is for one-off corrections.

Usage:
    python -m pdm.labels.outcomes import --csv cmms_export.csv
    python -m pdm.labels.outcomes add --asset-id engine-17 --event failure --cycle 212
    python -m pdm.labels.outcomes list
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

EVENT_TYPES = ("failure", "maintenance")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcome_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('failure', 'maintenance')),
    cycle INTEGER NOT NULL CHECK (cycle >= 0),
    event_ts REAL,
    recorded_at REAL NOT NULL,
    source TEXT,
    note TEXT,
    UNIQUE (asset_id, event_type, cycle)
);
"""


class OutcomeStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def add(
        self,
        asset_id: str,
        event_type: str,
        cycle: int,
        event_ts: float | None = None,
        source: str | None = None,
        note: str | None = None,
    ) -> bool:
        """Records one event. Returns False if the identical event already exists
        (re-importing the same export is safe)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}, got {event_type!r}")
        if not asset_id:
            raise ValueError("asset_id is required")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO outcome_events "
                "(asset_id, event_type, cycle, event_ts, recorded_at, source, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (asset_id, event_type, int(cycle), event_ts, time.time(), source, note),
            )
            return cur.rowcount == 1

    def import_csv(self, path: str | Path, source: str | None = None) -> dict:
        """CSV columns: asset_id, event_type, cycle, [event_ts], [note]. Bad rows are
        counted and skipped, never half-imported."""
        added = duplicate = bad = 0
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ts = row.get("event_ts")
                    ok = self.add(
                        row["asset_id"].strip(),
                        row["event_type"].strip().lower(),
                        int(row["cycle"]),
                        float(ts) if ts else None,
                        source=source or Path(path).name,
                        note=row.get("note"),
                    )
                except (KeyError, ValueError):
                    bad += 1
                    continue
                added += ok
                duplicate += not ok
        return {"added": added, "duplicate": duplicate, "bad": bad}

    def events(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT asset_id, event_type, cycle, event_ts, recorded_at, source, note "
                "FROM outcome_events ORDER BY asset_id, cycle"
            ).fetchall()
        return [dict(r) for r in rows]


def main() -> int:
    from pdm.common.config import get_settings

    parser = argparse.ArgumentParser(description="Record engine outcomes (failures, maintenance).")
    parser.add_argument("--db", default=get_settings().outcome_db)
    sub = parser.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--csv", required=True)
    add = sub.add_parser("add")
    add.add_argument("--asset-id", required=True)
    add.add_argument("--event", required=True, choices=EVENT_TYPES)
    add.add_argument("--cycle", required=True, type=int)
    add.add_argument("--note")
    sub.add_parser("list")
    args = parser.parse_args()

    store = OutcomeStore(args.db)
    if args.cmd == "import":
        print(store.import_csv(args.csv))
    elif args.cmd == "add":
        print(
            "added"
            if store.add(args.asset_id, args.event, args.cycle, note=args.note)
            else "duplicate"
        )
    else:
        for e in store.events():
            print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
