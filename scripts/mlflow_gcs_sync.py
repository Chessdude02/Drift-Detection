"""Syncs mlflow.db to/from GCS around any CI step that touches MLflow.

Every job that reads/writes the shared MLflow registry does:
    python scripts/mlflow_gcs_sync.py pull --bucket $BUCKET --db-path mlflow.db
    MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db  <run the actual mlflow-touching step>
    python scripts/mlflow_gcs_sync.py push --bucket $BUCKET --db-path mlflow.db

Two guards against a concurrent job silently clobbering another's write:
  1. A lock object (mlflow.db.lock) that `pull` must acquire before downloading -  a
     second job trying to pull while the lock is held fails loudly instead of racing.
     A lock older than --lock-max-age-seconds is assumed abandoned (a crashed job that
     never reached `push`) and is taken over, loudly, rather than deadlocking forever.
  2. A GCS generation-match precondition on `push`: the generation mlflow.db had at
     pull time is recorded in a sidecar file, and push only succeeds if that's still
     the current generation. If another job pushed in between, this fails loudly and
     exits non-zero rather than overwriting their data.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

DB_BLOB_NAME = "mlflow.db"
LOCK_BLOB_NAME = "mlflow.db.lock"
GENERATION_SIDECAR_SUFFIX = ".generation"
DEFAULT_LOCK_MAX_AGE_SECONDS = 1800  # 30 min: long enough for a slow training run,
# short enough that a crashed job doesn't deadlock every future run permanently.


def _lock_blob(bucket, prefix: str):
    return bucket.blob(f"{prefix}/{LOCK_BLOB_NAME}")


def _db_blob(bucket, prefix: str):
    return bucket.blob(f"{prefix}/{DB_BLOB_NAME}")


def _generation_sidecar(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + GENERATION_SIDECAR_SUFFIX)


def _lock_payload() -> str:
    now = time.time()
    return json.dumps(
        {
            "host": socket.gethostname(),
            "acquired_at": now,
            "acquired_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
    )


def acquire_lock(bucket, prefix: str, max_age_seconds: int) -> None:
    """Creates the lock blob. Raises SystemExit if a fresh lock is already held; takes
    over (loudly) a lock older than max_age_seconds, assuming its holder crashed.
    """
    blob = _lock_blob(bucket, prefix)
    expected_generation = 0  # GCS precondition sentinel: "object must not exist"

    if blob.exists():
        blob.reload()
        expected_generation = blob.generation
        try:
            payload = json.loads(blob.download_as_text())
            age = time.time() - payload["acquired_at"]
            holder, acquired_at_iso = payload["host"], payload["acquired_at_iso"]
        except Exception:
            age, holder, acquired_at_iso = float("inf"), "unknown", "unknown"

        if age < max_age_seconds:
            raise SystemExit(
                f"mlflow.db is locked by another job (host={holder}, acquired "
                f"{acquired_at_iso}, {age:.0f}s ago). Refusing to proceed and risk a "
                f"lost concurrent write. If no other job is actually running, wait for "
                f"the lock to go stale (>{max_age_seconds}s) or delete "
                f"gs://{bucket.name}/{prefix}/{LOCK_BLOB_NAME} manually."
            )
        print(
            f"::warning::Found a STALE lock (host={holder}, acquired {acquired_at_iso}, "
            f"{age:.0f}s ago, past the {max_age_seconds}s staleness threshold) - "
            "assuming that job crashed without releasing it, taking over.",
            file=sys.stderr,
        )

    try:
        blob.upload_from_string(_lock_payload(), if_generation_match=expected_generation)
    except PreconditionFailed:
        raise SystemExit(
            "Lost a race to acquire the mlflow.db lock (another job grabbed it at the "
            "same instant). Re-run this job."
        )
    print(f"Acquired mlflow.db lock (gs://{bucket.name}/{prefix}/{LOCK_BLOB_NAME})")


def release_lock(bucket, prefix: str) -> None:
    try:
        _lock_blob(bucket, prefix).delete()
        print(f"Released mlflow.db lock (gs://{bucket.name}/{prefix}/{LOCK_BLOB_NAME})")
    except Exception as e:
        print(
            f"::warning::Failed to release mlflow.db lock: {e} "
            "(it will be reclaimed once stale)",
            file=sys.stderr,
        )


def pull(bucket, prefix: str, db_path: Path, max_age_seconds: int) -> None:
    acquire_lock(bucket, prefix, max_age_seconds)
    blob = _db_blob(bucket, prefix)
    sidecar = _generation_sidecar(db_path)

    if blob.exists():
        blob.reload()
        blob.download_to_filename(str(db_path))
        sidecar.write_text(str(blob.generation))
        print(
            f"Pulled mlflow.db from gs://{bucket.name}/{prefix}/{DB_BLOB_NAME} "
            f"(generation={blob.generation})"
        )
    else:
        sidecar.write_text("0")
        print(
            f"No existing mlflow.db at gs://{bucket.name}/{prefix}/{DB_BLOB_NAME} - "
            "starting fresh (first-ever run against this store)"
        )


def push(bucket, prefix: str, db_path: Path) -> None:
    sidecar = _generation_sidecar(db_path)
    expected_generation = int(sidecar.read_text().strip()) if sidecar.exists() else 0
    blob = _db_blob(bucket, prefix)

    push_failed = False
    try:
        blob.upload_from_filename(str(db_path), if_generation_match=expected_generation)
        print(f"Pushed mlflow.db to gs://{bucket.name}/{prefix}/{DB_BLOB_NAME}")
    except PreconditionFailed:
        push_failed = True
        print(
            "::error::Another job pushed a newer mlflow.db to GCS between our pull and "
            f"this push (expected generation {expected_generation}, no longer current). "
            "Refusing to overwrite their data. This job's MLflow writes are LOST from "
            "the shared store - re-run this job (it will pull the newer state first).",
            file=sys.stderr,
        )
    finally:
        release_lock(bucket, prefix)

    if push_failed:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync mlflow.db to/from GCS around a CI step.")
    sub = parser.add_subparsers(dest="command", required=True)

    pull_p = sub.add_parser("pull", help="Acquire the lock and download mlflow.db")
    pull_p.add_argument("--bucket", required=True)
    pull_p.add_argument("--prefix", default="mlflow")
    pull_p.add_argument("--db-path", default="mlflow.db")
    pull_p.add_argument("--lock-max-age-seconds", type=int, default=DEFAULT_LOCK_MAX_AGE_SECONDS)

    push_p = sub.add_parser(
        "push", help="Upload mlflow.db (if unchanged upstream) and release the lock"
    )
    push_p.add_argument("--bucket", required=True)
    push_p.add_argument("--prefix", default="mlflow")
    push_p.add_argument("--db-path", default="mlflow.db")

    args = parser.parse_args()
    bucket = storage.Client().bucket(args.bucket)

    if args.command == "pull":
        pull(bucket, args.prefix, Path(args.db_path), args.lock_max_age_seconds)
    else:
        push(bucket, args.prefix, Path(args.db_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
