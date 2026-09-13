"""Unit tests for scripts/mlflow_gcs_sync.py's locking and generation-match logic,
against an in-memory fake GCS bucket (no real network/credentials needed) that
faithfully implements the precondition semantics we depend on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlflow_gcs_sync as sync  # noqa: E402


class FakeBlob:
    def __init__(self, store: dict, name: str):
        self._store = store
        self.name = name
        self.generation = 0

    def exists(self) -> bool:
        return self.name in self._store

    def reload(self) -> None:
        if self.name in self._store:
            self.generation = self._store[self.name][1]

    def download_as_text(self) -> str:
        return self._store[self.name][0].decode()

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self._store[self.name][0])

    def upload_from_string(self, data: str, if_generation_match=None) -> None:
        self._upload(data.encode(), if_generation_match)

    def upload_from_filename(self, path: str, if_generation_match=None) -> None:
        self._upload(Path(path).read_bytes(), if_generation_match)

    def _upload(self, data: bytes, if_generation_match) -> None:
        current = self._store.get(self.name, (b"", 0))[1]
        if if_generation_match is not None and if_generation_match != current:
            raise PreconditionFailed("generation mismatch")
        new_gen = current + 1
        self._store[self.name] = (data, new_gen)
        self.generation = new_gen

    def delete(self) -> None:
        del self._store[self.name]


class FakeBucket:
    def __init__(self, name: str = "fake-bucket"):
        self.name = name
        self._store: dict = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, name)


def test_pull_first_ever_run_creates_zero_generation_sidecar(tmp_path):
    bucket = FakeBucket()
    db_path = tmp_path / "mlflow.db"
    sync.pull(bucket, "mlflow", db_path, max_age_seconds=1800)

    sidecar = sync._generation_sidecar(db_path)
    assert sidecar.read_text().strip() == "0"
    assert not db_path.exists()  # nothing to download yet


def test_push_first_ever_run_succeeds_and_releases_lock(tmp_path):
    bucket = FakeBucket()
    db_path = tmp_path / "mlflow.db"
    db_path.write_bytes(b"fake sqlite content")
    sync._generation_sidecar(db_path).write_text("0")

    sync.push(bucket, "mlflow", db_path)

    assert bucket._store["mlflow/mlflow.db"][0] == b"fake sqlite content"
    assert "mlflow/mlflow.db.lock" not in bucket._store  # lock released


def test_pull_then_push_round_trip_updates_remote(tmp_path):
    bucket = FakeBucket()
    bucket._store["mlflow/mlflow.db"] = (b"v1 content", 1)

    db_path = tmp_path / "mlflow.db"
    sync.pull(bucket, "mlflow", db_path, max_age_seconds=1800)
    assert db_path.read_bytes() == b"v1 content"

    db_path.write_bytes(b"v2 content (modified locally)")
    sync.push(bucket, "mlflow", db_path)

    assert bucket._store["mlflow/mlflow.db"][0] == b"v2 content (modified locally)"


def test_push_fails_loudly_on_concurrent_write_instead_of_overwriting(tmp_path):
    bucket = FakeBucket()
    bucket._store["mlflow/mlflow.db"] = (b"original", 1)

    db_path = tmp_path / "mlflow.db"
    sync.pull(bucket, "mlflow", db_path, max_age_seconds=1800)

    # Simulate another job pushing a newer version in between our pull and our push.
    bucket._store["mlflow/mlflow.db"] = (b"someone else's newer write", 2)
    del bucket._store["mlflow/mlflow.db.lock"]  # that other job released its own lock

    db_path.write_bytes(b"our stale local change")
    with pytest.raises(SystemExit):
        sync.push(bucket, "mlflow", db_path)

    # The concurrent write must survive untouched - not silently overwritten.
    assert bucket._store["mlflow/mlflow.db"][0] == b"someone else's newer write"
    # Lock is still released even though the push failed.
    assert "mlflow/mlflow.db.lock" not in bucket._store


def test_acquire_lock_fails_loudly_when_fresh_lock_held(tmp_path):
    bucket = FakeBucket()
    sync.acquire_lock(bucket, "mlflow", max_age_seconds=1800)

    with pytest.raises(SystemExit, match="locked by another job"):
        sync.acquire_lock(bucket, "mlflow", max_age_seconds=1800)


def test_acquire_lock_takes_over_a_stale_lock(tmp_path, capsys):
    bucket = FakeBucket()
    stale_payload = '{"host": "old-runner", "acquired_at": %f, "acquired_at_iso": "old"}' % (
        time.time() - 10000
    )
    bucket._store["mlflow/mlflow.db.lock"] = (stale_payload.encode(), 1)

    sync.acquire_lock(bucket, "mlflow", max_age_seconds=60)

    captured = capsys.readouterr()
    assert "STALE lock" in captured.err
    assert "mlflow/mlflow.db.lock" in bucket._store


def test_release_lock_missing_does_not_raise(tmp_path, capsys):
    bucket = FakeBucket()
    sync.release_lock(bucket, "mlflow")  # no lock present at all
    captured = capsys.readouterr()
    assert "Failed to release" in captured.err
