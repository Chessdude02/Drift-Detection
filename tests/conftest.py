from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES_DIR / "cmapss_sample.txt"
BEARING_SAMPLE_DIR = FIXTURES_DIR / "ims_bearing_sample"

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    """Build a data/raw-shaped directory (train/test/RUL files) from the tiny fixture."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    full = pd.read_csv(SAMPLE_FILE, sep=r"\s+", header=None, engine="python")

    # train file: the fixture as-is.
    full.to_csv(raw_dir / "train_FD001.txt", sep=" ", header=False, index=False)

    # test file: truncate each unit's last 5 cycles off, RUL file records the truth for those.
    test_rows = []
    true_rul = []
    for unit, grp in full.groupby(0):
        grp = grp.sort_values(1)
        cut = max(len(grp) - 5, 1)
        kept = grp.iloc[:cut]
        removed = len(grp) - cut
        test_rows.append(kept)
        true_rul.append(removed)
    test_df = pd.concat(test_rows)
    test_df.to_csv(raw_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(true_rul).to_csv(raw_dir / "RUL_FD001.txt", sep=" ", header=False, index=False)

    return raw_dir


@pytest.fixture
def bearing_raw_dir() -> Path:
    """A ~2%-scale, real-format sample of the NASA IMS Bearing run-to-failure dataset
    (24 snapshot files, 300 samples/file vs. the real ~984 files x 20480 samples) — see
    tests/fixtures/ims_bearing_sample and its generation notes in the fixture itself.
    Read-only and small enough to use directly; no tmp_path copy needed.
    """
    return BEARING_SAMPLE_DIR


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", (tmp_path / "mlruns").as_posix())

    import mlflow

    # mlflow's fluent API caches the "active experiment id" as process-global state.
    # Without resetting it here, a bare `mlflow.start_run()` in a later test can pick up
    # an experiment id left over from an EARLIER test's (different) tracking store and
    # fail with "No Experiment with id=N exists" against this test's fresh store.
    # end_run() clears the active-run stack; set_experiment re-resolves "Default"
    # against the new URI, which always exists in a fresh store.
    mlflow.end_run()
    mlflow.set_experiment("Default")
    return uri
