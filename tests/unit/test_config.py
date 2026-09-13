from __future__ import annotations

import mlflow
import pytest
from mlflow import MlflowClient

from pdm.common.config import configure_mlflow_env, get_settings, set_experiment_with_artifact_root

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # get_settings() is a process-wide lru_cache; without clearing it before AND after
    # each test here, a Settings object built from this test's monkeypatched env (often
    # pointing at a tmp_path that gets deleted at teardown) would leak into whichever
    # test runs next in this session.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_configure_mlflow_env_raises_on_present_but_empty_uri(monkeypatch):
    # This is the exact failure signature of an unset GitHub Actions secret referenced
    # via ${{ secrets.X }}: the env var is set, but to an empty string.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="set but empty"):
        configure_mlflow_env()


def test_configure_mlflow_env_leaves_a_real_uri_untouched(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///somewhere.db")
    get_settings.cache_clear()
    configure_mlflow_env()
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == "sqlite:///somewhere.db"


def test_configure_mlflow_env_falls_back_to_default_when_fully_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()
    configure_mlflow_env()
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == "sqlite:///mlflow.db"


def test_set_experiment_with_artifact_root_uses_configured_remote_root(
    mlflow_tracking_uri, monkeypatch
):
    # A real remote URI (the actual production case: gs://...) already has a scheme
    # and must pass through unchanged.
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "gs://fake-bucket/mlartifacts")
    get_settings.cache_clear()

    set_experiment_with_artifact_root("my_new_experiment")

    client = MlflowClient()
    exp = client.get_experiment_by_name("my_new_experiment")
    assert exp.artifact_location == "gs://fake-bucket/mlartifacts/my_new_experiment"


def test_set_experiment_with_artifact_root_normalizes_bare_local_path(
    mlflow_tracking_uri, monkeypatch, tmp_path
):
    # Regression test: a bare local path (e.g. Windows `C:/Users/.../mlruns`, no
    # scheme) must be converted to a proper file:// URI before use - passing it raw
    # fails because e.g. "C:" parses as a URI scheme, not a drive letter, and MLflow's
    # artifact-repository registry doesn't recognize that scheme.
    bare_path = tmp_path / "artifacts-stand-in"
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", bare_path.as_posix())
    get_settings.cache_clear()

    set_experiment_with_artifact_root("my_new_experiment")

    client = MlflowClient()
    exp = client.get_experiment_by_name("my_new_experiment")
    expected = bare_path.resolve().as_uri() + "/my_new_experiment"
    assert exp.artifact_location == expected

    # And logging a real run's artifact must actually work end-to-end (this is exactly
    # what failed before the fix - not just the string value being "close enough").
    with mlflow.start_run():
        mlflow.log_metric("x", 1.0)


def test_set_experiment_with_artifact_root_leaves_existing_experiment_alone(
    mlflow_tracking_uri, monkeypatch, tmp_path
):
    # First creation with no artifact root configured -> local default.
    get_settings.cache_clear()
    set_experiment_with_artifact_root("stable_experiment")
    client = MlflowClient()
    original_location = client.get_experiment_by_name("stable_experiment").artifact_location

    # A later call with a DIFFERENT configured root must not move an existing experiment.
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", (tmp_path / "different-root").as_posix())
    get_settings.cache_clear()
    set_experiment_with_artifact_root("stable_experiment")

    assert client.get_experiment_by_name("stable_experiment").artifact_location == original_location


def test_set_experiment_with_artifact_root_none_uses_mlflow_default(
    mlflow_tracking_uri, monkeypatch
):
    monkeypatch.delenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", raising=False)
    get_settings.cache_clear()

    set_experiment_with_artifact_root("plain_experiment")

    client = MlflowClient()
    exp = client.get_experiment_by_name("plain_experiment")
    assert exp is not None
    assert exp.artifact_location.startswith("file://") or exp.artifact_location.startswith(
        "mlflow-artifacts:"
    )
