from __future__ import annotations

import pytest
from mlflow import MlflowClient

from pdm.evaluation.registry import (
    get_baseline_scores,
    get_image_tag,
    get_production_version,
    get_rollback_target,
    mark_not_operationally_ready,
    promote_version,
    rollback_to_previous,
    set_image_tag,
)

pytestmark = pytest.mark.unit

MODEL_NAME = "test_ims_bearing_rul_registry"


@pytest.fixture
def client(mlflow_tracking_uri):
    c = MlflowClient()
    c.create_registered_model(MODEL_NAME)
    return c


def _new_version(client: MlflowClient) -> str:
    return client.create_model_version(MODEL_NAME, source="file:///fake/model").version


def test_get_production_version_none_when_nothing_promoted(client):
    v1 = _new_version(client)
    assert get_production_version(client, MODEL_NAME) is None
    assert get_rollback_target(client, MODEL_NAME) is None
    assert v1  # created but never promoted


def test_first_promotion_sets_baseline_tags_and_no_previous(client):
    v1 = _new_version(client)
    scores = {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}
    promote_version(client, MODEL_NAME, v1, scores, holdout_version="v1")

    prod = get_production_version(client, MODEL_NAME)
    assert prod is not None
    assert prod.version == v1
    assert prod.current_stage == "Production"
    assert get_baseline_scores(prod) == scores
    assert get_rollback_target(client, MODEL_NAME) is None


def test_second_promotion_archives_first_and_tags_it_previous(client):
    v1 = _new_version(client)
    promote_version(
        client, MODEL_NAME, v1, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}, "v1"
    )

    v2 = _new_version(client)
    scores_v2 = {"f2_score": 0.5, "precision": 0.55, "pr_auc": 0.65}
    promote_version(client, MODEL_NAME, v2, scores_v2, "v1")

    prod = get_production_version(client, MODEL_NAME)
    assert prod.version == v2
    assert get_baseline_scores(prod) == scores_v2

    previous = get_rollback_target(client, MODEL_NAME)
    assert previous.version == v1
    assert previous.current_stage == "Archived"
    # v1's own original baseline is untouched by v2's promotion.
    assert get_baseline_scores(previous) == {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}


def test_rollback_reinstates_previous_version_and_its_original_baseline(client):
    v1 = _new_version(client)
    scores_v1 = {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}
    promote_version(client, MODEL_NAME, v1, scores_v1, "v1")

    v2 = _new_version(client)
    promote_version(
        client, MODEL_NAME, v2, {"f2_score": 0.5, "precision": 0.55, "pr_auc": 0.65}, "v1"
    )

    rolled_back_to = rollback_to_previous(client, MODEL_NAME)
    assert rolled_back_to.version == v1

    prod = get_production_version(client, MODEL_NAME)
    assert prod.version == v1
    assert prod.current_stage == "Production"
    assert get_baseline_scores(prod) == scores_v1

    # v2 is now the rollback target if we needed to roll forward again.
    previous = get_rollback_target(client, MODEL_NAME)
    assert previous.version == v2
    assert previous.current_stage == "Archived"


def test_rollback_with_no_previous_version_raises(client):
    v1 = _new_version(client)
    promote_version(
        client, MODEL_NAME, v1, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}, "v1"
    )

    with pytest.raises(ValueError, match="rollback_production"):
        rollback_to_previous(client, MODEL_NAME)


def test_rollback_with_no_production_version_raises(client):
    with pytest.raises(ValueError, match="No current Production version"):
        rollback_to_previous(client, MODEL_NAME)


def test_image_tag_round_trip(client):
    v1 = _new_version(client)
    set_image_tag(client, MODEL_NAME, v1, "sha-abc123")
    refreshed = client.get_model_version(MODEL_NAME, v1)
    assert get_image_tag(refreshed) == "sha-abc123"


def test_get_baseline_scores_none_when_tags_missing(client):
    v1 = _new_version(client)
    version = client.get_model_version(MODEL_NAME, v1)
    assert get_baseline_scores(version) is None


def test_mark_not_operationally_ready_sets_tags_on_the_version(client):
    v1 = _new_version(client)
    promote_version(
        client, MODEL_NAME, v1, {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}, "v1"
    )

    mark_not_operationally_ready(
        client, MODEL_NAME, v1, "val_rmse (61.1) is 2.0x failure_horizon (30); precision=0.52"
    )

    refreshed = client.get_model_version(MODEL_NAME, v1)
    assert refreshed.tags["operational_readiness"] == "not_ready"
    assert "2.0x failure_horizon" in refreshed.tags["operational_readiness_note"]
    # Marking readiness doesn't disturb stage or the baseline tags set at promotion.
    assert refreshed.current_stage == "Production"
    assert get_baseline_scores(refreshed) == {"f2_score": 0.4, "precision": 0.5, "pr_auc": 0.6}
