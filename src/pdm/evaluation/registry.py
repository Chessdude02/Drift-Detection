"""MLflow model registry helpers for staged promotion + rollback.

Uses model VERSION TAGS (not a separate datastore) to carry state across CI runs:

- `baseline_f2_score` / `baseline_precision` / `baseline_pr_auc` / `baseline_holdout_version`
  / `baseline_recorded_at`: the scores a version had *when it became Production* — read
  back as the comparison baseline for the next candidate, so we never need to re-run
  prod's scoring on every promotion decision.
- `rollback_production`: "true" on exactly one version at a time — whichever version
  was Production immediately before the current one. This is the rollback target; it
  moves forward by one each time `promote_version` or `rollback_to_previous` runs, so
  the pointer always trails Production by exactly one step in either direction
  (matching the requested rollback workflow — never a full history walk).
- `image_tag`: the Docker image tag (git SHA) built alongside this model version, so a
  promotion/rollback knows which container image to point the K8s deployment at.
"""

from __future__ import annotations

import datetime as dt
import logging

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

ROLLBACK_PRODUCTION_TAG = "rollback_production"
IMAGE_TAG_KEY = "image_tag"
BASELINE_KEYS = ("baseline_f2_score", "baseline_precision", "baseline_pr_auc")


def get_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    versions = client.get_latest_versions(model_name, stages=["Production"])
    return versions[0] if versions else None


def get_rollback_target(client: MlflowClient, model_name: str) -> ModelVersion | None:
    """The one version currently tagged `rollback_production=true`, if any."""
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.tags.get(ROLLBACK_PRODUCTION_TAG) == "true":
            return v
    return None


def get_baseline_scores(version: ModelVersion) -> dict[str, float] | None:
    """Reads the frozen baseline_* tags off a (typically Production) model version.
    Returns None if any are missing (e.g. a version that predates this tagging scheme).
    """
    if not all(key in version.tags for key in BASELINE_KEYS):
        return None
    return {
        "f2_score": float(version.tags["baseline_f2_score"]),
        "precision": float(version.tags["baseline_precision"]),
        "pr_auc": float(version.tags["baseline_pr_auc"]),
    }


def get_image_tag(version: ModelVersion) -> str | None:
    return version.tags.get(IMAGE_TAG_KEY)


def set_image_tag(client: MlflowClient, model_name: str, version: str, image_tag: str) -> None:
    client.set_model_version_tag(model_name, version, IMAGE_TAG_KEY, image_tag)


def _move_rollback_tag(
    client: MlflowClient, model_name: str, from_version: ModelVersion | None, to_version: str
) -> None:
    """Clears rollback_production off whichever version currently holds it, and sets it
    on `to_version` instead — used identically by promote_version (moving it to the
    version being replaced) and rollback_to_previous (moving it to the version being
    replaced by the rollback), so the tag always trails Production by exactly one step.
    """
    stale = get_rollback_target(client, model_name)
    if stale is not None:
        client.delete_model_version_tag(model_name, stale.version, ROLLBACK_PRODUCTION_TAG)
    if from_version is not None and from_version.version != to_version:
        client.set_model_version_tag(
            model_name, from_version.version, ROLLBACK_PRODUCTION_TAG, "true"
        )


def promote_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    scores: dict[str, float],
    holdout_version: str,
) -> None:
    """Promotes `version` to Production, freezing its baseline_* tags for future gate
    checks and shifting the `rollback_production` marker to whichever version this
    replaces (the old Production version, if any).
    """
    old_prod = get_production_version(client, model_name)
    _move_rollback_tag(client, model_name, from_version=old_prod, to_version=version)

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True
    )

    client.set_model_version_tag(model_name, version, "baseline_f2_score", str(scores["f2_score"]))
    client.set_model_version_tag(
        model_name, version, "baseline_precision", str(scores["precision"])
    )
    client.set_model_version_tag(model_name, version, "baseline_pr_auc", str(scores["pr_auc"]))
    client.set_model_version_tag(model_name, version, "baseline_holdout_version", holdout_version)
    client.set_model_version_tag(
        model_name, version, "baseline_recorded_at", dt.datetime.now(dt.timezone.utc).isoformat()
    )
    # A version that's now Production can't simultaneously be the rollback target.
    client.delete_model_version_tag(model_name, version, ROLLBACK_PRODUCTION_TAG)
    logger.info("Promoted %s v%s to Production", model_name, version)


def rollback_to_previous(client: MlflowClient, model_name: str) -> ModelVersion:
    """Reverts Production to the version tagged `rollback_production=true`, and moves
    that tag onto the version being replaced by the rollback (so a subsequent
    roll-forward is symmetric). Raises ValueError if there is no current Production
    version or no rollback target.

    This only ever goes back one step. If the resulting Production version is ALSO
    found to be bad, that is not automated — see RUNBOOK.md.
    """
    current_prod = get_production_version(client, model_name)
    if current_prod is None:
        raise ValueError(f"No current Production version of {model_name!r} to roll back from")

    target = get_rollback_target(client, model_name)
    if target is None:
        raise ValueError(
            f"No version of {model_name!r} tagged {ROLLBACK_PRODUCTION_TAG!r}; "
            "cannot determine a rollback target"
        )

    client.transition_model_version_stage(
        name=model_name, version=target.version, stage="Production", archive_existing_versions=True
    )
    _move_rollback_tag(client, model_name, from_version=current_prod, to_version=target.version)

    logger.info(
        "Rolled back %s Production from v%s to v%s",
        model_name,
        current_prod.version,
        target.version,
    )
    return target
