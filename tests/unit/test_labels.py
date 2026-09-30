"""Outcome store and the rules that turn outcomes into labels (pdm.labels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.labels.build import (
    LABEL_UNIT_OFFSET,
    build_labels,
    load_label_dataset,
    write_label_dataset,
)
from pdm.labels.outcomes import OutcomeStore

pytestmark = pytest.mark.unit

COLS = ["f1", "f2"]
CAP = 125


def _rows(asset, cycles, shadow=False):
    # newest first, like InferenceLog.read_for_assets
    return [
        {"asset_id": asset, "cycle": c, "features": {"f1": c, "f2": -c}, "shadow": shadow}
        for c in sorted(cycles, reverse=True)
    ]


def _event(asset, kind, cycle):
    return {"asset_id": asset, "event_type": kind, "cycle": cycle}


def test_failure_gives_exact_capped_labels():
    df, stats = build_labels(_rows("a", [10, 100, 190]), [_event("a", "failure", 200)], COLS, CAP)
    by_cycle = df.set_index("time_in_cycles")
    assert by_cycle.loc[190, "rul"] == 10 and by_cycle.loc[190, "true_rul"] == 10
    assert by_cycle.loc[10, "rul"] == CAP and by_cycle.loc[10, "true_rul"] == 190
    assert set(df["label_kind"]) == {"failure"}
    assert stats["rows_failure"] == 3


def test_maintenance_keeps_only_readings_whose_capped_label_is_known():
    df, stats = build_labels(
        _rows("b", [10, 60, 100]), [_event("b", "maintenance", 150)], COLS, CAP
    )
    # 150-10=140 >= 125 -> exact label 125; 150-60=90 and 150-100=50 -> unknown, dropped
    assert df["time_in_cycles"].tolist() == [10]
    assert df["rul"].tolist() == [CAP] and np.isnan(df["true_rul"].iloc[0])
    assert stats["dropped_censored_below_cap"] == 2


def test_readings_split_into_lives_by_events():
    events = [_event("c", "maintenance", 300), _event("c", "failure", 500)]
    df, _ = build_labels(_rows("c", [100, 400]), events, COLS, CAP)
    # 100 belongs to the life ending in maintenance at 300 (200 >= cap: label 125);
    # 400 to the life ending in failure at 500 (label 100). Different synthetic units.
    by_cycle = df.set_index("time_in_cycles")
    assert by_cycle.loc[100, "label_kind"] == "censored_at_cap"
    assert by_cycle.loc[400, "rul"] == 100
    assert df["unit_number"].nunique() == 2
    assert (df["unit_number"] >= LABEL_UNIT_OFFSET).all()


def test_rows_that_cannot_be_labelled_are_counted():
    rows = (
        _rows("d", [5])  # no outcome yet
        + _rows("e", [5], shadow=True)
        + [{"asset_id": None, "cycle": 5, "features": {"f1": 1, "f2": 1}}]
        + [{"asset_id": "e", "cycle": None, "features": {"f1": 1, "f2": 1}}]
        + [{"asset_id": "e", "cycle": 6, "features": {"f1": 1}}]  # missing f2
        + _rows("e", [7, 7])  # logged twice (client retry)
    )
    df, stats = build_labels(rows, [_event("e", "failure", 50)], COLS, CAP)
    assert stats["dropped_no_outcome_yet"] == 1
    assert stats["dropped_shadow"] == 1
    assert stats["dropped_no_asset_or_cycle"] == 2
    assert stats["dropped_missing_features"] == 1
    assert stats["dropped_duplicate"] == 1
    assert df["time_in_cycles"].tolist() == [7]


def test_outcome_store_import_is_idempotent_and_skips_bad_rows(tmp_path):
    export = tmp_path / "cmms.csv"
    export.write_text(
        "asset_id,event_type,cycle\n"
        "e1,failure,200\n"
        "e2,Maintenance,150\n"
        "e3,exploded,10\n"  # unknown type
        "e4,failure,not-a-number\n"
    )
    store = OutcomeStore(tmp_path / "outcomes.db")
    assert store.import_csv(export) == {"added": 2, "duplicate": 0, "bad": 2}
    assert store.import_csv(export) == {"added": 0, "duplicate": 2, "bad": 2}
    assert [e["event_type"] for e in store.events()] == ["failure", "maintenance"]


def test_label_dataset_round_trip_and_feature_check(tmp_path):
    df, stats = build_labels(_rows("a", [190]), [_event("a", "failure", 200)], COLS, CAP)
    manifest = write_label_dataset(df, stats, tmp_path, {"test": True})
    loaded, loaded_manifest = load_label_dataset(tmp_path, COLS)
    assert len(loaded) == 1 and loaded_manifest["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match="feature columns"):
        load_label_dataset(tmp_path, COLS + ["f3"])
    empty, _ = load_label_dataset(tmp_path / "nothing", COLS)
    assert isinstance(empty, pd.DataFrame) and empty.empty
