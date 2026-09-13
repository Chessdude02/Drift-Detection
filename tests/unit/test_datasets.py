from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.data.datasets import get_adapter

pytestmark = pytest.mark.unit


def _synthetic_bearing_df(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"rul": np.arange(n - 1, -1, -1), "x": np.arange(n)})


def test_bearing_split_train_set_includes_low_rul_examples():
    """The whole point of the fix: training must see near-failure (low RUL) examples,
    not just a temporally-truncated high-RUL range."""
    df = _synthetic_bearing_df(1000)
    split = get_adapter("ims_bearing")["split"]
    train_idx, val_idx = split(df, val_split=0.2, random_state=42)

    train_rul = df.iloc[train_idx]["rul"]
    assert train_rul.min() < 10, (
        f"training set's minimum RUL was {train_rul.min()} - it should include "
        "near-failure examples, not just a high-RUL range"
    )


def test_bearing_split_train_and_val_partition_all_rows_exactly_once():
    df = _synthetic_bearing_df(500)
    split = get_adapter("ims_bearing")["split"]
    train_idx, val_idx = split(df, val_split=0.2, random_state=42)

    assert len(train_idx) + len(val_idx) == len(df)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert set(train_idx) | set(val_idx) == set(range(len(df)))


def test_bearing_split_respects_val_split_fraction():
    df = _synthetic_bearing_df(1000)
    split = get_adapter("ims_bearing")["split"]
    train_idx, val_idx = split(df, val_split=0.2, random_state=42)

    assert len(val_idx) == 200
    assert len(train_idx) == 800


def test_bearing_split_is_reproducible_given_same_random_state():
    df = _synthetic_bearing_df(500)
    split = get_adapter("ims_bearing")["split"]
    train_idx_1, val_idx_1 = split(df, val_split=0.2, random_state=7)
    train_idx_2, val_idx_2 = split(df, val_split=0.2, random_state=7)

    np.testing.assert_array_equal(train_idx_1, train_idx_2)
    np.testing.assert_array_equal(val_idx_1, val_idx_2)


def test_bearing_split_differs_across_random_states():
    df = _synthetic_bearing_df(500)
    split = get_adapter("ims_bearing")["split"]
    _, val_idx_a = split(df, val_split=0.2, random_state=1)
    _, val_idx_b = split(df, val_split=0.2, random_state=2)

    assert set(val_idx_a) != set(val_idx_b)
