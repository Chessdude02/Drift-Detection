"""Life-stage matching for the drift check.

Engine sensors move as engines wear, so a fleet that is younger or older on average than
the reference looks like "drift" to a plain distribution test even when nothing is wrong
(measured on real FD001: 66-100% false retrain triggers for young/old fleets). Here the
reference is resampled so its predicted-RUL mix matches the current window's before the
two are compared, so only changes *at the same life stage* count as drift.

Both sides use predicted RUL (the reference stores out-of-fold predictions, the current
window stores what the served model predicted), so matching never needs true RUL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REFERENCE_PREDICTION_COLUMN = "predicted_rul"


def add_oof_predictions(
    feature_df: pd.DataFrame, feature_cols: list[str], model_cfg: dict, n_splits: int = 5
) -> pd.DataFrame:
    """Adds `predicted_rul`: out-of-fold predictions, grouped by engine, so each reference
    row gets a prediction from a model that never saw that engine - the same situation as
    a live prediction. In-sample predictions would be unrealistically close to the truth.
    """
    from pdm.training.train import _fit_model

    out = feature_df.copy()
    n_units = out["unit_number"].nunique()
    n_splits = max(2, min(n_splits, n_units))
    preds = np.zeros(len(out))
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(out, groups=out["unit_number"]):
        train = out.iloc[train_idx]
        model = _fit_model(
            model_cfg["algorithm"], model_cfg["params"], train[feature_cols], train["rul"]
        )
        preds[val_idx] = model.predict(out.iloc[val_idx][feature_cols])
    out[REFERENCE_PREDICTION_COLUMN] = preds
    return out


def match_life_stage(
    reference: pd.DataFrame,
    current_predictions,
    reference_column: str = REFERENCE_PREDICTION_COLUMN,
    bin_width: float = 10.0,
    max_rul: float = 130.0,
    sample_rows: int = 5000,
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Resamples `reference` (with replacement) so its predicted-RUL histogram matches
    `current_predictions`. Returns (resampled reference, coverage), where coverage is the
    share of current rows whose RUL bin exists in the reference; bins the reference lacks
    are dropped from the target mix. The fixed seed keeps a given check reproducible.
    """
    edges = np.arange(bin_width, max_rul, bin_width)
    ref_bins = np.digitize(np.asarray(reference[reference_column], dtype=float), edges)
    cur_bins = np.digitize(np.asarray(current_predictions, dtype=float), edges)

    ref_share = pd.Series(ref_bins).value_counts(normalize=True)
    cur_share = pd.Series(cur_bins).value_counts(normalize=True)
    covered = cur_share[cur_share.index.isin(ref_share.index)]
    coverage = float(covered.sum())
    if coverage == 0.0:
        return reference, 0.0
    target = covered / coverage

    weights = pd.Series(ref_bins).map(lambda b: target.get(b, 0.0) / ref_share[b]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(reference), size=sample_rows, replace=True, p=weights / weights.sum())
    return reference.iloc[idx], coverage
