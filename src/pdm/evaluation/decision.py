"""Scores a model by the maintenance decisions it drives, not by RMSE.

The rule being scored: for each engine, walk its cycles in order and schedule
maintenance the first time the model's signal (predicted RUL, or its interval's lower
bound) drops to `threshold`. Maintenance takes `lead_time` cycles to happen (parts,
crew, a slot in the hangar). Each engine then ends one of three ways:

- planned: maintenance happened before failure. Cost = planned-maintenance cost plus
  the useful life thrown away (true RUL at maintenance time).
- late: the rule fired, but with less than `lead_time` cycles left. It fails first.
- missed: the rule never fired. It fails.

Late and missed both count as unplanned failures. This needs run-to-failure histories
(the true RUL at every cycle), so it is scored on held-out FD001 training engines, never
on NASA's truncated test set.

The costs in config/decision.yaml are PLACEHOLDERS showing the shape of the trade-off
(a failure costs ~10x a planned visit). Replace them with real numbers before trusting
any threshold this picks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_policy(
    trajectories: pd.DataFrame,
    signal: np.ndarray,
    threshold: float,
    lead_time: int,
    costs: dict,
) -> dict:
    """`trajectories` needs unit_number, time_in_cycles and true_rul (uncapped), one row
    per cycle, run to failure; `signal` is aligned with its rows."""
    df = trajectories[["unit_number", "time_in_cycles", "true_rul"]].copy()
    df["signal"] = np.asarray(signal, dtype=float)
    df = df.sort_values(["unit_number", "time_in_cycles"])

    planned = late = missed = 0
    wasted = []
    for _, g in df.groupby("unit_number", sort=False):
        fired = g.index[g["signal"].to_numpy() <= threshold]
        if len(fired) == 0:
            missed += 1
            continue
        rul_at_trigger = float(g.loc[fired[0], "true_rul"])
        if rul_at_trigger < lead_time:
            late += 1
        else:
            planned += 1
            wasted.append(rul_at_trigger - lead_time)

    n = planned + late + missed
    failures = late + missed
    wasted_total = float(np.sum(wasted)) if wasted else 0.0
    total_cost = (
        costs["unplanned_failure"] * failures
        + costs["planned_maintenance"] * planned
        + costs["per_wasted_cycle"] * wasted_total
    )
    return {
        "threshold": float(threshold),
        "n_engines": n,
        "unplanned_failures": failures,
        "late": late,
        "missed": missed,
        "planned": planned,
        "mean_wasted_cycles": wasted_total / planned if planned else 0.0,
        "cost_per_engine": total_cost / n if n else 0.0,
    }


def choose_threshold(
    trajectories: pd.DataFrame,
    signal: np.ndarray,
    thresholds,
    lead_time: int,
    costs: dict,
) -> dict:
    """Threshold with the lowest cost per engine; ties go to fewer failures, then to
    less wasted life."""
    results = [simulate_policy(trajectories, signal, t, lead_time, costs) for t in thresholds]
    return min(
        results,
        key=lambda r: (
            round(r["cost_per_engine"], 9),
            r["unplanned_failures"],
            r["mean_wasted_cycles"],
        ),
    )


def threshold_grid(cfg: dict) -> np.ndarray:
    g = cfg["threshold_grid"]
    return np.arange(g["start"], g["stop"] + 1e-9, g["step"])
