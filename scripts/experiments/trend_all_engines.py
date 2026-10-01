"""Does the trend-feature model make better or worse maintenance decisions?
(docs/decisions.md D-27 left this open: better RMSE, but ~1 more failure on the 20-engine
frozen holdout, too few engines to tell luck from a real effect.)

Uses every engine instead of 20. For each dataset (FD001, FD003), all 100 run-to-failure
engines are split into 5 groups of 20; each group is held out once while the baseline
(config/training.yaml) and the trend model (config/training_trend.yaml) are trained on
the other 80 exactly as training does (pdm.training.train.fit_bundle: cross-validated
interval + maintenance threshold chosen on training engines only). The held-out engines
are then run through the maintenance rule (pdm.evaluation.decision.simulate_policy) one
by one, so every engine gets an outcome from models that never saw it. Repeated with
REPEATS different group splits.

Because both models face the same engines, the comparison is paired per engine:
trend minus baseline cost and failures, with a bootstrap 95% interval over engines.

Writes reports/trend_all_engines.json. ~30-45 minutes.

Usage (from the repo root):
    python scripts/experiments/trend_all_engines.py --raw-dir data/raw
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from pdm.common.config import load_yaml
from pdm.data.datasets import _cmapss_split, get_adapter
from pdm.evaluation.decision import simulate_policy
from pdm.training.train import fit_bundle

DATASETS = ["FD001", "FD003"]
REPEATS = 2
FOLDS = 5
BOOTSTRAP = 2000


def all_engines(raw_dir: Path, cfg: dict, subset: str):
    """Every engine of `subset` (no frozen-holdout exclusion: here each engine is held
    out by the K-fold loop instead)."""
    dataset_cfg = {**cfg["dataset"], "subset": subset, "holdout_units_file": None}
    return get_adapter("cmapss")["load"](raw_dir, dataset_cfg, cfg["features"])


def per_engine_outcomes(bundle, rows: pd.DataFrame, cols, decision_cfg) -> list[dict]:
    frame = bundle.predict_frame(rows[cols])
    signal = frame["rul_lower"].where(frame["rul_lower"].notna(), frame["rul"]).to_numpy()
    out = []
    for unit, idx in rows.groupby("unit_number").indices.items():
        engine = rows.iloc[idx]
        p = simulate_policy(
            engine,
            signal[idx],
            bundle.maintenance_threshold,
            decision_cfg["lead_time_cycles"],
            decision_cfg["costs"],
        )
        err = frame["rul"].to_numpy()[idx] - engine["rul"].to_numpy()
        near = engine["true_rul"].to_numpy() < 60
        inside = (engine["rul"].to_numpy() >= frame["rul_lower"].to_numpy()[idx]) & (
            engine["rul"].to_numpy() <= frame["rul_upper"].to_numpy()[idx]
        )
        out.append(
            {
                "unit": int(unit),
                "cost": p["cost_per_engine"],
                "failure": int(p["unplanned_failures"]),
                "missed": int(p["missed"]),
                "late": int(p["late"]),
                "wasted": p["mean_wasted_cycles"] if p["planned"] else None,
                "sq_err_sum": float(np.sum(err**2)),
                "sq_err_near_sum": float(np.sum(err[near] ** 2)),
                "n_rows": int(len(idx)),
                "n_near": int(near.sum()),
                "inside": int(inside.sum()),
                "threshold": float(bundle.maintenance_threshold),
            }
        )
    return out


def bootstrap_ci(diffs: np.ndarray, rng) -> tuple[float, float]:
    means = rng.choice(diffs, size=(BOOTSTRAP, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(records: pd.DataFrame) -> dict:
    s = {}
    for variant, g in records.groupby("variant"):
        s[variant] = {
            "engine_evaluations": int(len(g)),
            "cost_per_engine": float(g["cost"].mean()),
            "failure_rate": float(g["failure"].mean()),
            "missed_rate": float(g["missed"].mean()),
            "late_rate": float(g["late"].mean()),
            "mean_wasted_cycles": float(g["wasted"].dropna().mean()),
            "rmse_all_cycles": float(np.sqrt(g["sq_err_sum"].sum() / g["n_rows"].sum())),
            "rmse_last_60_cycles": float(np.sqrt(g["sq_err_near_sum"].sum() / g["n_near"].sum())),
            "interval_coverage": float(g["inside"].sum() / g["n_rows"].sum()),
            "thresholds_chosen": sorted(set(g["threshold"])),
        }
    # Paired: same engine, same repeat, trend minus baseline.
    wide = records.pivot_table(
        index=["repeat", "unit"], columns="variant", values=["cost", "failure"]
    )
    rng = np.random.default_rng(0)
    paired = {}
    for metric in ("cost", "failure"):
        # Average the repeats per engine first, so the interval is over engines.
        d = (wide[(metric, "trend")] - wide[(metric, "baseline")]).groupby(level="unit").mean()
        lo, hi = bootstrap_ci(d.to_numpy(), rng)
        paired[f"{metric}_diff_trend_minus_baseline"] = {
            "mean": float(d.mean()),
            "ci95": [lo, hi],
            "engines_worse": int((d > 0).sum()),
            "engines_better": int((d < 0).sum()),
            "engines_same": int((d == 0).sum()),
        }
    s["paired"] = paired
    return s


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="reports/trend_all_engines.json")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    configs = {"baseline": load_yaml("training.yaml"), "trend": load_yaml("training_trend.yaml")}
    for cfg in configs.values():
        cfg["labels"] = {"enabled": False}
        cfg["model"]["params"]["verbose"] = -1
    decision_cfg = load_yaml(configs["baseline"]["decision_config"])

    report = {"repeats": REPEATS, "folds": FOLDS, "datasets": {}}
    for subset in args.datasets:
        data = {name: all_engines(raw_dir, cfg, subset) for name, cfg in configs.items()}
        units = data["baseline"][0]["unit_number"].unique()
        records = []
        start = time.time()
        for repeat in range(REPEATS):
            order = np.random.default_rng(1000 + repeat).permutation(units)
            groups = np.array_split(order, FOLDS)
            for fold, held in enumerate(groups):
                for variant, cfg in configs.items():
                    df, cols = data[variant]
                    train = df[~df["unit_number"].isin(held)].reset_index(drop=True)
                    test = df[df["unit_number"].isin(held)].reset_index(drop=True)
                    fold_cfg = copy.deepcopy(cfg)
                    fold_cfg["model"]["params"]["random_state"] = 42 + repeat
                    bundle = fit_bundle(train, cols, fold_cfg, _cmapss_split)["bundle"]
                    for rec in per_engine_outcomes(bundle, test, cols, decision_cfg):
                        records.append({**rec, "variant": variant, "repeat": repeat, "fold": fold})
                print(
                    f"{subset} repeat {repeat} fold {fold} done ({time.time() - start:.0f}s)",
                    flush=True,
                )
        frame = pd.DataFrame(records)
        summary = summarise(frame)
        report["datasets"][subset] = {"summary": summary, "records": records}
        print(json.dumps({subset: summary}, indent=2), flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
