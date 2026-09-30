"""One-time: pick the frozen C-MAPSS holdout engines and write them to
data/holdout/cmapss_<subset>_holdout_<version>.json.

These engines are excluded from every training run (config/training.yaml
dataset.holdout_units_file) and used only to judge models: the maintenance-decision
cost needs run-to-failure histories, which NASA's truncated test set does not have.
Once written the file is frozen - every champion/challenger comparison uses the same
engines, so scores stay comparable over time. Make a new version rather than editing it.

Usage:
    python scripts/build_cmapss_holdout.py --subset FD001 --n-units 20 --version v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from pdm.data.cmapss import load_train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--n-units", type=int, default=20)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    out = Path("data/holdout") / f"cmapss_{args.subset}_holdout_{args.version}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists and is frozen; pick a new --version")

    units = sorted(load_train(Path(args.raw_dir), args.subset)["unit_number"].unique())
    chosen = sorted(
        int(u) for u in np.random.default_rng(args.seed).choice(units, args.n_units, replace=False)
    )
    out.write_text(
        json.dumps(
            {
                "subset": args.subset,
                "version": args.version,
                "units": chosen,
                "seed": args.seed,
                "n_units_in_subset": len(units),
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "note": "Never train on these engines. Used by pdm.evaluation.champion_challenger.",
            },
            indent=2,
        )
    )
    print(f"Wrote {out}: {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
