"""Checks for the NASA C-MAPSS dataset and prints documented fetch instructions.

The dataset has no stable programmatic/authenticated API, so this script never
blindly curls a URL that may move or require auth. It only verifies presence and
tells the user exactly how to obtain the files, per data/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def missing_files(raw_dir: Path, subsets: list[str]) -> list[str]:
    missing = []
    for subset in subsets:
        for prefix in ("train", "test", "RUL"):
            f = raw_dir / f"{prefix}_{subset}.txt"
            if not f.exists():
                missing.append(f.name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for NASA C-MAPSS raw data files.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--subsets", nargs="+", default=["FD001"], help="Subsets to check for, e.g. FD001 FD002"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_files(raw_dir, args.subsets)

    if not missing:
        print(f"All required files present in {raw_dir}.")
        return 0

    print(f"Missing {len(missing)} file(s) in {raw_dir}: {', '.join(missing)}")
    print()
    print("See data/README.md for download instructions (NASA PCoE data repository")
    print("or the Kaggle mirror 'behrad3d/nasa-cmaps'). This script does not download")
    print("automatically since there is no stable, authenticated fetch URL.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
