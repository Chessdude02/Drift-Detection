This project supports two NASA/PCoE predictive-maintenance datasets, selected via
`dataset.type` in the training config (`config/training.yaml` for C-MAPSS,
`config/training_bearing.yaml` for IMS Bearing). Neither is bundled in this repo (both
are large, and PCoE's distribution terms ask that you download them directly).

# C-MAPSS dataset

This project trains on NASA's Commercial Modular Aero-Propulsion System Simulation
(C-MAPSS) turbofan engine degradation dataset.

## How to get it

Option A — NASA PCoE data repository:
1. Visit the NASA Prognostics Center of Excellence data set repository and find
   "Turbofan Engine Degradation Simulation Data Set" (CMAPSSData.zip).
2. Download and unzip it.
3. Copy `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (and FD002-FD004 if you
   want them) into `data/raw/`.

Option B — Kaggle mirror:
1. `pip install kaggle` and configure your API token (`~/.kaggle/kaggle.json`).
2. `kaggle datasets download -d behrad3d/nasa-cmaps -p data/raw --unzip`
3. Verify the files land as `data/raw/train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`.

## Verify

```
python -m pdm.data.download --subsets FD001
```

This only checks for file presence and tells you what's missing — it does not
fetch anything itself.

## Tests don't need the full dataset

`tests/fixtures/cmapss_sample.txt` is a tiny (~50 row, 2-unit) hand-trimmed sample
committed to the repo so unit tests and CI run fully offline.

# IMS Bearing dataset

The IMS Bearing run-to-failure vibration dataset (Center for Intelligent Maintenance
Systems, University of Cincinnati; originally hosted on NASA's PCoE data repository,
now mirrored via the PHM Society) records accelerometer channels as a set of snapshot
files, collected roughly every 10 minutes, until a bearing fails.
`src/pdm/data/ims_bearing.py` reads that snapshot-file format directly;
`src/pdm/data/bearing_features.py` extracts per-snapshot vibration health features
(RMS, kurtosis, skewness, crest factor, etc.) and constructs the run-to-failure RUL
target from snapshot order.

## Exact source

**Download:** <https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip> — the
current working mirror (NASA's original PCoE hosting has been retired/migrated; this
S3 URL is what the PHM Society's repository index at
<https://data.phmsociety.org/nasa/> now points to). NASA's own landing page, for
citation/context, is
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>.
Cite as: J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007), IMS,
University of Cincinnati.

**What's inside the zip — three independent test-to-failure experiments, each its own
folder of snapshot files (verify exact top-level nesting once you unzip; mirrors have
varied this):**

| Folder | Channels | Column layout | Bearing that actually failed |
|---|---|---|---|
| `1st_test/` | 8 | 2 accelerometers (x/y) per bearing: `b1x,b1y,b2x,b2y,b3x,b3y,b4x,b4y` | bearing3 (inner race), bearing4 (rolling element) |
| `2nd_test/` | 4 | 1 accelerometer per bearing: `b1,b2,b3,b4` | **bearing1** (outer race) |
| `3rd_test/` | 4 | 1 accelerometer per bearing: `b1,b2,b3,b4` | **bearing3** (outer race) |

Every file: whitespace/tab-delimited, no header, 20,480 rows (1 second at 20kHz),
filename = the snapshot's timestamp as `YYYY.MM.DD.HH.MM.SS` (lexicographic filename
order = chronological order — exactly what `pdm.data.ims_bearing.list_snapshot_files`
relies on).

## What this repo's code expects

`config/training_bearing.yaml` is written for the **4-channel format** (`channels:
[bearing1, bearing2, bearing3, bearing4]`) — that's `2nd_test/` or `3rd_test/`, **not**
`1st_test/` (8 columns, would need a different channel list and doesn't map 1:1 onto a
single `failure_channel`). The config currently assumes **`2nd_test`**
(`failure_channel: bearing1`); if you use `3rd_test` instead, change
`failure_channel` to `bearing1`.

To use it locally:
1. Download and extract the zip above.
2. Point `config/training_bearing.yaml`'s `dataset.raw_dir` (or `--raw-dir` on the
   CLI) directly at the `2nd_test/` folder — a flat directory of timestamp-named
   snapshot files, used as-is, no renaming.
3. In CI (`build-and-push-dev.yaml`), the same flat layout is expected under
   `data/raw_bearing/` after the GCS fetch step — see RUNBOOK.md's "Training data"
   section for the bucket path convention (upload `2nd_test/`'s files, not the folder
   itself, directly under the `vN/` prefix).

## Holdout set

The fixed, versioned holdout set (`scripts/build_holdout.py`, `data/holdout/
holdout_v1.csv`, RUNBOOK.md) must come from data that was **never used in training** —
for the real dataset, the natural choice is **`3rd_test/`** (a genuinely separate
physical experiment, not just a later time slice of the same run), with
`--failure-channel bearing3` for that build. Do not build the holdout from a
chronological tail of the same `2nd_test/` run used for training — see
`src/pdm/evaluation/holdout.py`'s module docstring for why the holdout must be built
from data outside the training run entirely, not carved out of it after the fact.

## Tests don't need the full dataset

`tests/fixtures/ims_bearing_sample/` (training) and `tests/fixtures/
ims_bearing_holdout_sample/` (holdout) are tiny, real-format synthetic samples — not
excerpts of the real dataset — with an injected degrading trend on a designated
channel (`bearing4` in both, an arbitrary fixture choice unrelated to which bearing
fails in the real data), committed to the repo so `tests/integration/
test_train_bearing.py` and the holdout tests run fully offline and fast. See each
fixture's generation notes / the module docstrings in `src/pdm/evaluation/holdout.py`
and `tests/integration/test_train_bearing.py` for what they do and don't prove.
