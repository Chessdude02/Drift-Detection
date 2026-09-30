# Decisions

Why the project is the way it is. One entry per decision: what the problem was, what was
considered, what was chosen, and what it costs. Numbers come from real NASA C-MAPSS data
unless stated otherwise; `executions.md` has the commands that produced them.

Status values: **Accepted** (in the code), **Rejected** (considered and turned down),
**Open** (known problem, not fixed yet).

---

## D1. Measure the model on NASA's official test set, over several seeds

**Status:** Accepted

**Problem.** `train.py` reports one number: RMSE on a random 20% of training engines,
over every cycle, from one split. On FD001 that number was 18.9. It isn't the number the
C-MAPSS literature reports, and one split of ~20 engines is noisy.

**Decision.** `scripts/benchmark_real_data.py` scores the unchanged pipeline on the
official test set (the last cycle of each test engine vs `RUL_FD00x.txt`), on all four
subsets, over 5 seeds, next to a "predict the average" baseline.

**Result.** FD001 test RMSE is 22.3 ± 0.8, not 18.9. The internal val RMSE across seeds
is 21.3 ± 2.6, so 18.9 was a lucky split. The baseline scores 42.0.

**Cost / what it tells us.**
- The model is mediocre. Published FD001 results are roughly 12-18 RMSE.
- The CI gate (`max_rmse: 35`) passes it easily, so the gate cannot tell a good model
  from a weak one. Not changed here (see D10).

---

## D2. Fix the serving/drift feature names to match training (`_roll_mean_20`)

**Status:** Accepted

**Problem.** Training builds 20-cycle rolling means (`max(rolling_windows)` = 20).
`config/serving.yaml`, the Helm copy of it, and `config/drift.yaml` listed 10-cycle
columns (`_roll_mean_10`). Verified against a real trained model: MLflow's schema check
rejected **every** `/predict` call, and the drift job would find no shared columns and
exit.

**Options.**
1. Point serving and drift at `_roll_mean_20`. Keeps the model that was benchmarked.
2. Change training to a 10-cycle window. Changes the model and every number in D1.

**Decision.** Option 1. Added `tests/unit/test_config_consistency.py`, which fails if
serving or drift columns stop matching training features.

---

## D3. Set the drift threshold from real data, not Evidently's default

**Status:** Accepted (threshold later re-set by D4)

**Problem.** Evidently's default per-column test (normed Wasserstein distance, 0.1)
flagged 90-100% of windows of **unseen but normal** FD001 engines as drift. Engines
differ from each other by more than 0.1 on their own, so the retrain job would fire on
nearly every check.

**Decision.** Set the per-column threshold from data (`scripts/calibrate_drift.py`):
- Pick it on normal windows from 60 engines. It's the 99th percentile of the smallest
  threshold that would not fire.
- Report every rate on 40 other engines that played no part in picking it.

The threshold is `drift.stattest_threshold` in `config/drift.yaml`.

---

## D4. Match the reference to the fleet's life stage, instead of a mid-life reference

**Status:** Accepted (the mid-life option was **Rejected**)

**Problem.** After D3, a fleet that was younger or older than the training data still
looked like drift. Engines wear, so sensor values move with age. The reference is built
from engines run all the way to failure.

**Correction to an earlier claim.** It was first said that "a live fleet is mostly
mid-life." That's only true for some fleets. A fleet that replaces engines as they fail
sends readings from every life stage evenly, which matches the run-to-failure reference.
The real problem is a fleet whose **age mix shifts**, younger or older.

**Options compared** (`scripts/experiments/compare_drift_methods.py`,
`reports/drift_method_comparison.json`; each option's threshold picked on 60 engines,
rates on 40 unseen engines):

| Scenario | Full run-to-failure reference | Mid-life reference (requested) | Life-stage matched |
|---|---|---|---|
| Normal steady fleet | 0% | 0% | 0% |
| Normal 5-engine fleet | 9% | 3% | 5% |
| Young fleet | 66% | 0% | 1% |
| NASA test set (young) | 86% | 0% | 0% |
| Old fleet (near failure) | **100%** | **100%** | 0% |
| FD002 (real shift) | 100% | 100% | 100% |
| FD003 (new fault mode) | 100% | **0%** | 100% |

- **Mid-life reference: rejected.** It fixes young fleets but still false-alarms on old
  fleets. To stay quiet on normal data its threshold had to loosen to 1.35, which made
  it **miss FD003 entirely**.
- **Life-stage matched: chosen.** Before each check, the reference is resampled so its
  mix of predicted remaining life matches the current window's. Only changes at the
  same life stage count as drift.

**How it works.**
- `src/pdm/drift/life_stage.py`. The reference stores an out-of-fold prediction per row
  (`predicted_rul`), from a model that never saw that engine, like a live prediction.
- The inference log already stores the served model's prediction for each window.
- Neither side needs the true remaining life, which isn't known in production.

**Cost.**
- `scripts/seed_reference_data.py` now trains 5 models to build the reference, which
  takes seconds on FD001.
- An old reference file without `predicted_rul` still works, but falls back to no
  matching and logs a warning.
- If drift makes the model's predictions themselves wrong, matching uses wrong ages.
  On FD002 and FD003 this didn't hide the drift (still caught 97-100%), but it's a
  weakness in principle.

---

## D5. A per-sensor residual check that only raises an alert

**Status:** Accepted

**Problem.** The retrain trigger needs more than half of the 14 columns to drift, so one
broken sensor is invisible to it. An offset on 3 of 14 sensors was never caught, even at
2 standard deviations.

**Options compared** (same experiment as D4; rule = "any one sensor over threshold"):

| Scenario | Per-sensor distance | Residual check |
|---|---|---|
| One sensor offset 0.5 std | 1% | 100% |
| One sensor offset 1 std | 100% | 100% |
| One sensor stuck | 86% | 100% |
| False alarms, normal and ageing fleets | up to 6% | up to 2% |

**Decision.** The residual check (`src/pdm/drift/sensor_check.py`):
- Each sensor is predicted from the other 13 by a linear model fitted on the reference.
  Wear and engine-to-engine differences move sensors together, so prediction errors
  stay stable. A biased or stuck sensor breaks the pattern.
- It's pushed as the `pdm_sensor_fault{sensor=...}` gauge, which drives the
  `SensorFaultSuspected` alert.

**It never triggers retraining.**
- The residual check also fires when half the columns shift together, so it could have
  served as the retrain rule too. But a single broken sensor sets it off 20-100% of the
  time (66-100% for an offset, 20% when stuck).
- Retraining on a broken sensor's data would bake the fault into the model. The right
  response is to fix the sensor.

**Result** (calibrated, 40 unseen engines): the alert fires 100% for a single-sensor
offset of 0.5-2 std or a stuck sensor, and names the right sensor first 100% of the
time. False alarms on normal and ageing fleets are 0-3.3%.

**Cost.**
- A broken sensor still triggers the *retrain* check up to 9% of the time. See D9.

---

## D6. Leave shadow traffic out of the drift window

**Status:** Accepted

**Problem.** Mirrored (shadow) requests are logged a second time with the candidate
model's prediction. Counting them double-counts inputs, and since D4 it would also mix
another model's predictions into life-stage matching.

**Decision.** `run_drift_check.main` drops rows with `shadow = 1`.

**Cost.** A window can hold fewer than `lookback_rows` rows when shadow traffic is on.

---

## D7. Pin SQLAlchemy below 2.1

**Status:** Accepted

**Problem.** A fresh `pip install` pulled SQLAlchemy 2.1, which removed a class MLflow
2.x imports. `import mlflow` with a SQLite store crashed, so training could not run on a
clean machine.

**Decision.** Add `sqlalchemy>=2.0,<2.1` to `requirements.txt`. Checked on a clean
install: it resolved to 2.0.54, and an MLflow run logged to SQLite.

**Cost.** Revisit when moving to MLflow 3.

---

## D8. Windows need about 5+ engines; not enforced yet

**Status:** Open

**Problem.** With few engines in a window, one engine's quirks look like drift or a fault
(benchmark, false retrain / false sensor alert):

| Engines in window | 1 | 2 | 3 | 5 | 10+ |
|---|---|---|---|---|---|
| False retrain | 60% | 15% | 5% | 0% | 0% |
| False sensor alert | 85% | 60% | 15% | 5% | 0% |

**Why not fixed.** The inference log stores features and the prediction, but not which
engine a reading came from, so the drift job can't count engines.

**Recommended fix.**
1. Add an engine/asset ID to the `/predict` request and the inference log.
2. Skip the check, or widen the window, until it covers at least 5 engines.

---

## D9. A broken sensor can still trigger a retrain

**Status:** Open

**Problem.** In calibration, a single broken sensor tripped the retrain check up to 9% of
the time (0% for a 0.5 std offset, 5-9% for larger offsets or a stuck sensor).

**Recommended fix.** When the sensor alert fires in the same check, hold the retrain for
a human to confirm. Not done here, to keep this change focused. It's a small change in
`run_drift_check.main`.

---

## D10. Known gaps found, not fixed in this change

**Status:** Open

- **The drift job has no reference file inside the cluster.** `docker/drift.Dockerfile`
  doesn't copy `data/processed/reference.parquet`, and
  `deploy/k8s/drift-check-cronjob.yaml` doesn't mount it. The job would fail at its
  first step in the cluster. Needs a volume or a build step.
- **The model-quality gate is too loose.** `max_rmse: 35` against a real test RMSE of
  22.3 (D1).
- **The model is mediocre.** 22.3 RMSE on FD001 against published results of roughly
  12-18.
- **One unit test fails** on Linux, before and after these changes:
  `tests/unit/test_config.py::test_set_experiment_with_artifact_root_none_uses_mlflow_default`.
  Likely a Windows/Linux path difference in how MLflow reports the artifact location.
- **All drift numbers are FD001 only**, with 120 windows per scenario. A measured 0%
  means "probably under ~2.5%", not "never". Recalibrate if the features, window size,
  model or reference data change.
- **The bearing dataset (IMS) was not benchmarked here.** The RUNBOOK's bearing numbers
  come from one failure event.
