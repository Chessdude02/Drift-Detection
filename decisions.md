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

**Status:** Resolved by D15 (kept for the numbers)

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

**Status:** Resolved by D16 (kept for the numbers)

**Problem.** In calibration, a single broken sensor tripped the retrain check up to 9% of
the time (0% for a 0.5 std offset, 5-9% for larger offsets or a stuck sensor).

**Recommended fix.** When the sensor alert fires in the same check, hold the retrain for
a human to confirm. Not done here, to keep this change focused. It's a small change in
`run_drift_check.main`.

---

## D10. Known gaps found, not fixed in this change

**Status:** Open

- ~~**The drift job has no reference file inside the cluster.**~~ Resolved by D17: the
  job now downloads the Production model's reference from MLflow.
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

---

# Phase 2: making the loop real-world

## D11. Freeze 20 FD001 engines as a holdout that no model ever trains on

**Status:** Accepted

**Problem.** Judging maintenance decisions needs each engine's full history to failure.
NASA's test set stops each engine early, so you can't tell whether a maintenance call
was in time.

**Decision.** `scripts/build_cmapss_holdout.py` picked 20 of the 100 FD001 training
engines (seed 2024) and wrote them to `data/holdout/cmapss_FD001_holdout_v1.json`.
Training drops them (`dataset.holdout_units_file`), and every model is judged on them.
The file is frozen: a change means a new version, never an edit.

**Cost.**
- Training uses 80 engines instead of 100. With seed 42, internal validation RMSE went
  from 18.9 to 24.9. That split is noisy (±2.6 across seeds), so part of the change is
  which engines landed in validation. NASA test RMSE stayed about the same (22.7 vs
  22.3 ± 0.8).
- 20 failures is a small sample, so decision-cost differences under about 5-10% are
  noise (see D14).

## D12. Serve a calibrated interval, using scikit-learn for the bounds

**Status:** Accepted

**Decision.**
- Each model is now a bundle (`src/pdm/training/rul_model.py`): the LightGBM point
  model, two quantile models, and a conformal correction measured on the validation
  engines.
- `/predict` returns `rul_lower` and `rul_upper`, aiming for 90% of true values inside.

**What went wrong first.** LightGBM's quantile mode predicted exactly 125 for every row
as the upper bound, because about half the training targets sit at the 125 cap. The
bound never moved. scikit-learn's quantile model learns a real bound (about 30 near
failure, about 110 far from it), so both bounds use it.

**Result.** On NASA's test set, 91% of true values fall inside, with an average width
of 65 cycles.

**Cost.** 65 cycles is wide. The interval is honest, but it tells a planner less than
they'd want. A better model (Phase F) is the way to narrow it.

## D13. Score models on maintenance decisions, and ship the threshold with the model

**Status:** Accepted

**Decision.**
- `src/pdm/evaluation/decision.py` replays each engine and applies "schedule maintenance
  when the lower bound drops to H, with a 10-cycle lead time". It counts failures
  (late or missed) and wasted life, and prices them from `config/decision.yaml`.
- Training picks the cheapest H on the validation engines and stores it inside the
  model. `/predict` returns `maintenance_recommended`.

**Cost.** The costs are **placeholders** (a failure costs 10x a planned visit). The
chosen threshold depends heavily on that ratio, so it must be set with real numbers
before anyone relies on it.

## D14. Champion/challenger gate replaces the fixed RMSE gate for promotion

**Status:** Accepted

**Decision.**
- `src/pdm/evaluation/champion_challenger.py` re-scores the Production model and the
  candidate on the same frozen data each time: the 20 holdout engines plus NASA's
  test set.
- The candidate is promoted only if it's no worse on every check: cost within 5%, no
  extra failures, RMSE within 1, coverage at least 80%, RMSE at most 30.
- The first promotion needs a human to confirm.
- The retrain job now runs `python -m pdm.training.retrain`: train, then gate. A
  rejected candidate stays registered but is never promoted.

**Why "no worse" instead of "better".** A retrain on newer data that is no worse is
worth shipping, and 20 engines can't reliably show a small improvement.

**Weakness found while testing (Open).** Changing only the random seed moved decision
cost by 8%, more than the 5% tolerance. A retrain on identical data (v3) was
**rejected** because the model it faced (v2) had a lucky seed. So promotions partly
depend on luck. Fix: reduce seed-to-seed variance, for example by averaging several
seeds (tested in Phase F), or by widening the tolerance, at the cost of catching fewer
real regressions.

## D15. Require an engine ID on every prediction, and enforce 5+ engines per window

**Status:** Accepted

**Problem.** Without knowing which engine a reading came from:
- the drift check can't count engines (D8), and
- failures and maintenance can't be matched back to predictions to make new labels
  (Phase D).

**Decision.**
- `/predict` takes `asset_id`, plus optional `cycle` and `observed_at`.
- `asset_id` is required by default (`serving.yaml input_validation.require_asset_id`).
- The inference log gained those columns, plus the model version and any input
  warnings. Existing databases are upgraded in place.
- The drift job reads the last 24 hours (capped at 500 rows) instead of the last 500
  rows, and skips the check when the window has fewer than 5 engines.
- The `DriftCheckSkipped` alert fires if that goes on for 6 hours.

**Cost.**
- **This breaks the API:** clients that don't send `asset_id` get a 422 error. Setting
  `require_asset_id: false` restores the old behaviour, but then the engine minimum
  can't be enforced.
- A small fleet that never reaches 5 engines a day gets no drift monitoring. That's
  deliberate: a check that is wrong 15-60% of the time is worse than none, and the
  alert makes the gap visible.

## D16. Validate inputs at the door; hold drift retrains when a sensor looks broken

**Status:** Accepted

**Decision.**
- Missing features, a missing `asset_id`, or NaN/infinite values get a 422 error,
  counted in `pdm_input_rejected_total{reason}`.
- Values outside the training range (plus a 5% margin, stored in the model) are
  **accepted but flagged**: the response carries `input_warnings`, and the count goes
  to `pdm_input_out_of_range_total{feature}`. A genuinely unusual reading should reach
  the drift and sensor checks, not be thrown away.
- If a check says "retrain" and the sensor-fault check also fires, the retrain is held.
  The `RetrainHeldForSensorFault` alert fires and a human decides.
  (`sensor_check.hold_retrain_on_fault`.)

**Cost.** A held retrain waits for a person. That's intended: an automatic retrain on
possibly bad data is the riskier default.

## D17. The drift reference ships with the model, not as a hand-seeded file

**Status:** Accepted

**Problem.**
- The drift job read `data/processed/reference.parquet`, seeded by hand. Nothing kept
  that file in step with the model in Production.
- Inside the cluster the file didn't exist at all (D10).

**Decision.**
- Every training run logs its own reference (training features plus out-of-fold
  predicted remaining life) as an MLflow artifact next to the model.
- The drift job loads the reference of whichever version is in Production
  (`reference.source: model`, `src/pdm/drift/reference.py`). The job already had
  `MLFLOW_TRACKING_URI`, so nothing needs mounting.
- If the artifact is missing, the job fails loudly instead of falling back to the
  file. Comparing traffic against another model's training data is the mismatch this
  change removes. `source: file` still works for local runs.

**Checked.** Train, then promote, then 500 real rows from 20 unseen holdout engines,
then the real drift job. Result: it loaded "cmapss_rul v1", saw 20 engines, drift 0.0,
action `none`. Peak memory was 377 MiB against the job's 512 MiB limit, so the limit
was raised to 768 MiB.

**Cost.** Training takes a few seconds longer: 5 extra model fits to build the
out-of-fold predictions.

## D18. A runbook entry for every alert

**Status:** Accepted

`RUNBOOK.md` now has an "Alerts" section: for each of the 7 alerts, what it means and
what to do, plus how to read a gate decision. Alerts nobody knows how to act on get
ignored.
