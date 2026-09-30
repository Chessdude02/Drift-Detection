# Decisions

**Project:** predictive-maintenance MLOps on NASA C-MAPSS. It predicts each engine's
remaining useful life (RUL, the cycles left before failure), detects input drift and
broken sensors, and safely retrains from real maintenance outcomes.

This file logs every meaningful decision, so anyone can see why the project is built
the way it is.
- Entries are never deleted. A changed decision is marked "Superseded" and a new entry
  explains why.
- Numbers come from real NASA data unless stated otherwise.
- The commands that produced them are in [`run_log.md`](run_log.md). How the code runs
  is in [`execution.md`](execution.md).

Decisions made before this log started (LightGBM, FastAPI, MLflow registry, Argo
Rollouts canary, SQLite inference log, Evidently) are **not** logged here. Their
reasons were never written down, and reconstructing them would be guesswork. Where the
code states a reason, it's in the relevant module's docstring.

## Summary

| ID | Title | Status | Date |
|---|---|---|---|
| D-1 | Measure the model on NASA's official test set, over 5 seeds | Accepted | 2026-09-30 |
| D-2 | Fix serving/drift feature names to match training (`_roll_mean_20`) | Accepted | 2026-09-30 |
| D-3 | Set the drift threshold from real data (global 0.47) | Superseded by D-4 | 2026-09-30 |
| D-4 | Life-stage-matched drift reference (mid-life reference rejected) | Accepted | 2026-09-30 |
| D-5 | Per-sensor residual check that only raises an alert | Accepted (verdict logic refined by D-21) | 2026-09-30 |
| D-6 | Leave shadow traffic out of the drift window | Accepted | 2026-09-30 |
| D-7 | Pin SQLAlchemy below 2.1 | Accepted | 2026-09-30 |
| D-8 | Windows need about 5+ engines (finding, not enforced) | Superseded by D-15 | 2026-09-30 |
| D-9 | A broken sensor can still trigger a retrain (finding) | Superseded by D-16 | 2026-09-30 |
| D-10 | Record known gaps instead of widening each change | Accepted | 2026-09-30 |
| D-11 | Freeze 20 FD001 engines as a never-trained-on holdout | Accepted | 2026-09-30 |
| D-12 | Serve a calibrated interval; scikit-learn quantile models | Accepted | 2026-09-30 |
| D-13 | Score models on maintenance decisions; threshold ships with the model | Accepted | 2026-09-30 |
| D-14 | Champion/challenger gate replaces the fixed RMSE gate | Accepted | 2026-09-30 |
| D-15 | Require an engine ID; 24h windows; skip under 5 engines | Accepted | 2026-09-30 |
| D-16 | Validate inputs; hold drift retrains when a sensor looks broken | Accepted (hold condition superseded by D-21) | 2026-09-30 |
| D-17 | Drift reference ships with the model in MLflow | Accepted | 2026-09-30 |
| D-18 | A runbook entry for every alert | Accepted | 2026-09-30 |
| D-19 | New labels from recorded outcomes, including maintained engines | Accepted | 2026-09-30 |
| D-20 | Prove the loop adapts on a real new fleet (FD003 replay) | Accepted | 2026-09-30 |
| D-21 | Tell a broken sensor from a fleet-wide shift | Accepted | 2026-09-30 |
| D-22 | End-to-end test of the whole loop without Docker | Accepted | 2026-09-30 |
| D-23 | Move docs to `/docs`; split run log from how-it-runs doc | Accepted | 2026-09-30 |
| D-24 | Static call graph with pyan3 + Graphviz | Accepted | 2026-09-30 |
| D-25 | One OpenMP thread per serving process | Accepted | 2026-09-30 |
| D-26 | Cross-validated calibration of interval and threshold | Accepted | 2026-09-30 |
| D-27 | Trend features (5- minus 20-cycle mean) | Proposed | 2026-09-30 |
| D-28 | Hyperparameter tuning and seed averaging | Rejected | 2026-09-30 |

---

## D-1: Measure the model on NASA's official test set, over 5 seeds
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** `train.py` reported one number, validation RMSE (root-mean-square error,
  in cycles) on a random 20% of training engines, from one split: 18.9 on FD001. That
  isn't the number the C-MAPSS literature reports, and one split of about 20 engines is
  noisy.
- **Options considered:**
  1. Keep `train.py`'s validation RMSE.
     *Pro:* already exists. *Con:* not comparable with published work; a single split.
  2. Score the last cycle of each engine in NASA's test set against `RUL_FD00x.txt`,
     over several seeds.
     *Pro:* the standard benchmark; shows noise. *Con:* slower; needs the NASA files.
  3. Option 2 plus a "predict the average" baseline.
     *Pro:* shows whether the model beats doing nothing clever. *Con:* one more number.
- **Decision:** Option 3 (`scripts/benchmark_real_data.py`), on all four subsets.
- **Factors that led to it:** Comparability with published results; the noise had to be
  visible to judge any single number.
- **Trade-offs accepted:** About 5 minutes per run; it measures the pipeline as
  configured, not a tuned model.
- **Expected effect:** A test RMSE close to the 18.9 validation number.
- **Actual effect:**
  - FD001 test RMSE is **22.3 ± 0.8**, not 18.9. Across seeds, validation RMSE is
    21.3 ± 2.6, so 18.9 was a lucky split.
  - FD002 21.4, FD003 21.7, FD004 24.0.
  - The average-predicting baseline scores 42–46.
  - Published FD001 results are roughly 12–18, so the model is mediocre.
  - The old gate (`max_rmse: 35`) passes it easily.
- **Evidence:** `reports/benchmark_cmapss.json`, `README.md` "Measured results",
  commit `a4fffd4`.
- **Related:** D-11, D-14.

## D-2: Fix serving/drift feature names to match training (`_roll_mean_20`)
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:**
  - Training builds 20-cycle rolling means (`max(rolling_windows)` = 20).
  - `config/serving.yaml`, its Helm copy, and `config/drift.yaml` listed 10-cycle
    columns.
  - MLflow's input-format check rejected every `/predict` call, and the drift job found
    no shared columns.
- **Options considered:**
  1. Point serving and drift at `_roll_mean_20`.
     *Pro:* keeps the benchmarked model; config-only change. *Con:* clients must send
     20-cycle features.
  2. Change training to a 10-cycle window.
     *Pro:* matches the old configs. *Con:* changes the model and invalidates every
     number in D-1.
- **Decision:** Option 1, plus `tests/unit/test_config_consistency.py`, which fails if
  the configs drift apart again.
- **Factors that led to it:** Smallest change; keeps the measured model.
- **Trade-offs accepted:** Any existing client sending `_roll_mean_10` must change.
- **Expected effect:** `/predict` works; the drift job finds its columns.
- **Actual effect:** Confirmed against a real trained model. Before: an MLflow error.
  After: a prediction (121.3 on a dummy row).
- **Evidence:** commit `a4fffd4`, `run_log.md` section 2.
- **Related:** D-17.

## D-3: Set the drift threshold from real data (global 0.47)
- **Date:** 2026-09-30
- **Status:** Superseded by D-4
- **Context:** Evidently's default per-column test flagged 90–100% of windows of
  *normal* unseen FD001 engines as drift. That test is the normed Wasserstein distance:
  how far apart two distributions are, divided by the reference's spread, with a
  threshold of 0.1. Engines differ from each other by more than 0.1 on their own, so
  the retrain job would fire on almost every check.
- **Options considered:**
  1. Keep the default.
     *Pro:* no work. *Con:* useless; it's almost always "drift".
  2. Raise the threshold to a value picked from normal data on some engines, and check
     it on others.
     *Pro:* evidence-based; honest held-out check. *Con:* specific to these features
     and window size.
  3. Switch to a p-value test (Kolmogorov–Smirnov).
     *Pro:* familiar. *Con:* with 500+ rows, tiny differences become "significant";
     same problem.
- **Decision:** Option 2 (`scripts/calibrate_drift.py`). The threshold is the 99th
  percentile over normal windows from 60 engines, checked on 40 other engines. It came
  out at 0.47.
- **Factors that led to it:** Test evidence: the default fired on 90%+ of normal
  windows.
- **Trade-offs accepted:** Must be recalibrated if features, window size or reference
  change.
- **Expected effect:** Few false alarms, real shifts still caught.
- **Actual effect:**
  - False alarms 0% (large fleet) and 4.5% (5-engine fleet); FD002 caught 100%,
    FD003 99.5%.
  - **But** a younger or older fleet still triggered 66–100% of the time, which led
    to D-4.
- **Evidence:** commit `a864e9a`, `run_log.md` section 4.
- **Related:** D-4 (supersedes it; threshold now 0.32).

## D-4: Life-stage-matched drift reference (mid-life reference rejected)
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:**
  - Sensors change as engines wear, so a fleet younger or older than the training data
    looked like drift.
  - The user asked for a reference built from mid-life engine data.
- **Options considered** (each with its own threshold picked on 60 engines, rates
  measured on 40 unseen engines):
  1. The full run-to-failure reference, as before.
     *Pro:* simple. *Con:* young fleet 66% and old fleet 100% false alarms.
  2. A mid-life reference (requested).
     *Pro:* young fleet 0%. *Con:* old fleet still 100%; its threshold had to loosen to
     1.35, so it **missed FD003 entirely (0%)**.
  3. Life-stage matched: before each check, resample the reference so its mix of
     predicted RUL matches the window's.
     *Pro:* young 1%, old 0%, FD002/FD003 100%. *Con:* needs predicted RUL on both
     sides; if drift corrupts predictions, the matching is off.
- **Decision:** Option 3 (`src/pdm/drift/life_stage.py`). The per-column threshold was
  recalibrated to **0.32**.
- **Factors that led to it:** Option 3 was the only one without ageing false alarms
  that still caught both real shifts. Option 2's premise ("live fleets are mostly
  mid-life") was wrong, and was corrected in the discussion: a fleet that replaces
  engines as they fail covers every life stage evenly.
- **Trade-offs accepted:**
  - The reference stores out-of-fold predictions: 5 extra model fits.
  - Old reference files fall back to no matching, with a warning.
- **Expected effect:** Ageing stops triggering retrains; real shifts are still caught.
- **Actual effect** (40 unseen engines, 120 windows each):
  - Retrain false alarms on normal and ageing fleets: 0–0.8%.
  - FD002 caught 100%, FD003 97%.
  - The production Evidently path agrees: normal 0%, FD003 100%.
- **Evidence:** `reports/drift_method_comparison.json`,
  `reports/drift_calibration.json`, `scripts/experiments/compare_drift_methods.py`,
  commit `c3f8ac0`.
- **Related:** D-3, D-5, D-17.

## D-5: Per-sensor residual check that only raises an alert
- **Date:** 2026-09-30
- **Status:** Accepted (the fault-vs-shift verdict was refined by D-21)
- **Context:** The retrain trigger needs more than half of the 14 columns to drift, so
  one broken sensor was invisible to it. A 3-of-14 offset was never caught, even at
  2 standard deviations.
- **Options considered** (rule: "any one sensor over threshold"):
  1. Per-sensor distance on raw values.
     *Pro:* simple. *Con:* missed a 0.5 std offset (1% caught); a stuck sensor 86%.
  2. Residual check: predict each sensor from the other 13 with a linear model, and
     watch that prediction's error (the "residual").
     *Pro:* wear moves sensors together, so residuals ignore ageing; caught 0.5 std
     and stuck sensors 100%. *Con:* one broken sensor also nudges the others'
     residuals.
  3. Also use the residual check as the retrain trigger.
     *Pro:* one mechanism. *Con:* one broken sensor would trigger a retrain 20–100% of
     the time.
- **Decision:** Option 2, **alert only** (`SensorFaultSuspected`). It never triggers a
  retrain.
- **Factors that led to it:** Test evidence; retraining on a broken sensor's data would
  bake the fault into the model.
- **Trade-offs accepted:** Needs its own threshold (0.46) and recalibration.
- **Expected effect:** Single broken sensors get caught and named; no automatic
  retrains from them.
- **Actual effect:**
  - Offsets of 0.5–2 std and stuck sensors: caught 100%, right sensor named first
    100%.
  - False alarms 0–3.3%.
  - It also fired on every sensor during fleet-wide shifts, which D-21 fixes.
- **Evidence:** `reports/drift_calibration.json`, `src/pdm/drift/sensor_check.py`,
  commit `c3f8ac0`.
- **Related:** D-9, D-16, D-21.

## D-6: Leave shadow traffic out of the drift window
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Mirrored ("shadow") requests are logged a second time, with a candidate
  model's prediction. The drift check reads predictions for life-stage matching (D-4).
- **Options considered:**
  1. Include them.
     *Pro:* no code. *Con:* double-counts inputs; mixes in another model's predictions.
  2. Exclude them in the drift job.
     *Pro:* one line. *Con:* a window can hold fewer rows than `lookback_rows`.
  3. Stop logging shadow rows.
     *Pro:* cleaner log. *Con:* loses the data for comparing the shadow model.
- **Decision:** Option 2.
- **Factors that led to it:** Simplicity; keeps the shadow data for other uses.
- **Trade-offs accepted:** Slightly smaller windows when shadowing is on.
- **Expected effect:** Drift and matching use live traffic only.
- **Actual effect:** Covered by unit tests (`test_read_window_by_time_drops_shadow`).
  Not measured on real mirrored traffic, because there is none in this environment.
- **Evidence:** `src/pdm/drift/run_drift_check.py` `read_window`.
- **Related:** D-4, D-19 (labels also skip shadow rows).

## D-7: Pin SQLAlchemy below 2.1
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** A fresh install pulled SQLAlchemy 2.1, which removed a class MLflow 2.x
  imports. `import mlflow` with a SQLite store crashed.
- **Options considered:**
  1. Pin `sqlalchemy>=2.0,<2.1`.
     *Pro:* one line. *Con:* must be revisited later.
  2. Upgrade to MLflow 3.
     *Pro:* current. *Con:* the registry-stage API this repo uses changed; a large
     migration.
  3. Pin MLflow to an exact version.
     *Pro:* reproducible. *Con:* doesn't fix the SQLAlchemy range.
- **Decision:** Option 1.
- **Factors that led to it:** Smallest fix that makes a fresh install work.
- **Trade-offs accepted:** Revisit when moving to MLflow 3.
- **Expected effect:** Clean installs work.
- **Actual effect:** A clean install resolved SQLAlchemy 2.0.54, and an MLflow run
  logged to SQLite.
- **Evidence:** `requirements.txt`, `run_log.md` section 7.
- **Related:** none.

## D-8: Windows need about 5+ engines (finding, not enforced)
- **Date:** 2026-09-30
- **Status:** Superseded by D-15
- **Context:** The benchmark showed false alarms when a window covers few engines.
- **Options considered:**
  1. Enforce a minimum engine count.
     *Pro:* removes the false alarms. *Con:* impossible then; requests carried no
     engine ID.
  2. Document it now and enforce it once IDs exist.
     *Pro:* honest. *Con:* the gap stays open until then.
- **Decision:** Option 2 at the time; D-15 later enforced it.
- **Factors that led to it:** No engine ID in the inference log.
- **Trade-offs accepted:** Small-fleet windows stayed unreliable until D-15.
- **Expected effect:** n/a (finding).
- **Actual effect:**

  | Engines in window | 1 | 2 | 3 | 5 | 10+ |
  |---|---|---|---|---|---|
  | False retrain | 60% | 15% | 5% | 0% | 0% |
  | False sensor alert | 85% | 60% | 15% | 5% | 0% |

- **Evidence:** `reports/benchmark_cmapss.json`, `run_log.md` section 8.
- **Related:** D-15.

## D-9: A broken sensor can still trigger a retrain (finding)
- **Date:** 2026-09-30
- **Status:** Superseded by D-16
- **Context:** In calibration, a single broken sensor tripped the retrain check up to 9%
  of the time.
- **Options considered:**
  1. Hold the retrain when a sensor alert fires in the same check.
     *Pro:* safe. *Con:* needs a person to release it.
  2. Accept it.
     *Pro:* no work. *Con:* risk of training on bad data.
- **Decision:** Recorded, then implemented as option 1 in D-16.
- **Factors that led to it:** Kept the earlier change focused.
- **Trade-offs accepted:** The risk remained until D-16.
- **Expected effect:** n/a (finding).
- **Actual effect:** Retrain trigger rates for a broken sensor: 0% at 0.5 std, 7% at
  1 std, 9% at 2 std, 5% when stuck.
- **Evidence:** `reports/drift_calibration.json`.
- **Related:** D-16, D-21.

## D-10: Record known gaps instead of widening each change
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Several problems were found that were outside the change in hand.
- **Options considered:**
  1. Fix everything at once.
     *Pro:* nothing left open. *Con:* huge, hard-to-review changes.
  2. Record each gap here and fix it in a focused change.
     *Pro:* reviewable. *Con:* gaps stay open for a while.
- **Decision:** Option 2.
- **Factors that led to it:** Reviewability.
- **Trade-offs accepted:** Gaps persist until fixed.
- **Expected effect:** Nothing is forgotten.
- **Actual effect:** Status of the recorded gaps:
  - The drift job had no reference in the cluster: **fixed by D-17**.
  - The quality gate was too loose: **replaced by D-14**.
  - The model is mediocre (22.3 against published 12–18): **open**, planned model
    work.
  - `tests/unit/test_config.py::test_set_experiment_with_artifact_root_none_uses_mlflow_default`
    fails on Linux before and after every change here: **open**. It's likely a
    Windows/Linux path difference.
  - Drift numbers are for FD001 only, and "0%" means "probably under ~2.5%" with 120
    windows: **open, by nature**.
  - The IMS bearing dataset was not benchmarked: **open**.
- **Evidence:** this file.
- **Related:** D-14, D-17.

## D-11: Freeze 20 FD001 engines as a never-trained-on holdout
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Scoring maintenance decisions (D-13) needs each engine's full history to
  failure. NASA's test set cuts every engine off before failure.
- **Options considered:**
  1. Use NASA's test set only.
     *Pro:* no training data lost. *Con:* can't tell whether a maintenance call was in
     time.
  2. Freeze 20 of the 100 training engines as a holdout.
     *Pro:* real run-to-failure histories never seen in training. *Con:* 20% less
     training data; 20 failures is a small sample.
  3. Use a different subset (FD003) as the holdout.
     *Pro:* no FD001 loss. *Con:* a different fault mode, not the same population.
- **Decision:** Option 2. `scripts/build_cmapss_holdout.py` (seed 2024) wrote
  `data/holdout/cmapss_FD001_holdout_v1.json`. The file is frozen; changes go in a new
  version.
- **Factors that led to it:** The only option with same-population run-to-failure data.
- **Trade-offs accepted:** Less training data; noisy cost estimates (see D-14).
- **Expected effect:** Slightly worse model; a trustworthy judging set.
- **Actual effect:** NASA test RMSE is unchanged in practice (22.7 vs 22.3 ± 0.8). With
  seed 42, internal validation RMSE went from 18.9 to 24.9, but that split is noisy
  (± 2.6).
- **Evidence:** `run_log.md` section 10, commit `54570a3`.
- **Related:** D-13, D-14.

## D-12: Serve a calibrated interval; scikit-learn quantile models
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** A planner needs to know how sure the model is, not just one number.
- **Options considered:**
  1. Conformal interval around the point prediction: a fixed ± width set from errors on
     unseen engines.
     *Pro:* simple, guaranteed coverage. *Con:* the same width everywhere.
  2. LightGBM quantile models plus conformal correction (CQR, "conformalized quantile
     regression").
     *Pro:* the width adapts per row. *Con:* **in practice the upper bound predicted
     exactly 125 for every row**. About half the targets sit at the 125 cap, so the
     model starts at 125 and never moves.
  3. scikit-learn `HistGradientBoostingRegressor` quantile models plus the conformal
     correction.
     *Pro:* learned a real bound (about 30 near failure, about 110 far from it).
     *Con:* slower to predict than LightGBM.
- **Decision:** Option 3 (`src/pdm/training/rul_model.py`). `/predict` returns
  `rul_lower` and `rul_upper` with a 90% coverage target.
- **Factors that led to it:** Test evidence: option 2 was broken.
- **Trade-offs accepted:** Three models per request; wide intervals.
- **Expected effect:** About 90% of true values inside the interval.
- **Actual effect:**
  - NASA test set: **91% coverage, 65 cycles wide** on average. The first LightGBM
    attempt was 86 wide with a constant upper bound.
  - **Found later:** requests took about 72 ms each during the FD003 replay (D-20). On
    an idle machine a full `POST /predict` takes about 17 ms. The 72 ms came from CPU
    contention making every model call's thread pool thrash, which is fixed by D-25.
- **Evidence:** `run_log.md` section 10, commit `54570a3`.
- **Related:** D-13.

## D-13: Score models on maintenance decisions; threshold ships with the model
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Nobody acts on "RMSE 22". They act on "schedule maintenance when
  predicted life drops below X".
- **Options considered:**
  1. Keep RMSE only.
     *Pro:* standard. *Con:* says nothing about missed failures or wasted life.
  2. A classification score at a fixed horizon (like the bearing pipeline's F2).
     *Pro:* exists in the repo. *Con:* ignores timing and wasted life.
  3. Replay each engine, apply "maintain when the lower bound ≤ H" with a 10-cycle lead
     time, and price failures and wasted life; pick H per model.
     *Pro:* measures the decision itself. *Con:* depends on cost numbers.
- **Decision:** Option 3 (`src/pdm/evaluation/decision.py`). H is chosen on validation
  engines and stored in the model; `/predict` returns `maintenance_recommended`.
- **Factors that led to it:** It's the business outcome.
- **Trade-offs accepted:** **The costs in `config/decision.yaml` are placeholders** (a
  failure costs 10x a planned visit). The chosen H depends heavily on that ratio.
- **Expected effect:** Few or no unplanned failures, at the price of some wasted life.
- **Actual effect:** v1 on the 20 holdout engines: 0 failures, 26.8 cycles wasted per
  engine, cost 12.7 per engine, H = 9.
- **Evidence:** `run_log.md` section 10.
- **Related:** D-11, D-14.

## D-14: Champion/challenger gate replaces the fixed RMSE gate
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** `max_rmse: 35` passed a mediocre model; a gate that only checks a fixed
  number can't tell whether a new model is worse than the current one.
- **Options considered:**
  1. Tighten the fixed threshold.
     *Pro:* simple. *Con:* still ignores the current model; picking the number is
     arbitrary.
  2. Compare against scores frozen at the last promotion (the bearing pipeline's way).
     *Pro:* cheap. *Con:* old numbers become incomparable when scoring code or data
     change.
  3. Re-score the Production model and the candidate on the same frozen data every
     time, with "no worse than" rules.
     *Pro:* always like-for-like. *Con:* slower; 20 engines are noisy.
- **Decision:** Option 3 (`src/pdm/evaluation/champion_challenger.py`,
  `config/champion_challenger.yaml`):
  - cost within 5% of the champion's, no extra failures, RMSE within 1;
  - coverage at least 80%, RMSE at most 30;
  - a human confirms the very first promotion.

  The retrain job runs `python -m pdm.training.retrain`: train, then gate.
- **Factors that led to it:** Comparability; "no worse" suits retrains on newer data.
- **Trade-offs accepted:** Tolerances are needed because the holdout is small.
- **Expected effect:** Bad candidates are blocked; equal or better ones are promoted.
- **Actual effect:**

  | Model | Seed | Cost per engine | Result |
  |---|---|---|---|
  | v1 | 42 | 12.68 | promoted after human confirmation |
  | v2 | 7 | 11.72 | promoted over v1 |
  | v3 | 42 (same as v1) | 12.68 | **rejected** vs v2 (limit 12.30) |

  **Changing only the seed moves cost by about 8%, more than the 5% tolerance**, so
  promotions partly depend on seed luck.

  **Later correction (D-26, D-28):** the cause was not model randomness. LightGBM with
  these settings gives the same model for every seed. The seed only changed which 16
  engines calibrated the interval and chose the threshold. Averaging seeds had no
  effect (D-28). Calibrating on all engines by cross-validation cut the cost spread
  from ±1.90 to ±0.20 (D-26), which is now well inside the 5% tolerance.
- **Evidence:** `run_log.md` section 10.
- **Related:** D-11, D-13, D-20.

## D-15: Require an engine ID; 24h windows; skip under 5 engines
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Without an engine ID, the drift check can't count engines (D-8), and
  outcomes can't be matched to predictions (D-19).
- **Options considered:**
  1. Make `asset_id` optional.
     *Pro:* old clients keep working. *Con:* the engine minimum and labels silently
     don't work.
  2. Require `asset_id` by default, with a config switch.
     *Pro:* the checks are guaranteed to work. *Con:* breaks old clients (422 errors).
  3. Infer engines from feature patterns.
     *Pro:* no client change. *Con:* unreliable guesswork.
- **Decision:** Option 2 (`serving.yaml input_validation.require_asset_id: true`), plus
  optional `cycle` and `observed_at`.
  - The inference log is upgraded in place.
  - The drift job reads the last 24 hours (max 500 rows) and skips when fewer than 5
    engines are present.
  - The `DriftCheckSkipped` alert fires after 6 hours of skipping.
- **Factors that led to it:** D-8's measurements.
- **Trade-offs accepted:** **This breaks the API.** Small fleets that never reach 5
  engines a day get no drift monitoring (made visible by the alert).
- **Expected effect:** No drift decisions from windows that are too small.
- **Actual effect:** Covered by tests (`tests/unit/test_drift_actions.py`,
  `test_serving_api.py`). On a real 20-engine window the check ran normally (D-17).
- **Evidence:** commit `16127ec`.
- **Related:** D-8, D-19.

## D-16: Validate inputs; hold drift retrains when a sensor looks broken
- **Date:** 2026-09-30
- **Status:** Accepted (hold condition superseded by D-21)
- **Context:** Bad inputs reached the model silently, and D-9 showed broken sensors
  could trigger retrains.
- **Options considered:**
  1. Reject anything outside the training range.
     *Pro:* strict. *Con:* throws away exactly the unusual readings the drift checks
     need.
  2. Reject missing or NaN/infinite values; accept out-of-range values but flag them.
     *Pro:* keeps real anomalies visible. *Con:* flagged predictions still get served.
  3. Accept everything.
     *Pro:* nothing breaks. *Con:* garbage in, garbage out.
- **Decision:** Option 2.
  - The 422 errors are counted in `pdm_input_rejected_total`; out-of-range values
    return `input_warnings` and are counted in `pdm_input_out_of_range_total`.
  - A retrain is held when the sensor check fires in the same drift check
    (`RetrainHeldForSensorFault`).
- **Factors that led to it:** Keep anomalies observable; don't retrain on bad data.
- **Trade-offs accepted:** A held retrain waits for a person.
- **Expected effect:** Clean rejections; no retrains from broken sensors.
- **Actual effect:**
  - Validation works as tested.
  - **But the hold fired wrongly on a real fleet shift:** FD003 flagged all 14 sensors,
    so its retrain was held (found in D-20). The hold now requires a *diagnosed* sensor
    fault (D-21).
- **Evidence:** commit `16127ec`, `reports/label_loop_fd003.json` (first run in
  `run_log.md` section 13).
- **Related:** D-9, D-20, D-21.

## D-17: Drift reference ships with the model in MLflow
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** The drift job read a hand-seeded `reference.parquet` that nothing kept in
  step with Production, and that didn't exist inside the cluster at all.
- **Options considered:**
  1. Bake the file into the drift image.
     *Pro:* simple. *Con:* goes stale with every retrain.
  2. Mount it from a volume.
     *Pro:* updatable. *Con:* still a separate step that can fall out of step.
  3. Log it with each training run; the drift job downloads the Production model's
     copy.
     *Pro:* always matches the model being monitored. *Con:* the job depends on MLflow
     being up.
- **Decision:** Option 3 (`src/pdm/drift/reference.py`, `reference.source: model`). A
  missing artifact is an error; there's no silent fallback to a file.
- **Factors that led to it:** Correctness; the drift job already had
  `MLFLOW_TRACKING_URI`.
- **Trade-offs accepted:** A few seconds more training (5 extra fits); the drift job
  fails if MLflow is down.
- **Expected effect:** Reference and model always match; the job works in the cluster.
- **Actual effect:**
  - Real run: loaded "cmapss_rul v1", saw 20 engines, drift 0.0, action `none`.
  - Peak memory was 377 MiB against a 512 MiB limit, so the limit was raised to
    768 MiB.
- **Evidence:** `run_log.md` section 12, commit `ef2fa62`.
- **Related:** D-4, D-10.

## D-18: A runbook entry for every alert
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** The new alerts had no instructions; alerts nobody knows how to act on get
  ignored.
- **Options considered:**
  1. Put the instructions in the alert text.
     *Pro:* visible where the alert fires. *Con:* too short for real steps.
  2. A runbook section per alert, linked from the alert.
     *Pro:* room for steps. *Con:* must be kept in sync.
- **Decision:** Option 2 (`RUNBOOK.md` "Alerts", "Recording outcomes").
- **Factors that led to it:** Operability.
- **Trade-offs accepted:** Sync burden.
- **Expected effect:** On-call can act without reading code.
- **Actual effect:** Not yet measured. Nobody has been on call for it yet.
- **Evidence:** `RUNBOOK.md`.
- **Related:** D-15, D-16, D-21.

## D-19: New labels from recorded outcomes, including maintained engines
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** The retrain job trained on the same NASA files every time, so a
  drift-triggered retrain rebuilt the same model.
- **Options considered:**
  1. Keep retraining on base data.
     *Pro:* no work. *Con:* can't adapt; drift detection is pointless.
  2. Use failures only.
     *Pro:* exact labels. *Con:* most real engines get maintained before failing, which
     throws away most of the data.
  3. Use failures, plus maintained engines where the label is still exact because of
     the 125 cap.
     *Pro:* uses more data without guessing. *Con:* readings within 125 cycles of a
     maintenance are still dropped.
  4. Estimate the unknown remaining life of maintained engines (survival modelling).
     *Pro:* uses everything. *Con:* much more complex, and adds assumptions.
- **Decision:** Option 3.
  - `src/pdm/labels/outcomes.py` stores events and imports CMMS (maintenance-system)
    CSV exports.
  - `src/pdm/labels/build.py` matches events to logged readings by `asset_id` and
    `cycle`.
  - Retraining rebuilds the labels, and each model logs a data manifest (raw and label
    file hashes).
- **Factors that led to it:** Exact labels only; simplicity over survival modelling.
- **Trade-offs accepted:**
  - Someone must record outcomes.
  - Logged readings hold only the served features, so a feature change orphans old
    readings.
  - SQLite read over a shared volume while serving writes to it is unreliable on
    network filesystems; a real database is needed at scale.
- **Expected effect:** The model can learn from new fleets.
- **Actual effect** (FD003 replay, 12,365 readings): 10,584 labelled (9,899 from
  failures, 685 from maintained engines at the cap), 1,781 dropped as too close to a
  maintenance. The effect on quality is D-20.
- **Evidence:** `reports/label_loop_fd003.json`, `tests/unit/test_labels.py`.
- **Related:** D-15, D-20.

## D-20: Prove the loop adapts on a real new fleet (FD003 replay)
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** Nothing had shown that the loop actually improves a model on new data.
- **Options considered:**
  1. Unit tests only.
     *Pro:* fast. *Con:* proves the plumbing, not the value.
  2. Replay a real fleet with a different fault mode (FD003) through the real API,
     record outcomes, retrain, and score on FD003 engines never replayed.
     *Pro:* real evidence. *Con:* about 15–25 minutes per run.
- **Decision:** Option 2 (`scripts/label_loop_demo.py`): 60 engines replayed (about 50%
  fail, 30% maintained, 20% still running); 40 kept aside for scoring.
- **Factors that led to it:** "Retrain on new data" had to be shown to work, not
  assumed.
- **Trade-offs accepted:** Runtime. One seed, one fleet.
- **Expected effect:** Fewer failures on FD003 after retraining, and no loss on FD001.
- **Actual effect** (first run, 40 unseen FD003 engines):

  | | Before | After |
  |---|---|---|
  | Unplanned failures | 15 of 40 | 0 of 40 |
  | Cost per engine | 45.2 | 11.8 |
  | RMSE, last 60 cycles | 29.0 | 21.4 |

  The gate confirmed no loss on FD001 (cost 12.2 vs 12.7).

  **Also found:** the drift check held the retrain as a "sensor fault", which led to
  D-21. The demo only retrained because the script calls retrain directly. After D-21,
  the re-run's drift check chose `retrain` by itself, with identical before/after
  numbers.

  Caveat: one seed and one fleet. The "after" model had seen 48 FD003 engine lives, so
  this shows adaptation to a fleet it now has labels for, not generalisation to unseen
  fault modes.
- **Evidence:** `reports/label_loop_fd003.json`, `run_log.md` section 13.
- **Related:** D-14, D-19, D-21.

## D-21: Tell a broken sensor from a fleet-wide shift
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** On FD003, all 14 sensors were flagged, so the retrain was held (D-16).
  Counting flags can't separate the two cases: a 2 std offset on *one* sensor flagged
  6–14 sensors, and FD003 flagged 13–14.
- **Options considered:**
  1. A count cutoff ("fault only if ≤ N sensors flagged").
     *Pro:* simple. *Con:* the ranges overlap, so no cutoff works.
  2. Remove the top-scoring sensor, re-score the rest, and repeat up to
     `max_culprits` (2). If that explains every flag, it's a fault; otherwise it's a
     fleet-wide shift.
     *Pro:* clean separation in tests. *Con:* extra model fits per check; 3+ broken
     sensors look like a shift.
  3. Skip the hold when drift share is high.
     *Pro:* trivial. *Con:* a large single-sensor fault also raises drift share.
- **Decision:** Option 2 (`diagnose` in `src/pdm/drift/sensor_check.py`). Only a
  `sensor_fault` verdict names culprits, raises the alert, or holds a retrain.
- **Factors that led to it:** Test evidence (below).
- **Trade-offs accepted:** Faults on 3+ sensors at once trigger a retrain instead of an
  alert, and then the gate is the only protection.
- **Expected effect:** FD003-style shifts retrain automatically; single faults still get
  caught and held.
- **Actual effect** (FD001 holdout windows):

  | Case | Result |
  |---|---|
  | One sensor offset 0.5–3 std, or stuck | 0 sensors left flagged after removing the top one; right sensor named 30 of 30 |
  | Two sensors offset | both named 29 of 30 |
  | FD003 fleet | "system-wide shift" 30 of 30 (12–13 left flagged) |
  | Normal traffic | "ok" 30 of 30 |

  **Full FD003 loop re-run with this fix** (`reports/label_loop_fd003.json`): the drift
  check gave drift share 0.86, verdict `system_wide_shift`, no faulty sensors, and
  action **`retrain`**. Before the fix it was `hold_for_sensor_fault`. Every other
  number was identical to the first run (the run is fully seeded).
- **Evidence:** `tests/unit/test_drift_checks.py`,
  `tests/integration/test_end_to_end_loop.py`, `run_log.md` section 13.
- **Related:** D-5, D-16, D-20.

## D-22: End-to-end test of the whole loop without Docker
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** No Docker was available here, so the kind cluster couldn't run, and the
  chain had only been tested in pieces.
- **Options considered:**
  1. Wait for a Docker environment.
     *Pro:* tests the real cluster. *Con:* nothing tested meanwhile.
  2. An in-process test of the production code on a small synthetic fleet.
     *Pro:* runs anywhere in about a minute; tests every stage and decision.
     *Con:* doesn't test Kubernetes manifests or Argo.
  3. Use real NASA data in the test.
     *Pro:* realistic. *Con:* too slow, and the data isn't committed.
- **Decision:** Option 2 (`tests/integration/test_end_to_end_loop.py`,
  `tests/synthetic_cmapss.py`). It covers:
  - train, human-confirmed first promotion;
  - serve, then drift check (`none` for the same fleet, `retrain` for a new one);
  - CSV outcome import, labels, retrain plus gate;
  - rollback;
  - a broken sensor gets named and holds the retrain.
- **Factors that led to it:** An environment limit (no Docker).
- **Trade-offs accepted:** Kubernetes stays untested; synthetic data checks plumbing,
  not quality.
- **Expected effect:** Wiring breaks show up in CI.
- **Actual effect:** 2 tests pass in about 74 seconds. They caught one wiring problem
  while being written: serving loads the model by the name in `serving.yaml`.
- **Evidence:** `tests/integration/test_end_to_end_loop.py`.
- **Related:** D-19, D-21.

## D-23: Move docs to `/docs`; split run log from how-it-runs doc
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** The user asked for `docs/decisions.md` and `docs/execution.md` in a fixed
  format. The existing root `executions.md` was a log of runs and measured numbers,
  which is a different document from "how the code runs".
- **Options considered:**
  1. Rename `executions.md` to `docs/execution.md` and rewrite it.
     *Pro:* one file. *Con:* loses the run evidence the decisions cite.
  2. Move the run log to `docs/run_log.md` (unchanged) and write `docs/execution.md`
     new.
     *Pro:* keeps the evidence; each file has one job. *Con:* three documents instead
     of two.
- **Decision:** Option 2, using `git mv` to keep history. Old entries D1–D22 were
  rewritten into the required format with the same numbers. Where they had recorded
  only one option, the alternatives that were actually considered were added.
- **Factors that led to it:** "Never delete an entry"; evidence must stay traceable.
- **Trade-offs accepted:** One extra file.
- **Expected effect:** Anyone can find why (decisions), how (execution), and proof (run
  log).
- **Actual effect:** Done in this change.
- **Evidence:** `docs/`.
- **Related:** D-24.

## D-24: Static call graph with pyan3 + Graphviz
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:** `docs/execution.md` needs an auto-generated function call graph.
- **Options considered:**
  1. pyan3: static analysis, reading the code without running it.
     *Pro:* fast; covers all code, even paths no test runs. *Con:* approximate; misses
     calls through variables such as `adapter["load"](...)`.
  2. pycallgraph: records calls while the code runs.
     *Pro:* exact for what ran. *Con:* only shows paths that ran; needs a full training
     run; the project is unmaintained.
- **Decision:** Option 1 (`scripts/generate_call_graph.py`). It writes `.dot`, `.svg`
  and `.txt` files to `docs/call_graph/`, with absolute paths stripped. Hand-written
  call chains in `execution.md` cover what pyan3 misses.
- **Factors that led to it:** Completeness and speed; no need to run training.
- **Trade-offs accepted:** Approximate edges.
- **Expected effect:** Graphs regenerate in seconds.
- **Actual effect:** Four graphs (full, retrain, drift, serving) generate in a few
  seconds.
  - The first attempt put absolute machine paths in tooltips, so the script now strips
    them.
  - After more modules were added, Graphviz 2.43 crashed on the full graph ("trouble in
    init_rank") with pyan3's `--grouped` layout. `-Gnewrank=true` didn't help;
    `--nested-groups` renders fine, so the script uses that.
- **Evidence:** `docs/call_graph/`.
- **Related:** D-23.

## D-25: One OpenMP thread per serving process
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:**
  - Requests took about 72 ms in the FD003 replay, against about 17 ms on an idle
    machine.
  - LightGBM and scikit-learn use OpenMP, a library that splits work across threads.
    By default it starts a thread per CPU core on every call.
  - Serving scores one row per request, so there's nothing worth splitting.
- **Options considered:**
  1. Leave the default.
     *Pro:* nothing to change. *Con:* see the measurements below.
  2. Limit threads at runtime with `threadpoolctl`, once at startup.
     *Pro:* configurable in YAML. *Con:* **didn't work.** OpenMP limits apply only to
     the thread that sets them, and FastAPI runs requests on other worker threads.
     Measured 19.3 ms, no better. Entering the limit on every call instead costs about
     5 ms itself (14.0 vs 17.8 ms).
  3. `OMP_NUM_THREADS=1` in the serving image.
     *Pro:* process-wide from start; one line. *Con:* applies to every OpenMP call in
     serving, which is fine for one-row requests.
- **Decision:** Option 3 (`docker/serve.Dockerfile`). Only the serving image: training
  and drift jobs process large batches, where threads help.
- **Factors that led to it:** The measurements below; option 2 failed its own test.
- **Trade-offs accepted:** If serving ever scores big batches, it won't parallelise
  them.
- **Expected effect:** Slightly faster when idle; much steadier under load.
- **Actual effect** (full `POST /predict`, 4-CPU machine):

  | | Idle | All 4 cores busy |
  |---|---|---|
  | Default threads | 17.4–18.3 ms | **630–1,359 ms** |
  | `OMP_NUM_THREADS=1` | 16.6–17.3 ms | **30–32 ms** |

  Under load the default would trip the 1-second `ServingHighLatencyP95` alert. Not
  yet measured inside the real container or cluster (no Docker here).
- **Evidence:** measurements in `run_log.md` section 14.
- **Related:** D-12, D-20.

## D-26: Cross-validated calibration of interval and threshold
- **Date:** 2026-09-30
- **Status:** Accepted
- **Context:**
  - Holdout cost varied ±1.90 per engine across seeds (D-14): more than the gate's 5%
    tolerance, so promotions were partly luck.
  - The experiment showed the seed only changed *which* 16 validation engines set the
    interval and the maintenance threshold.
- **Options considered:**
  1. Widen the gate tolerance.
     *Pro:* one number. *Con:* the gate stops catching real regressions.
  2. Average several seeds (D-28).
     *Pro:* standard for noisy models. *Con:* **no effect**: the models are identical
     across seeds.
  3. Cross-validation: every training engine is predicted by models that never saw it
     (5 engine-grouped folds). Those predictions set the interval and threshold; the
     final models train on all engines.
     *Pro:* uses all 80 engines instead of 16. *Con:* 6 fits instead of 1.
- **Decision:** Option 3 (`model.calibration: cross_validation`, `cv_folds: 5`,
  `_fit_bundle_cv` in `src/pdm/training/train.py`). The old split is still available
  with `calibration: split`.
- **Factors that led to it:** The measurements below; no API change.
- **Trade-offs accepted:**
  - Training takes 30 s instead of about 15 s.
  - Rows from one engine are correlated, so the conformal coverage guarantee is
    approximate. Measured coverage: 86–88%.
- **Expected effect:** A smaller cost spread; slightly better RMSE, because the final
  model sees 25% more engines.
- **Actual effect** (5 seeds each, `reports/model_quality_cv.json`):

  | | Split (before) | Cross-validation (now) |
  |---|---|---|
  | Holdout cost per engine | 12.99 ± 1.90 | **12.38 ± 0.20** |
  | Holdout unplanned failures (average) | 0.2 | **0.0** |
  | NASA test RMSE | 22.28 ± 0.27 | **21.40 ± 0.00** |
  | Interval coverage / width | 89% / 58.9 | 88% / 57.0 |

  A real `pdm.training.train` run with the new config took 30 s. It scored test RMSE
  21.40, coverage 87%, holdout cost 12.21, 0 failures, threshold 12.
- **Evidence:** `reports/model_quality_cv.json`,
  `scripts/experiments/model_quality.py`, `tests/unit/test_cv_calibration.py`,
  `run_log.md` section 15.
- **Related:** D-12, D-13, D-14, D-28.

## D-27: Trend features (5- minus 20-cycle mean)
- **Date:** 2026-09-30
- **Status:** Proposed (not adopted)
- **Context:** The model is mediocre (RMSE about 21–22 against published 12–18, D-1).
  One cheap extra signal is how fast each sensor is changing.
- **Options considered:**
  1. Keep the current 17 features.
     *Pro:* no API change. *Con:* weaker RMSE.
  2. Add 14 trend features (5-cycle mean minus 20-cycle mean, per sensor).
     *Pro:* big RMSE gain. *Con:* every client must compute and send 14 more
     features; drift checks must be recalibrated; old logged readings can't become
     labels (D-19).
- **Decision:** Not adopted yet. The evidence disagrees with itself, and the change is
  expensive.
- **Factors that led to it:**

  | | Cross-validation, current features | Cross-validation, + trend |
  |---|---|---|
  | NASA test RMSE | 21.40 | **17.75** |
  | Interval width | 57.0 | 48.9 |
  | Validation cost (training data) | 11.97 ± 0.20 | **10.84 ± 0.05** |
  | **Holdout** cost | **12.38 ± 0.20** | 14.72 ± 2.01 |
  | **Holdout** unplanned failures | **0.0** | 0.8 |

  Better accuracy on average, but on the 20 frozen holdout engines it misses about one
  failure. It chooses lower thresholds (8–11 instead of 12–15), leaving less margin.
  20 engines are too few to say whether that's real or luck, and the gate would
  reject it today on failures.
- **Trade-offs accepted:** Leaving a likely RMSE gain on the table.
- **Expected effect:** n/a until adopted.
- **Actual effect:** Not yet measured in production.
- **Next step:** Measure decision cost with cross-validation across all 100 FD001
  engines, plus FD003, before deciding. If adopted, it needs a versioned API.
- **Evidence:** `reports/model_quality.json`, `reports/model_quality_cv.json`.
- **Related:** D-1, D-19, D-26.

## D-28: Hyperparameter tuning and seed averaging
- **Date:** 2026-09-30
- **Status:** Rejected
- **Context:** Cheap model-quality ideas, tested with the same 5 seeds as everything
  else.
- **Options considered:**
  1. Three tuned settings (`tuned_a/b/c`), with the winner chosen on internal
     validation RMSE (the test set is never used to choose).
     *Pro:* standard. *Con:* see below.
  2. Averaging 5 seeds of the point model.
     *Pro:* reduces model variance. *Con:* see below.
  3. Keep the defaults.
- **Decision:** Option 3. `ensemble_seeds` stays in the code, off by default: it
  matters if a future configuration makes LightGBM random.
- **Factors that led to it:**
  - **Seed averaging changed nothing** (identical to the baseline to every decimal
    place). Without row or column sampling, LightGBM is deterministic.
  - **Tuning:** validation RMSE picked `tuned_b`, which had the worst holdout cost
    (14.30, 0.4 failures), so RMSE is the wrong way to choose. `tuned_c` looked best
    on holdout cost with a split, but with cross-validation it matches the defaults
    (12.29 vs 12.38; test RMSE 21.39 vs 21.40) and is slower (11.8 vs 8.2 ms). Its
    split advantage was mostly the calibration lottery.
- **Trade-offs accepted:** None meaningful.
- **Expected effect:** n/a.
- **Actual effect:** n/a (not adopted).
- **Evidence:** `reports/model_quality.json`, `reports/model_quality_cv.json`.
- **Related:** D-14, D-26.
