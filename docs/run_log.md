# Run log

What was run, in order, and what came out. Use it to reproduce any number in the README
or [`decisions.md`](decisions.md). Commands run from the repo root on Linux, Python 3.11.

## 0. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt -r requirements.txt   # dev pulls train + drift
pip install -e .
```

- Package versions used: mlflow 2.22.5, lightgbm 4.5.0, evidently 0.4.40, pandas 2.3.3,
  numpy 1.26.4, scikit-learn 1.9.1, SQLAlchemy 2.0.x.
- Before the pin in step 7, the install pulled SQLAlchemy 2.1.1 and `import mlflow`
  crashed (`ImportError: FallbackAsyncAdaptedQueuePool`). See [`decisions.md`](decisions.md) D-7.

## 1. Get the real data

```bash
curl -o cmapss.zip "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
unzip cmapss.zip
unzip "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -d cmapss
cp cmapss/*FD00*.txt data/raw/
python -m pdm.data.download --subsets FD001 FD002 FD003 FD004   # checks presence
```

`data/raw/` is git-ignored; the data isn't committed.

## 2. Train with the repo's own pipeline (FD001)

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  python -m pdm.training.train --raw-dir data/raw --config-name training.yaml --json-out fd001_train.json
```

- Result: `val_rmse 18.90`, gate (`max_rmse 35`) passed, registered as `cmapss_rul` v1.
- Loading that model and calling it with `config/serving.yaml`'s columns failed MLflow's
  schema check (`_roll_mean_10` vs `_roll_mean_20`). Fixed in commit `a4fffd4` (D-2).
  After the fix, the same call returned a prediction.

## 3. Benchmark the model and the drift check

```bash
python scripts/benchmark_real_data.py --raw-dir data/raw    # ~5 min
```

Writes `reports/benchmark_cmapss.json`. Model results (5 seeds, official test set):

| Subset | Test RMSE | MAE | NASA score | Baseline RMSE |
|---|---|---|---|---|
| FD001 | 22.3 ± 0.8 | 16.0 | 3,654 | 42.0 |
| FD002 | 21.4 ± 0.2 | 16.5 | 5,576 | 45.0 |
| FD003 | 21.7 ± 0.7 | 15.9 | 2,132 | 43.6 |
| FD004 | 24.0 ± 0.5 | 18.7 | 6,860 | 45.6 |

- First run: the original drift check flagged 90-100% of normal unseen-engine windows.
- Control: windows drawn from the reference engines themselves scored 0%, which ruled
  out a flaw in the test setup.

## 4. First calibration: global threshold (commit `a864e9a`)

```bash
python scripts/calibrate_drift.py --raw-dir data/raw
```

- The script at that commit set a per-column threshold of 0.47.
- False alarms on 40 unseen engines: 0% (large fleet), 4.5% (5-engine fleet).
- Still left: ageing fleets tripped retraining 100%, and single broken sensors were
  never caught. These led to D-4 and D-5.

## 5. Compare reference strategies and sensor checks

```bash
python scripts/experiments/compare_drift_methods.py    # ~25 min
```

Writes `reports/drift_method_comparison.json`. Compared:

- **References:** full run-to-failure, mid-life, and life-stage matched.
- **Per-sensor checks:** plain distance vs residual.

Outcome:
- Life-stage matched was the only reference with no ageing false alarms that still
  caught FD003.
- The residual check caught one broken sensor at 0.5 std and a stuck sensor 100% of the
  time.
- Tables in [`decisions.md`](decisions.md) D-4 and D-5.

## 6. Build the two checks and re-calibrate

- Code: `src/pdm/drift/life_stage.py`, `src/pdm/drift/sensor_check.py`, and
  `evaluate_window` in `src/pdm/drift/run_drift_check.py`.
- `scripts/seed_reference_data.py` now adds out-of-fold `predicted_rul`.
- Config: `config/drift.yaml`. Alert: `SensorFaultSuspected` in
  `monitoring/prometheus/alerts.yaml`.

```bash
python scripts/calibrate_drift.py --raw-dir data/raw    # ~20 min
```

Writes `reports/drift_calibration.json`. Thresholds: `drift.stattest_threshold 0.32`,
`sensor_check.threshold 0.46`. On 40 unseen engines, 120 windows per scenario:

| Scenario | Retrain | Sensor alert |
|---|---|---|
| Normal, large fleet | 0% | 0% |
| Normal, 5-engine fleet | 0.8% | 3.3% |
| Young fleet / NASA test set / old fleet | 0% | 0% |
| FD002 | 100% | 100% |
| FD003 | 97% | 100% |
| One sensor offset 0.5 / 1 / 2 std | 0% / 7% / 9% | 100% (right sensor named first: 100%) |
| One sensor stuck | 5% | 100% (right sensor named first: 100%) |

Cross-check through the production Evidently path: a normal fleet triggered retraining
0% of the time, FD003 100%.

## 7. Pin SQLAlchemy and check a clean install

```bash
python -m venv /tmp/venv-pin
/tmp/venv-pin/bin/pip install -r requirements.txt       # resolved SQLAlchemy 2.0.54
MLFLOW_TRACKING_URI=sqlite:////tmp/venv-pin/t.db /tmp/venv-pin/bin/python -c \
  "import mlflow; mlflow.set_experiment('x'); mlflow.start_run(); mlflow.log_metric('m', 1)"
```

Result: the run logged without errors.

## 8. Re-run the benchmark with the final config

```bash
python scripts/benchmark_real_data.py --raw-dir data/raw
```

- The first re-run showed 10% false retrains and 25% false sensor alerts on the "5-engine"
  window.
- Cause: that window took the last 500 rows sorted by engine, so it really held only 2-3
  engines.
- Fix: the benchmark now draws from exactly k engines. The window-size sweep that
  followed is in [`decisions.md`](decisions.md) D-8 (needs about 5+ engines per window).

Final drift rows in `reports/benchmark_cmapss.json` (original check → final check):

| Scenario | Retrain before | Retrain now | Sensor alert now |
|---|---|---|---|
| 20 unseen FD001 engines | 90% | 0% | 0% |
| 1 / 2 / 3 / 5 unseen engines | 100% / 100% / 100% / 85% | 60% / 15% / 5% / 0% | 85% / 60% / 15% / 5% |
| NASA test set | 100% | 0% | 0% |
| Near failure (RUL ≤ 30) | 100% | 0% | 5% |
| FD003 / FD002 | 100% / 100% | 100% / 100% | 100% / 100% |
| One sensor offset 1 std | 90% | 5% | 100% |

## 9. Tests and lint

```bash
ruff check src scripts tests
black --check src scripts tests
python -m pytest -q
```

- New tests: `tests/unit/test_config_consistency.py`, `tests/unit/test_drift_checks.py`,
  and an added case in `tests/unit/test_drift.py`.
- One test fails before and after these changes (Linux path issue, [`decisions.md`](decisions.md) D-10):
  `tests/unit/test_config.py::test_set_experiment_with_artifact_root_none_uses_mlflow_default`.

## 10. Phase 2A: intervals, decision cost, champion/challenger gate

```bash
python scripts/build_cmapss_holdout.py --subset FD001 --n-units 20 --version v1
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m pdm.training.train --raw-dir data/raw
python -m pdm.evaluation.champion_challenger --candidate-version 1 --promote --confirm-bootstrap
python -m pdm.training.retrain --raw-dir data/raw        # train + gate in one step
```

| Model | Seed | NASA test RMSE | Interval coverage / width | Holdout cost per engine | Holdout failures | Threshold | Gate result |
|---|---|---|---|---|---|---|---|
| v1 | 42 | 22.7 | 91% / 65 | 12.68 | 0 | 9 | promoted (human-confirmed first promotion) |
| v2 | 7 | 23.2 | 85% / 54 | 11.72 | 0 | 11 | promoted over v1 |
| v3 | 42 | 22.7 | 91% / 65 | 12.68 | 0 | 9 | **rejected** vs v2 (cost 12.68 > limit 12.30) |

- The first interval attempt used LightGBM quantile models. The upper bound came out as
  exactly 125 on every row (D-12), so the bounds were switched to scikit-learn.
- Tests: 110 passed, plus the one failure that predates these changes.

## 11. Phase 2B: engine IDs, input validation, drift-window rules

No new measurements. The window-size numbers behind `min_engines: 5` are in section 8
and D-8. Checked by tests:

- `tests/unit/test_serving_api.py`: 422 without `asset_id` and for NaN values;
  out-of-range values flagged but still predicted; `asset_id`, `cycle`, `observed_at`
  and model version reach the log.
- `tests/unit/test_inference_log.py`: an old-schema database is upgraded in place, and
  its old rows keep working.
- `tests/unit/test_drift_actions.py`: skip under 5 engines, hold on sensor fault,
  sensor alert alone never retrains, shadow rows dropped, time window.
- Suite: 128 passed, plus the one failure that predates these changes.

## 12. Phase 2C: reference ships with the model; runbook

```bash
python -m pdm.training.train --raw-dir data/raw          # also logs drift_reference/reference.parquet
python -m pdm.evaluation.champion_challenger --candidate-version 1 --promote --confirm-bootstrap
# 500 rows from the 20 holdout engines written to the inference log, then:
INFERENCE_LOG_DB=... python -m pdm.drift.run_drift_check --dry-run
```

Output: `Drift reference: cmapss_rul v1 (run ...) (16679 rows)`, then
`action=none engines=20 share_of_drifted_columns=0.0`. Peak memory 377 MiB.

## 13. Phase 2D/E: outcome labels, the FD003 loop, fault-vs-shift diagnosis

```bash
# fresh store: train v1, promote it after human confirmation, then run the loop
export MLFLOW_TRACKING_URI=sqlite:///loop/mlflow.db INFERENCE_LOG_DB=loop/log.db \
       OUTCOME_DB=loop/outcomes.db LABELS_DIR=loop/labels
python -m pdm.training.train --raw-dir data/raw
python -m pdm.evaluation.champion_challenger --candidate-version 1 --promote --confirm-bootstrap
python scripts/label_loop_demo.py --raw-dir data/raw --json-out reports/label_loop_fd003.json
```

Replay: 13,167 requests through the real API (0 rejected), 60 FD003 engines (38 failed,
15 maintained, 7 still running). The replay takes about 70 ms per request, most of it
model-prediction overhead (D-12).

| Step | First run (before D-21) | Re-run (after D-21) |
|---|---|---|
| Drift check | share 0.86, **14 sensors flagged**, action `hold_for_sensor_fault` | share 0.86, verdict `system_wide_shift`, action **`retrain`** |
| Outcome import | 53 added | 53 added |
| Labels | 10,584 rows (9,899 failure, 685 capped-maintenance), 1,781 dropped as too close to maintenance | same |
| Gate (FD001 data) | approved: cost 12.24 vs 12.68, RMSE 22.30 vs 22.66, coverage 84% | same |

40 FD003 engines never replayed:

| | Before (v1) | After (v2) |
|---|---|---|
| Unplanned failures | 15 (7 missed, 8 late) | 0 |
| Cost per engine | 45.2 | 11.8 |
| RMSE, all cycles | 22.5 | 19.8 |
| RMSE, last 60 cycles | 29.0 | 21.4 |
| Wasted cycles per maintained engine | 23.0 | 18.4 |

Fault-vs-shift check (D-21), on FD001 holdout windows: removing the top sensor left 0
sensors flagged for every one-sensor fault (0.5-3 std offset, stuck), and 12-13 flagged
for FD003/FD002. Two-sensor faults were named correctly 29/30; FD003 was called a
system-wide shift 30/30; normal traffic "ok" 30/30.

End-to-end test (D-22): `pytest tests/integration/test_end_to_end_loop.py`, 2 passed in
about 74 s.

## 14. Phase 2F: serving latency

Single-row timings on a 4-CPU machine, after 5-10 warm-up calls. Commands are in
`decisions.md` D-25.

- Components, idle: LightGBM point model 0.6 ms; each scikit-learn quantile model
  2.6-3.4 ms; `predict_frame` 15 ms; pyfunc `predict` 17.3 ms (8.9 ms with a 1-thread
  limit in the calling thread); log insert 1.6 ms; `GET /healthz` 1.4 ms.
- Full `POST /predict` (300 calls idle, 40 calls under load; load = 4 busy-loop
  processes):

| Setting | Idle (2 runs) | Under load (2 runs) |
|---|---|---|
| default threads | 17.41, 18.31 ms | 629.95, 1358.99 ms |
| `OMP_NUM_THREADS=1` | 16.59, 17.25 ms | 32.41, 30.38 ms |
| threadpoolctl limit set once at startup | 19.26 ms (no effect) | not run |

## 15. Phase 2F: model quality

```bash
python scripts/experiments/model_quality.py --raw-dir data/raw          # pass 1 -> reports/model_quality.json
python scripts/experiments/model_quality.py --raw-dir data/raw \
  --variants baseline baseline_cv tuned_c tuned_c_cv trend trend_cv --out reports/model_quality_cv.json
```

Seeds 42, 0, 1, 2, 3. Values are mean ± std over seeds. Latency is single-row with one
thread.

| Variant | Val RMSE | Test RMSE | Coverage | Width | Holdout cost | Failures | Latency |
|---|---|---|---|---|---|---|---|
| baseline | 22.89 ± 1.55 | 22.28 ± 0.27 | 0.89 | 58.9 | 12.99 ± 1.90 | 0.2 | 7.8–8.4 ms |
| ensemble5 | identical to baseline | | | | | | 12.4 ms |
| tuned_a | 22.62 ± 1.57 | 22.04 ± 0.42 | 0.89 | 59.3 | 13.30 ± 1.70 | 0.2 | 10.9 ms |
| tuned_b | 22.30 ± 1.60 | 22.08 ± 0.48 | 0.88 | 59.8 | 14.30 ± 1.85 | 0.4 | 9.6 ms |
| tuned_c | 22.92 ± 1.66 | 22.27 ± 0.44 | 0.89 | 58.8 | 12.13 ± 0.41 | 0.0 | 12.2 ms |
| trend | 17.67 ± 1.13 | 18.11 ± 0.45 | 0.89 | 50.2 | 11.99 ± 1.99 | 0.2 | 8.0 ms |
| **baseline_cv** | 21.31 ± 0.44 | **21.40 ± 0.00** | 0.88 | 57.0 | **12.38 ± 0.20** | **0.0** | 8.2 ms |
| tuned_c_cv | 21.30 ± 0.45 | 21.39 ± 0.00 | 0.88 | 57.7 | 12.29 ± 0.17 | 0.0 | 11.8 ms |
| trend_cv | 16.52 ± 0.40 | 17.75 ± 0.00 | 0.86 | 48.9 | 14.72 ± 2.01 | 0.8 | 8.5 ms |

Validation (training-data-only) cost: baseline 11.73, baseline_cv 11.97, tuned_c_cv
11.77, trend 10.83, trend_cv 10.84. Validation RMSE picked tuned_b in pass 1 and
tuned_c in pass 2.

Adopted: `calibration: cross_validation` (D-26). A real training run then took 30 s:
test RMSE 21.40, coverage 87%, holdout cost 12.21, 0 failures, threshold 12. Suite: 144
passed, plus the one failure that predates these changes.

## 16. PR #2 CI fix

```bash
pytest -q tests/unit/test_config.py::test_set_experiment_with_artifact_root_none_uses_mlflow_default  # 1 failed (before)
ruff check src tests && black --check src tests                                                     # clean
pytest -q -m unit          # 129 passed
pytest -q -m integration   # 16 passed
pytest -q -m "not unit and not integration" --collect-only   # 0 tests (every test now runs in CI)
```

## 17. Trend model on all engines, and registered separately

```bash
python scripts/experiments/trend_all_engines.py --raw-dir data/raw   # -> reports/trend_all_engines.json
python -m pdm.training.train --config-name training.yaml             # registers cmapss_rul
python -m pdm.training.train --config-name training_trend.yaml       # registers cmapss_rul_trend
python -m pdm.evaluation.champion_challenger --candidate-version 1 --config-name training.yaml
python -m pdm.evaluation.champion_challenger --candidate-version 1 --config-name training_trend.yaml
```

All-engine evaluation: 100 engines × 2 repeats per dataset, every engine held out by
models that never saw it. Full table in `decisions.md` D-30.
- FD001: cost 12.37 → 11.06 (paired −1.30, 95% interval −1.58 to −1.04; 93/100 engines
  cheaper); failures 0 → 0.
- FD003: cost 13.13 → 11.97 (paired −1.16, 95% interval −3.69 to +1.52); failures 2 → 2.

Registration (fresh store; neither promoted, since a first promotion needs a human):
`cmapss_rul` v1 scored test RMSE 21.40, holdout cost 12.21, 0 failures.
`cmapss_rul_trend` v1 scored test RMSE 17.75, holdout cost 15.52, 1 failure.

Timing: one fold took **2,439 s** while a second training process competed for the 4
CPUs, and **49 s** once it was alone. That's the same thread thrashing as D-25: never
run two model-training jobs side by side on a small machine.
