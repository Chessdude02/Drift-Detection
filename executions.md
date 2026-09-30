# Executions

What was run, in order, and what came out. Use it to reproduce any number in the README
or `decisions.md`. Commands run from the repo root on Linux, Python 3.11.

## 0. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt -r requirements.txt   # dev pulls train + drift
pip install -e .
```

- Package versions used: mlflow 2.22.5, lightgbm 4.5.0, evidently 0.4.40, pandas 2.3.3,
  numpy 1.26.4, scikit-learn 1.9.1, SQLAlchemy 2.0.x.
- Before the pin in step 7, the install pulled SQLAlchemy 2.1.1 and `import mlflow`
  crashed (`ImportError: FallbackAsyncAdaptedQueuePool`). See `decisions.md` D7.

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
  schema check (`_roll_mean_10` vs `_roll_mean_20`). Fixed in commit `a4fffd4` (D2).
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
  never caught. These led to D4 and D5.

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
- Tables in `decisions.md` D4 and D5.

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
  followed is in `decisions.md` D8 (needs about 5+ engines per window).

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
- One test fails before and after these changes (Linux path issue, `decisions.md` D10):
  `tests/unit/test_config.py::test_set_experiment_with_artifact_root_none_uses_mlflow_default`.
