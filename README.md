# Predictive Maintenance MLOps

A reference implementation of an MLOps loop for predictive maintenance: train a
Remaining-Useful-Life (RUL) model on NASA's C-MAPSS turbofan dataset, serve it behind
FastAPI, and keep it healthy in production with automated CI/CD, staged rollouts with
rollback, drift-triggered retraining, shadow deployment, and Prometheus/Grafana
observability.

## Architecture

```
                         ┌────────────────────┐
   PR / push to main ──▶ │   GitHub Actions    │
                         │ ci / model-validation│
                         │  / build-and-push    │
                         └─────────┬───────────┘
                                   │ image pushed to GHCR
                                   ▼
                         ┌────────────────────┐        watches Prometheus
                         │  deploy.yaml (Helm) │◀───────────────────────────┐
                         └─────────┬───────────┘                           │
                                   ▼                                       │
   ┌───────────────────────────────────────────────────────┐              │
   │  kind cluster, namespace "pdm"                         │              │
   │                                                         │              │
   │   Argo Rollouts "pdm-serving" (canary 10%→50%→100%) ────┼──analysis───┘
   │        │ stable/canary Services       shadow Deployment │
   │        ▼                                    ▲           │
   │   Ingress (mirrors traffic to shadow) ───────┘           │
   │        │ /predict /healthz /readyz /metrics              │
   │        ▼                                                 │
   │   FastAPI pod ──▶ inference_log (PVC, SQLite)             │
   │        │                                                 │
   │        ▼ scraped by ServiceMonitor                       │
   │   kube-prometheus-stack (Prometheus + Grafana + AM)       │
   │        ▲                                                 │
   │        │ pushes drift score                              │
   │   pdm-drift-check CronJob ──Evidently──▶ reference vs.    │
   │        │  recent inference_log rows                      │
   │        └─ if drift > threshold: creates a Job from ──────▶ pdm-retrain CronJob
   │                                                             template, logs to MLflow
   └─────────────────────────────────────────────────────────┘
```

## Capability → files

| Capability | Where it lives |
|---|---|
| Training + MLflow tracking/registry | `src/pdm/training/train.py`, `config/training.yaml` |
| FastAPI serving + health/readiness | `src/pdm/serving/app.py` |
| Containerization | `docker/{serve,train,drift}.Dockerfile` |
| Staged deploy + rollback | `deploy/helm/pdm-serving/templates/{rollout,analysistemplate}.yaml`, `.github/workflows/deploy.yaml`, `scripts/rollback.ps1` |
| Shadow deployment | `deploy/helm/pdm-serving/templates/{rollout,ingress}.yaml` (shadow Deployment + Ingress mirror), `src/pdm/serving/app.py` (X-Shadow handling) |
| Drift-triggered retraining | `src/pdm/drift/run_drift_check.py`, `src/pdm/drift/trigger_retrain.py`, `deploy/k8s/drift-check-cronjob.yaml` |
| Prometheus/Grafana observability | `src/pdm/serving/metrics.py`, `deploy/helm/pdm-serving/templates/servicemonitor.yaml`, `monitoring/prometheus/*.yaml`, `deploy/grafana/**` |
| CI/CD | `.github/workflows/*.yaml` |

## Quickstart (local dev, Windows/PowerShell)

```powershell
./Makefile.ps1 venv
./Makefile.ps1 install
./Makefile.ps1 test              # runs fully offline against tests/fixtures/cmapss_sample.txt
```

### Train against the real dataset

```powershell
# See data/README.md for how to obtain the C-MAPSS files into data/raw/
python -m pdm.data.download --subsets FD001
./Makefile.ps1 train -- --raw-dir data/raw --config-name training.yaml
```

### Run the API locally

```powershell
./Makefile.ps1 serve
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

### Deploy to a local kind cluster

Prereqs: Docker Desktop, `kind`, `kubectl`, `helm`, and the `kubectl-argo-rollouts`
plugin on PATH.

```powershell
./Makefile.ps1 build-serve
./Makefile.ps1 build-train
./Makefile.ps1 build-drift
./Makefile.ps1 kind-up
# load the images into kind (or push to GHCR and reference that instead):
kind load docker-image pdm-serve:local --name pdm
helm upgrade --install pdm-serving deploy/helm/pdm-serving -n pdm `
  --set image.repository=pdm-serve --set image.tag=local
kubectl argo rollouts get rollout pdm-serving -n pdm --watch
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
```

### Rollback

```powershell
./scripts/rollback.ps1 abort   # cancel an in-progress canary
./scripts/rollback.ps1 undo    # revert to the previous stable revision
```

## Measured results (real NASA C-MAPSS data)

Reproduce with `python scripts/benchmark_real_data.py --raw-dir data/raw` (needs all four
subsets in `data/raw/`); raw numbers are in `reports/benchmark_cmapss.json`. Model =
the pipeline in `config/training.yaml` unchanged, mean ± std over 5 seeds.

**RUL model, scored on the official NASA test set** (last cycle of every test engine vs
`RUL_FD00x.txt`, true RUL capped at 125, same as training):

| Subset | Test engines | Test RMSE | Test MAE | NASA score (lower = better) | Baseline RMSE (predict mean) | Internal val RMSE |
|---|---|---|---|---|---|---|
| FD001 | 100 | 22.3 ± 0.8 | 16.0 | 3,654 | 42.0 | 21.3 ± 2.6 |
| FD002 | 259 | 21.4 ± 0.2 | 16.5 | 5,576 | 45.0 | 21.4 ± 1.0 |
| FD003 | 100 | 21.7 ± 0.7 | 15.9 | 2,132 | 43.6 | 19.6 ± 2.2 |
| FD004 | 248 | 24.0 ± 0.5 | 18.7 | 6,860 | 45.6 | 21.9 ± 1.7 |

- The model roughly halves the error of a naive baseline, but it is well short of
  published C-MAPSS results (roughly 12-18 RMSE on FD001). It passes the
  `max_rmse: 35.0` gate by a wide margin, so that gate does not separate a good model
  from a mediocre one.
- The single-split val RMSE that `train.py` logs (18.9 on FD001 with seed 42) is on the
  lucky side of its own ±2.6 spread; don't quote it as the model's accuracy.

**Drift detector** (`compute_drift_score`, `config/drift.yaml`: 500-row windows,
retrain when > 50% of columns drift; reference = 80 FD001 training engines; 20 windows
per scenario):

| Scenario | Real shift? | Mean share of drifted columns | Retrain triggered |
|---|---|---|---|
| Rows from the reference engines themselves | No | 0.02 | 0% |
| Random rows from 20 unseen FD001 engines | No | 0.80 | 90% |
| 5 unseen FD001 engines (production-like window) | No | 0.98 | 100% |
| FD001 official test set | No (earlier in life) | 1.00 | 100% |
| Unseen FD001 engines near failure (RUL ≤ 30) | Normal wear | 1.00 | 100% |
| FD003 (new fault mode) | Yes | 0.90 | 100% |
| FD002 (six operating conditions) | Yes | 1.00 | 100% |

- With the default Evidently test (normed Wasserstein distance, 0.1 per column), the
  detector flags almost any window of *new engines* as drift, because engines differ
  from each other more than that threshold allows. It cannot tell real drift from
  normal operation, so in production the retrain job would fire on nearly every check.
  The per-column test/threshold needs calibrating on unseen engines before the
  drift-triggered retraining loop means anything.

## Notes and known limitations

- **Shadow traffic** is mirrored via an NGINX Ingress annotation, which is inherently
  fire-and-forget (no mesh-grade guarantees). A service mesh (Istio/Linkerd + Flagger)
  would give stronger shadow-traffic semantics if this graduates beyond a reference repo.
- **Inference logging** uses SQLite on a PVC for simplicity. A production system with
  meaningful traffic would want a proper time-series/event store.
- **The `pdm-inference-log` PVC requests ReadWriteMany**, which kind's default
  `local-path` provisioner does not support; swap in an RWX StorageClass (e.g. NFS) for a
  real multi-node demo, or reduce it to one node for a single-writer/single-reader setup.
- `config/training.yaml`'s `evaluation.max_rmse: 35.0` is tuned for the real C-MAPSS
  FD001 dataset; the CI model-validation gate
  (`tests/integration/test_model_validation_cmapss.py`) uses a much looser threshold
  since it trains against the tiny 50-row fixture, not the real data.
- Tests live under `tests/unit/` (fast, isolated — no real pipeline run) and
  `tests/integration/` (real end-to-end pipeline runs against committed dataset
  samples); the whole suite runs in well under a minute. `config/training_bearing.yaml`
  and `src/pdm/data/{ims_bearing,bearing_features}.py` add a second supported dataset
  (NASA IMS Bearing vibration data) alongside C-MAPSS — see `data/README.md`.
