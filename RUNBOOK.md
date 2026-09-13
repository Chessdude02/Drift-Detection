# Runbook

Stub — to be completed in Phase 6. Operational procedures for the predictive-maintenance
CI/CD pipeline (training, staged promotion, rollback, drift response) will live here.

## Training data

`build-and-push-dev.yaml` trains against the real NASA IMS Bearing dataset hosted in
GCS — it does **not** fall back to the committed test fixture
(`tests/fixtures/ims_bearing_sample`, used only by `pr-checks.yaml`'s offline unit/
integration tests). A missing or misconfigured bucket fails that workflow loudly at its
"Require training data bucket configured" step rather than silently training on the
fixture.

**Source:** NASA IMS Bearing dataset, `2nd_test/` (4 channels, one accelerometer per
bearing; bearing1 fails). Exact download URL, full dataset structure, and why
`2nd_test` specifically (not `1st_test`/`3rd_test`) are in `data/README.md`'s "IMS
Bearing dataset" section — don't duplicate that detail here, this section only covers
where it lives in GCS and how to rotate it.

**Location and versioning convention:** upload `2nd_test/`'s snapshot files directly
under the version prefix — no nested `2nd_test/` subfolder, since
`training_bearing.yaml`'s `raw_dir` expects a flat directory of timestamp-named files:

```
gs://<BEARING_DATA_BUCKET>/raw_bearing/v1/2004.02.12.10.32.39   <- current real training data,
gs://<BEARING_DATA_BUCKET>/raw_bearing/v1/2004.02.12.10.42.39      flat, not nested
...
gs://<BEARING_DATA_BUCKET>/raw_bearing/v2/...                   <- a future replacement, once one exists
```

- `vars.BEARING_DATA_BUCKET` (repo variable) — the bucket name only, e.g. `my-org-pdm-data`.
- `vars.BEARING_DATA_VERSION` (repo variable, default `v1` if unset) — which version
  prefix to fetch.

**To rotate/update the data source:** upload the new snapshot files to a **new** version
prefix (e.g. `raw_bearing/v2/`) — never overwrite `v1/` in place, for the same reason
`holdout_v1.csv` is never overwritten (see `scripts/build_holdout.py`): a model trained
against a silently-changed dataset makes past runs' metrics incomparable. Once the new
prefix is uploaded and verified, bump `vars.BEARING_DATA_VERSION` to point at it. No
code or training-config change is needed — `training_bearing.yaml`'s `dataset.raw_dir`
stays `data/raw_bearing` regardless of which GCS version was fetched into it; the
version actually used is captured on every run via the `bearing_data_version` MLflow run
tag (`pdm.training.train`'s `--tag` mechanism).

**Access:** a dedicated GCP service account holds `storage.objectViewer` on
`BEARING_DATA_BUCKET` only. Its key is stored as the repo secret `secrets.GCP_SA_KEY`
and consumed by `google-github-actions/auth`.

## Rollback

The rollback workflow (`.github/workflows/rollback.yaml`) always reverts to the one
version tagged `rollback_production=true` in the MLflow model registry — it never walks
further back than that (see `src/pdm/evaluation/registry.py`). Promoting or rolling back
moves that tag by exactly one step, in either direction, so it always points at whichever
version Production is currently one step away from.

**If the rollback target is itself found to be bad** (i.e., rolling back didn't actually
fix the problem, or reveals the previous version had its own issue): this is **not**
automated. There is no "roll back twice" or "walk the full history" workflow. The fix is
a manual redeployment of a specific, known-good MLflow model version, chosen deliberately
by a human — transition that version to `Production` in the MLflow registry (e.g. via
`MlflowClient.transition_model_version_stage`) and re-run the K8s deploy step against its
tagged `image_tag`. Do not attempt to script an automatic multi-step rollback for this
case.

## Known gaps (GCP authentication)

`build-and-push-dev.yaml` authenticates to GCS with a long-lived service account key
(`secrets.GCP_SA_KEY`), not Workload Identity Federation (WIF). This is a deliberate,
temporary tradeoff, not an oversight: WIF was not already configured on this GCP
project, and setting it up (creating a Workload Identity Pool + Provider, binding it to
this repo) requires IAM permissions this account does not currently have (`permission
denied` when checking IAM setup access).

**Follow-ups, not yet scheduled:**
- Migrate to Workload Identity Federation once someone with sufficient IAM access on
  the project can set up the Workload Identity Pool + Provider. WIF removes the
  long-lived key entirely (short-lived tokens issued per-run instead), which is
  strictly better once available — this is not a permanent architectural choice.
- Until then, **rotate `secrets.GCP_SA_KEY` on a defined schedule — e.g. every 90
  days** (generate a new key for the same service account, update the repo secret,
  delete the old key version in GCP). No rotation schedule is currently enforced or
  automated; this is a manual process someone needs to own.

## Known gaps (failure-horizon calibration)

`config/evaluation.yaml`'s `failure.horizon_snapshots: 30` (and the F2=0.747/
precision=0.520/recall=0.839 it produces on `data/holdout/holdout_v1.csv`) was
calibrated by sweeping horizon values against **a single failure event**: the 3rd_test
holdout run has exactly one bearing (bearing3) failing once, giving `n_positive` as low
as 6-31 depending on the horizon tested. This is a small-sample point estimate, not a
statistically robust calibration — a different single run could plausibly have produced
a noticeably different "best" horizon, and there is no confidence interval or
cross-run variance behind this number.

**A more robust calibration is possible, not yet done:** the IMS Bearing dataset's
`1st_test/` run contains two additional, independent failure events (bearing3 inner
race failure, bearing4 rolling element failure) that could each contribute their own
horizon sweep, giving 3 failure events total instead of 1 to calibrate and validate
against. This isn't wired in because `1st_test/` uses an **8-channel format** (2
accelerometers per bearing: `b1x,b1y,b2x,b2y,b3x,b3y,b4x,b4y`) incompatible with the
current 4-channel `channels: [bearing1, bearing2, bearing3, bearing4]` config (see
`data/README.md`'s dataset structure table) — using it would need either a separate
8-channel feature/config path, or collapsing each bearing's x/y pair into one combined
channel before reusing the existing 4-channel pipeline. Flagging as a future path to a
statistically sturdier horizon calibration, not a current blocker.

## Known gaps (Helm/K8s layer)

This environment has no live Kubernetes cluster and no `kind`/real cluster access, so
the Helm chart (`deploy/helm/pdm-serving`) has only been validated statically:
`helm lint` (both prod-default and staging-representative values) and `helm template`
(rendered output inspected for correct namespaces, image tags, and Rollout/
AnalysisTemplate structure per environment). Both passed cleanly and one real bug was
found and fixed this way: the inference-log volume was an unconditional
`persistentVolumeClaim` referencing a PVC that only exists in the `pdm` namespace
(`deploy/k8s/pvc-inference-log.yaml`) — deploying to `pdm-staging` would have left the
pod stuck `Pending` forever. It's now `inferenceLog.persistent` (true = PVC, prod;
false = `emptyDir`, staging).

**What static rendering CANNOT verify — still genuinely unverified, not just untested:**

- **Actual rollout behavior.** Whether Argo Rollouts actually executes the canary steps
  (10% → pause → analysis → 50% → pause → analysis → 100%) correctly against a real
  Argo Rollouts controller, including weight-based traffic splitting via the nginx
  Ingress controller.
- **The Prometheus-gated AnalysisTemplate actually querying real metrics.** The
  `error-rate` and `p95-latency` queries assume `pdm_predictions_total` /
  `pdm_request_latency_seconds_bucket` are present in a real Prometheus at
  `http://prometheus-operated.monitoring.svc.cluster.local:9090` (the
  kube-prometheus-stack default service name) and that the `ServiceMonitor` resources
  are actually being scraped. None of this can be confirmed without a live cluster and
  real traffic.
- **Shadow traffic mirroring.** Whether the `nginx.ingress.kubernetes.io/mirror-target`
  annotation actually mirrors requests to the shadow Service as intended — this is an
  NGINX Ingress controller runtime behavior, not something a template render can check.
- **The `pdm-inference-log` PVC's ReadWriteMany access mode.** Whether the cluster's
  StorageClass actually supports RWX (kind's default `local-path` provisioner does
  not — see `deploy/k8s/pvc-inference-log.yaml`'s own comment). Only matters for `pdm`
  (prod); staging now uses `emptyDir` and doesn't depend on this.
- **MLflow reachability.** Whether `pdm-mlflow.pdm.svc.cluster.local:5000` is actually
  reachable from pods in `pdm-staging` (should be — cross-namespace service DNS is
  standard Kubernetes behavior — but never confirmed against a real cluster).
- **`kubectl argo rollouts status` as a staging health check.** Confirmed by reading
  Argo Rollouts' documented behavior (reports `Healthy` once pods pass their readiness
  probe) and by inspecting the rendered manifests, but never observed against a real
  rollout.

Treat the Helm/K8s layer as "internally consistent and carefully reasoned through," not
"proven correct" — the Python layer (training, evaluation, registry, promotion gate) has
68 real passing tests exercising actual behavior; this layer does not have an equivalent
yet. If a real kind cluster becomes available, re-running `scripts/setup_kind.ps1` plus
an actual `helm upgrade` and traffic test against each workflow would close this gap.
