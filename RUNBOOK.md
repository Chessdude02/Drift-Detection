# Runbook

Operational procedures for the predictive-maintenance
CI/CD pipeline: training, staged promotion, rollback, drift response, and what to do for
each alert (see "Alerts" at the end).

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

**Access:** a dedicated GCP service account (`pdm-ci-data-reader`) authenticates via
`secrets.GCP_SA_KEY` (`google-github-actions/auth`). It originally held only
`storage.objectViewer` on `BEARING_DATA_BUCKET` (read-only, matching its name) — that
is **no longer sufficient**. See "MLflow tracking persistence" below: the same
credential now also needs write access under that bucket's `mlflow/` prefix. Despite
the name, this SA is used for both purposes today; see that section for the cleaner
alternative (a second, write-scoped SA) that wasn't set up yet for expediency.

## MLflow tracking persistence (GCS-synced sqlite)

**The problem this fixes:** every workflow originally set `MLFLOW_TRACKING_URI` from
`secrets.MLFLOW_TRACKING_URI` — a secret that was never actually created on this repo.
`${{ secrets.MLFLOW_TRACKING_URI }}` on a nonexistent secret resolves to an **empty
string**, not "unset". MLflow treats an empty-string tracking URI as "use the local
`./mlruns` default" and raises nothing — so every CI job was silently writing its
training/scoring/promotion data to a directory inside its own ephemeral runner,
destroyed the instant the job finished. Worse than just "unrecoverable": every *job*
(not just every workflow run) gets its own fresh runner, so even `build-and-push-dev
.yaml`'s own `train` and `link-image-tag` jobs couldn't see each other's MLflow state.
The whole registry-tag-based promotion/rollback design requires one durable, shared
store - there wasn't one. Caught by actually triggering the workflow and reading real
logs, not by review.

**The fix:** `scripts/mlflow_gcs_sync.py` pulls the shared `mlflow.db` (sqlite) from
`gs://<BEARING_DATA_BUCKET>/mlflow/mlflow.db` at the start of every job that touches
MLflow, and pushes it back at the end. `MLFLOW_TRACKING_URI` is set directly to that
pulled file's local path (`sqlite:///$GITHUB_WORKSPACE/mlflow.db`) for the actual
business-logic step - never sourced from a secret in these 4 workflows, so the
empty-secret failure mode is structurally eliminated here, not just documented as a
risk. `src/pdm/common/config.py::configure_mlflow_env()` also now raises immediately if
`MLFLOW_TRACKING_URI` is ever present-but-empty, as defense-in-depth against this
recurring under a different secret name somewhere else later.

Model artifacts (not just metrics/params) are handled too: a bare sqlite-backed MLflow
client does **not** honor any environment variable for a default artifact root (verified
empirically - `MLFLOW_DEFAULT_ARTIFACT_ROOT` is a `mlflow server`-only flag with zero
effect on a direct client connection). `src/pdm/common/config.py::
set_experiment_with_artifact_root()` works around this by explicitly passing
`artifact_location=gs://<BEARING_DATA_BUCKET>/mlflow/mlartifacts/<experiment-name>`
when an experiment is created for the first time (only ever matters once per experiment
name - existing experiments keep whatever root they were first created with), so model
files land natively in GCS and never need separate syncing.

**Concurrency guards** (a basic guard was explicitly requested, not just documentation
of the risk): `mlflow_gcs_sync.py pull` acquires a lock object
(`mlflow/mlflow.db.lock`) before downloading - a second job trying to pull while a
fresh lock is held fails immediately with a clear error rather than racing. A lock
older than 30 minutes is assumed abandoned (a crashed job that never reached `push`)
and is taken over, loudly. `push` records the object generation `mlflow.db` had at
pull time and uploads with a GCS generation-match precondition - if another job pushed
in between, the precondition fails, the push is refused (never silently overwrites),
and the job exits non-zero with that job's writes explicitly called out as lost
(re-running pulls the newer state first). Tested against an in-memory fake GCS backend
implementing real precondition semantics (`tests/unit/test_mlflow_gcs_sync.py`), not
just reasoned about.

**Known limitation:** locking is a single global lock over the whole `mlflow.db`, not
per-experiment or per-run - two unrelated MLflow-touching jobs running at the same time
serialize against each other even if they'd never actually conflict. Acceptable for
this pipeline's actual usage pattern (occasional pushes to main, occasional manual
promotions), not necessarily if usage grows to frequent concurrent runs.

**Required IAM change:** the existing `pdm-ci-data-reader` service account's
`storage.objectViewer` role is **read-only** and does not permit the `push` half of
this sync (creating/overwriting `mlflow/mlflow.db` and `mlflow/mlflow.db.lock`, or
writing artifacts under `mlflow/mlartifacts/`). Before any of the 4 workflows can
complete successfully, grant that SA write access - either:
  - `roles/storage.objectAdmin` on the whole bucket (simplest, broadest), or
  - a custom role scoped to just the `mlflow/*` prefix via an IAM Condition (more
    least-privilege, more setup), or
  - a **second**, purpose-specific service account/key (`GCP_MLFLOW_SA_KEY`) used only
    by the "Authenticate to GCP" steps that precede an MLflow pull/push, keeping
    `pdm-ci-data-reader` genuinely read-only as its name implies. Not done yet, for
    expediency - worth doing if/when this graduates past a reference pipeline.

**Graduating beyond this:** this is the pragmatic fix given what's already provisioned,
not the final architecture. `deploy/k8s/mlflow-deployment.yaml` already exists for a
real hosted MLflow server (the standard answer), but needs a live K8s cluster (not
provisioned - see "known gaps (Helm/K8s layer)" below) and a concurrent-safe backend
database (Cloud SQL/Postgres - sqlite over a network filesystem isn't safe for
multiple concurrent writers, which is exactly the gap the lock file works around here).

## Operational readiness vs. "Production" stage

`stage=Production` in the MLflow registry means "passed the automated promotion gate" —
i.e. not a regression vs. the previous baseline (`src/pdm/evaluation/promotion.py`). It
does **not** mean "operationally ready" — the gate has no concept of an absolute quality
bar (e.g. RMSE at or below `failure_horizon`), so a known-weak model can legitimately sit
in Production.

When that's the case, `scripts/promote_to_production.py --readiness-note "..."` (wired to
`promote-staging-to-prod.yaml`'s `readiness_note` input) tags the promoted version
`operational_readiness=not_ready` plus the note itself
(`src/pdm/evaluation/registry.py::mark_not_operationally_ready`), directly on the model
version — not just in a config file's comments — so anyone reading the registry later
sees the caveat without having to separately check `config/training_bearing.yaml`. Leave
`readiness_note` blank for a promotion that does clear that bar.

**`ims_bearing_rul` v2 (promoted 2026-09-13):** tagged
`operational_readiness=not_ready` — val_rmse 61.1 is 2.0x the 30-snapshot
`failure_horizon`, meaning near-boundary RUL predictions carry error comparable to or
larger than the decision window itself (holdout precision at horizon=30 is 0.52 — see
`config/training_bearing.yaml`'s `max_rmse` comment for the full reasoning). Promoted
anyway, deliberately, to prove the CI/CD pipeline (train → gate → registry → staging eval
→ promotion gate → registry tag) works end-to-end for real, not as a claim that this
specific model is fit for operational use. Closing this requires model-quality work
(better features/model/split — see "known gaps (failure-horizon calibration)" above), not
a config or gate change.

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

**Current state (2026-09-13): `secrets.KUBE_CONFIG` does not exist on the repo at all**
(confirmed via `gh secret list`) — there is no cluster credential configured, not just an
unverified one. `promote-dev-to-staging.yaml`, `promote-staging-to-prod.yaml`, and
`rollback.yaml` will all fail at their "Configure kubeconfig" / deploy steps until a real
cluster + that secret exist. Because MLflow scoring and promotion are independent of the
K8s deploy mechanism by design (see each workflow's header comment), the steps in
`promote-dev-to-staging.yaml` were reordered so deploy/smoke-test failures
(`continue-on-error: true`) do not prevent holdout scoring from running and being pushed
to GCS — the job still correctly fails overall (final step), but the MLflow-side evidence
is captured either way. `promote-staging-to-prod.yaml`'s gate/promotion already ran before
its K8s steps, so it did not need the same restructuring. Closing this gap for real
requires either a real cluster + `KUBE_CONFIG`, or accepting (and explicitly deciding,
not silently allowing) that promotions exercise only the MLflow registry state change
until one exists.

## Alerts: what each one means and what to do

Every alert in `monitoring/prometheus/alerts.yaml` has an entry here. Shared first
steps:
- `kubectl get cronjobs,jobs -n pdm`
- the drift job's logs: `kubectl logs -n pdm job/<latest pdm-drift-check job>`
- the Production model: `MlflowClient().get_latest_versions("cmapss_rul", ["Production"])`

### DriftScoreHigh

**Meaning.** More than half the sensor columns differ from the Production model's
training data at the same life stage. The inputs have really changed: new operating
conditions, a new fault mode, a fleet change.

**Do.**
1. Check whether `RetrainHeldForSensorFault` is also firing. If it is, handle that first.
2. Otherwise a retrain Job was already created. Watch it: `kubectl logs -n pdm job/pdm-retrain-drift-...`.
3. Read the gate's decision in the job log ("Gate APPROVED/REJECTED ... reasons").
   - A rejection is normal and safe: Production is unchanged.
   - If it's rejected repeatedly while drift stays high, the model can't adapt with the
     data it has. Escalate: new labelled data or a model change is needed (see
     `docs/decisions.md` D-14 on gate noise).

### SensorFaultSuspected

**Meaning.** One sensor no longer tracks the others the way it did in training. The
`sensor` label names it; if several fire, the highest `pdm_sensor_fault_score` is the
likely culprit. Measured on FD001: catches a 0.5 std offset or a stuck sensor 100% of
the time; false alarms 0-3% on normal fleets.

**Do.**
1. Have maintenance check that sensor: calibration, wiring, a stuck value.
2. Don't retrain because of this alert. The model is fine; the input is wrong.
3. While it's firing, treat predictions from the affected engines with suspicion.

### RetrainHeldForSensorFault

**Meaning.** The drift check wanted to retrain, but a sensor also looks broken, so no
retrain Job was created (`sensor_check.hold_retrain_on_fault`).

**Do.**
1. Handle `SensorFaultSuspected` first.
2. If the sensor turns out to be fine and the drift is real, start the retrain by hand:
   `kubectl create job -n pdm --from=cronjob/pdm-retrain pdm-retrain-manual-$(date +%s)`.
3. If the sensor was broken, fix it and let the next checks run. Drift should go away.

### DriftCheckSkipped

**Meaning.** No drift decision for 6 hours.
- `skip_no_data`: no traffic.
- `skip_too_few_engines`: fewer than `current_window.min_engines` engines in the window
  (`pdm_drift_window_engines`). With fewer than 5 engines the checks are wrong 15-94%
  of the time (`docs/decisions.md` D-8).

**Do.**
1. Confirm traffic: `pdm_predictions_total` rate.
2. Confirm clients send `asset_id`: a spike in
   `pdm_input_rejected_total{reason="missing_asset_id"}` means they don't.
3. If the fleet is genuinely small, widen `current_window.lookback_hours` so a window
   covers 5+ engines. Don't lower `min_engines`.

### ServingErrorRateSpike

**Meaning.** More than 5% of live requests failed.

**Do.**
1. Look at `pdm_input_rejected_total` by `reason`.
   - `missing_asset_id`, `missing_features` or `non_finite` means a client or upstream
     data problem. The model is fine.
   - Otherwise check the serving pod logs for "Prediction failed".
2. If a canary is rolling out, Argo should roll it back. Otherwise
   `./scripts/rollback.ps1 undo`, then roll the model back (see Rollback).

### ServingHighLatencyP95

**Meaning.** p95 latency is over 1 second.

**Do.** Check pod CPU and memory, then check whether the model changed: the interval
model runs three models per request.

### NoModelLoaded

**Meaning.** No serving pod has a Production model, so all traffic is failing.

**Do.**
1. Check that the MLflow server is reachable from the pods.
2. Check that a Production version exists.
3. If the latest promotion is broken, roll back (`scripts/rollback_production.py`).

## Champion/challenger gate: reading a decision

`python -m pdm.training.retrain` (and the pdm-retrain CronJob) logs one line per check,
for example `cost_per_engine: challenger 12.675 vs champion 11.715 (limit 12.301) -> FAIL`.

- **Every check PASS:** the candidate was promoted. Its scores are on the version as
  `gate_*` tags.
- **Any check FAIL:** Production is unchanged. The candidate stays registered for
  inspection.
- **First promotion ever:** always rejected until a human reviews the scores and runs
  `python -m pdm.evaluation.champion_challenger --candidate-version N --promote --confirm-bootstrap`.
- **Known noise:** changing only the random seed moves holdout cost by about 8%, more
  than the 5% tolerance (`docs/decisions.md` D-14). One rejection on cost alone is not
  evidence the candidate is worse.

## Recording outcomes (the source of new training labels)

Retraining only improves the model if it gets new, true outcomes. The retrain job builds
labels from two things:
- the inference log (what the model saw, per `asset_id` and `cycle`), and
- the outcome store (what actually happened).

Nothing fills the outcome store automatically. Someone has to own this.

**What to record, per engine.**
- `failure` at the cycle it failed.
- `maintenance` at the cycle it was overhauled or replaced *before* failing.

Use the same `asset_id` and cycle count that clients send to `/predict`.

**How.** Export from the maintenance system (CMMS) as a CSV with the columns
`asset_id,event_type,cycle`, then:

```bash
python -m pdm.labels.outcomes import --csv cmms_export.csv    # safe to re-import
python -m pdm.labels.outcomes list
python -m pdm.labels.build                                    # optional: preview labels + counts
```

In the cluster, run the same commands in a pod with `/data` (the `pdm-mlflow-data`
volume) mounted and `OUTCOME_DB=/data/outcomes/outcomes.db`.

**Reading `python -m pdm.labels.build` counts.**
- `dropped_no_outcome_yet`: engines still running. Normal.
- `dropped_censored_below_cap`: readings within 125 cycles of a maintenance. Their true
  label is unknown, so they are dropped on purpose.
- `dropped_no_asset_or_cycle`: clients aren't sending `cycle`. **Fix the client**; those
  readings can never become labels.
