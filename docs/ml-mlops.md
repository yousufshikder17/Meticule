# MLOps

Meticule's MLOps layer adds experiment tracking, lineage, MLflow interoperability, Optuna studies, ONNX export and runtime,
Matplotlib reports, registry governance, a deployment lifecycle and drift monitoring on top of the ML lifecycle described in
[ml.md](ml.md). Everything extends the existing records: training still runs as ordinary durable jobs on the shared `runs`
queue and leases, models are still registry versions, serving is still a pinned endpoint, artifacts stay in the PostgreSQL
blob store (16 MB per blob), and every table is tenant-scoped with the same row-level security policy. Migrations
`008`–`016` are additive.

Everything runs self-hosted. MLflow is optional and only mirrors Meticule's records; Meticule never depends on it.

## Enabling

The core features (tracking, lineage, governance, deployments, monitoring) are on whenever `ML_ENABLED=true`. Four features
run a Python runner and are off by default. They share one interpreter with `ml/requirements-mlops.txt` installed:

```sh
python -m venv .mlops-venv
.mlops-venv/bin/python -m pip install -r ml/requirements-mlops.txt     # Windows: .mlops-venv\Scripts\python.exe
export ML_MLOPS_PYTHON="$PWD/.mlops-venv/bin/python"
export ML_ONNX_ENABLED=true      # ONNX export and ONNX Runtime inference
export ML_OPTUNA_ENABLED=true    # Optuna studies (set on the API and on ML workers)
export ML_REPORTS_ENABLED=true   # Matplotlib reports
export ML_MLFLOW_ENABLED=true ML_MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # optional token: ML_MLFLOW_TRACKING_TOKEN
```

The MLOps requirements pin the same scikit-learn, NumPy, SciPy, joblib and threadpoolctl versions as `ml/requirements.txt`.
That is required, not incidental: ONNX export unpickles scikit-learn artifacts under the same compatibility check the
scikit-learn runner uses. The runners are `ml/onnx_runner.py`, `ml/optuna_runner.py`, `ml/report_runner.py` and
`ml/mlflow_runner.py`; like the other runners they resolve relative to the working directory, so run the API and workers from
the package root. A disabled feature's routes answer 503 `ml_feature_not_configured`.

## Authorization

Writes (creating, changing, exporting, approving, deploying, evaluating drift, submitting feedback) need `tenant_admin`.
Reads and predictions need an active membership in the tenant. Another tenant's records are not found (404). The private
edition's ML roles do not exist here.

## Experiment tracking

Experiments gain `description`, `project`, `tags` (up to 20 string pairs) and an immutable `parentExperimentId`. Only the
description and tags can change (column-level grant plus a trigger). An experiment's status is derived from its jobs on
every read (`EMPTY`, `RUNNING`, `COMPLETED`, `PARTIALLY_COMPLETED`, `FAILED`, `CANCELLED`), so it cannot drift from them.
`POST /ml/experiments` still accepts a bare `{name}`.

- `GET /ml/experiments?project=&nameContains=&status=&tag=key:value&parentExperimentId=&createdAfter=&createdBefore=&limit=`
- `PATCH /ml/experiments/:id {description?, tags?}`
- `GET /ml/experiments/:id/runs`: every attempt of every job, with parameters, outcome, failure category and metrics.
- `POST /ml/training-runs/compare {trainingRunIds: [2..10]}`: parameters, metrics with the best run per metric, and
  environment differences. Metrics are declared comparable only when every run is completed and all share one cohort
  (dataset content, schema, features, split, seed, folds).
- `POST /ml/training-jobs/:id/reproduce`: queues an ordinary job with the original's frozen specification in the same
  experiment (`reproduces_job_id`). Refused when the original backend is not configured or the dataset bytes fail
  verification.
- `GET /ml/training-runs/:id/reproducibility/:candidate`: `identical` (same split and exactly equal metrics),
  `numerically_equivalent` (within 1e-9 relative), `different`, or `not_comparable` (a different specification), with the
  environment differences. It never claims bit-identical artifacts.

## Lineage

Lineage is the relational record read as one document; nothing is copied. Every query filters the caller's tenant.

- `GET /ml/lineage/model-versions/:id?verify=true`: upstream (experiment and its ancestors, orchestration, job, every
  attempt, checkpoints, training configuration, dataset version with content hash, feature pipeline), downstream
  (endpoints, deployments, ONNX exports) and the version's lifecycle events.
- `GET /ml/lineage/training-runs/:id`, `GET /ml/lineage/dataset-versions/:id`, `GET /ml/lineage/deployments/:id`.

`issues` lists defects found while reading: a snapshot that disagrees with its job, dataset content that differs from the
version, a resumed attempt whose checkpoint is missing, a truncated experiment chain, and, with `verify=true`, blob bytes
that are missing or fail their checksum.

## MLflow interoperability

`POST /ml/training-runs/:id/mlflow-export` copies one completed attempt to the configured tracking server: parameters
(`backend`, `algorithm`, `seed`, split fractions, identities, `hp.*`, `resolved.*`), metrics (`<partition>.<name>`, folds as
steps, epoch curves as `epoch.train_loss` / `epoch.validation_loss`), `meticule.*` identity tags and a
`meticule/lineage.json` artifact. The MLflow experiment is named `<prefix>/<tenant id>/<experiment id>`.

- Idempotent per attempt and server: a completed export is returned (`reused: true`); an MLflow run left incomplete by an
  interrupted export is deleted and written again, so retries never duplicate runs.
- A failure is recorded (`ml_mlflow_exports`, status `FAILED`, category only) and answered 502; the next call retries.
- Native model artifacts and checkpoints are listed under `unsupported` and never uploaded.
- `POST /ml/mlflow/imports {experimentId, mlflowRunId}` stores one run's parameters, latest metrics, user tags, status and
  times as data (`ml_external_runs`). No artifact or model is ever downloaded or executed.

The tracking URI comes only from configuration, so a request cannot point the exporter at another host, and the operator
is responsible for the server's own access control. The token, if any, is passed only to the MLflow runner process. Import
refuses a run that Meticule exported for another tenant (an experiment under `<prefix>/<other tenant>/` or a
`meticule.tenant_id` tag that is not the caller's) as not found. Meticule cannot isolate direct access to the server: one
MLflow server shared by several tenants lets anyone with access to that server read every tenant's exported runs. Use one
server per tenant where that matters.

## Optuna studies

`POST /ml/optuna-studies` takes the same bounded search space as `/ml/searches` plus `sampler` (`tpe` or `random`),
`maxTrials` (at most 50), `parallelism` (at most 4) and `timeoutSeconds` (60 to 86,400). Optuna's own samplers choose every
trial through ask-and-tell.

- Every trial is an ordinary durable training job. Each ask rebuilds an in-memory study from the durable history:
  completed jobs with the objective metric, failed or cancelled jobs as `FAIL`, open jobs as `RUNNING`. The sampler seed is
  the study seed plus the history length, so asks are deterministic and a rebuild never repeats an earlier draw.
- The worker's job-terminal hook asks for the next trials under the orchestration row lock. Results come only from
  committed job records, so a stale worker or a cancelled trial never informs the sampler. Cancelled studies accept no new
  trials (database guard). No trial is asked after the time limit; trials already queued or running finish normally.
- ML workers with `ML_OPTUNA_ENABLED` also sweep running studies every 30 seconds, so a study whose ask failed with nothing
  running, or whose time limit passed, advances or finalizes without waiting for a reader.
- Combinations the model's allowlist refuses are fed back as in-memory failures and never run.
- `GET /ml/optuna-studies/:id` adds `study`: trials (number, parameters, state, value, training run) and the best trial.
  Register the best trial with the ordinary `POST /ml/registry/:id/versions {trainingRunId}`.

There is no pruning (trials report no intermediate values), and only the TPE and random samplers are offered.

## Registry governance

- `PUT /ml/registry/:id/policy {minimumMetrics: [{name, partition, threshold}], requireApproval, requireVerifiedOnnx}`
  appends a policy version. Thresholds follow each stored metric's direction. A version is held to every policy version in
  force since it was registered (the requirements combine): tightening applies to existing versions, relaxing only to
  versions registered afterwards. So an administrator cannot relax a policy to promote their own version; register a new
  version to use a relaxed policy.
- `POST /ml/model-versions/:id/approvals {decision: APPROVED|REJECTED, note}`: by a `tenant_admin` other than the version's
  registrant. The newest decision counts. Separation of duties is keyed on the authenticated identity (the token subject),
  never on roles, so adding roles to the registrant changes nothing. A tenant admin can create service identities, so
  service identities can never approve: an approval is a person's decision. Both rules are enforced by the service and by a
  database trigger.
- Versions registered before migration `011` recover their registrant from the `ml.model_registered` audit event written at
  registration. A version without such an event keeps no registrant, and separation of duties cannot apply to it.
- `GET /ml/model-versions/:id/readiness` explains what passes and what is missing.
- `REGISTERED -> READY` evaluates the policy and the artifact's integrity under the version's row lock. A refusal is
  recorded with reason codes and answered 409 `{reasons}`. Without a policy, earlier promotion behavior is unchanged.
- Creating a plain endpoint (`POST /ml/endpoints`) applies the same check as a deployment: the version must still satisfy
  the policy and its artifact must verify.
- `GET /ml/model-versions/:id/events`: registration and every status change are written by a database trigger (actor and
  reason from transaction-local settings); approvals, refusals and deployment changes are recorded by the services. The
  history is append-only.

Reason codes: `artifact_integrity_failed`, `policy_metric_missing`, `policy_metric_threshold`, `approval_required`,
`approval_rejected`, `onnx_export_required`, `model_not_ready`, `backend_unavailable`, `inference_check_failed`,
`input_schema_incompatible`.

Meticule trusts the identity provider that signs its tokens to give each person one stable subject. It cannot tell when
one person holds two user identities.

## ONNX export and ONNX Runtime

Supported: scikit-learn classification and regression versions whose features are all numeric, with the numeric
preprocessing steps (`impute_median`, `impute_mean`, `standard_scale`, `min_max_scale`, `variance_threshold`). Verified for
all fifteen catalog classifiers and regressors; class labels must be integers or strings. Everything else (string or
boolean features, anomaly detection, every other backend) is recorded as `UNSUPPORTED` with its reason. No model is changed
to make a conversion succeed.

- `POST /ml/model-versions/:id/onnx-exports` converts with skl2onnx (opset 17, ai.onnx.ml 3), checks the model, runs it with
  ONNX Runtime on the attempt's held-out test rows (at most 256) and compares with the native artifact. Tolerance:
  regression `|onnx - native| <= 1e-4 + 1e-3 * |native|` (ONNX runs in float32); classification: every label equal and
  probabilities within 1e-4. `VERIFIED` (201) or `PARITY_FAILED` / `UNSUPPORTED` (422); exports are append-only.
- ONNX bytes are stored in the PostgreSQL blob store. An export larger than the 16 MB blob limit is recorded as
  `UNSUPPORTED` with category `output_limit_exceeded`.
- The runtime gate refuses, before a session exists: models over 64 MiB, external data in any tensor (initializers, sparse
  initializers and tensor attributes, subgraphs included), local functions, any domain other than the default and
  `ai.onnx.ml`, opsets outside 9–17 (ai.onnx.ml 1–3), any operator off the allowlist (subgraphs included), models that fail
  `onnx.checker`, and anything other than one float `[rows, features]` input. Sessions run on CPU with one thread.
- `GET /ml/onnx-exports/:id/content` (`tenant_admin`) returns the bytes of `VERIFIED` exports only, re-verified against
  their checksum. Exports record producer and runtime versions, opsets, operators, signature and the parity report.

Not supported: PyTorch, TensorFlow, JAX, XGBoost, LightGBM and Transformers exports, GPU execution providers, and any
model the platform did not export itself.

## Deployments

A deployment is a persistent record; the serving unit is an ordinary pinned endpoint created on activation. Serving is
synchronous HTTP inference in the API process: there is no separate serving process, remote host, traffic splitting,
canary rollout, autoscaling or GPU fleet behind a deployment.

| Status | Meaning | Next |
| --- | --- | --- |
| `CREATED` | no revision yet | `VALIDATING` |
| `VALIDATING` | a revision is being validated, nothing serves | `READY`, `FAILED` |
| `READY` | a validated revision exists, nothing serves | `ACTIVE`, `VALIDATING` |
| `ACTIVE` | one revision serves through its endpoint | `INACTIVE` (switching revisions stays `ACTIVE`) |
| `INACTIVE` | deactivated, or its endpoint was disabled | `ACTIVE`, `VALIDATING` |
| `FAILED` | validation failed and nothing deployable exists | `VALIDATING` |

- `POST /ml/deployments {name}`; `POST /ml/deployments/:id/revisions {modelVersionId, runtime: native|onnx, onnxExportId?,
  allowSchemaChange?, reason?}` validates the revision: model `READY`, the registry's policy and artifact integrity,
  runtime availability (the backend, or ONNX with a `VERIFIED` export), input-contract compatibility with the serving
  revision (prediction columns and task; overridable with `allowSchemaChange`), and a smoke inference on the training run's
  own test rows. Refusals answer 409 `{reasons}` and are recorded.
- `POST /ml/deployments/:id/activate {revision?, allowSchemaChange?}`, `/rollback {revision?, allowSchemaChange?}`
  (default: the revision that served before), `/deactivate`. Eligibility is checked again on every activation, and so is the
  input contract against the revision that served last, even when nothing serves at the moment. The new endpoint is
  created before the old one is disabled. Any unexpected error during validation fails the revision; it never stays
  `VALIDATING`.
- `POST /ml/deployments/:id/predict {rows}` answers with the prediction record plus `deployment_id` and `revision`.
- Database triggers enforce the transitions, keep validated revisions immutable, require an active endpoint to match its
  revision, and move a deployment to `INACTIVE` whenever its endpoint is disabled (model retirement or the endpoint API).
  Deployment events are append-only and mirrored into the model version's lifecycle.

## Monitoring and drift

- Each model version gets one immutable reference profile from its training partition: training deciles as bin edges for
  numeric features, the 20 most frequent values plus "other" for string and boolean features, a null bucket for each, and
  the target as the output reference (bins for regression, labels for classification). Text, timestamp and media features
  are not profiled.
- Every completed prediction stores only its counts per reference bin. Raw inputs are never stored.
- `POST /ml/deployments/:id/drift-evaluations {modelVersionId?, from?, to?, minSamples=100, thresholds}` evaluates a
  version pinned by one of the deployment's revisions (by default the serving one, else the one that served last) and
  compares the window (default the last 24 hours, at most 31 days) with the reference, per feature and for the output:
  population stability index (proportions floored at 1e-4), Jensen-Shannon divergence in bits, null rates, and a status
  from the PSI bands (moderate at 0.1, significant at 0.25 by default). Below `minSamples` observed rows the status is
  `insufficient_data`. Evaluations are persisted with their window and thresholds; re-running a window over the same
  profiles reproduces them. These are descriptive distances and conventional bands, not significance tests.
  Kolmogorov-Smirnov is not offered, because only binned counts are kept.
- `POST /ml/predictions/:id/feedback {labels}`: one ground-truth submission per completed classification or regression
  prediction, typed like the target. Monitoring then reports accuracy or MAE/RMSE.
- `GET /ml/deployments/:id/monitoring?from=&to=`: per model version, requests, rows, error rate, failure categories,
  latency (average, p50, p95, max), output summary and ground-truth performance, plus the latest drift status. The endpoint
  summary `GET /ml/endpoints/:id/monitoring` keeps its fields and adds `p50_latency_ms` and `failures_by_category`.
- Aggregation runs in SQL. `ml_prune_monitoring(interval)` lets the database owner delete old profiles (run it from a
  scheduled job); the API role cannot execute it or delete monitoring data. Nothing retrains, promotes or deactivates on
  drift.

## Reports

`POST /ml/reports` renders PNG or SVG (320–1600 by 240–1200 pixels) headless with Matplotlib. The platform computes every
number; the runner only draws five chart primitives from finite numbers and short labels, with mathtext off and text
stripped of control characters. SVG text is drawn as paths, and content is served with `nosniff` and a sandboxing CSP.

| Kind | Input | From |
| --- | --- | --- |
| `training_curves` | `jobId` | epoch history (iterative models) |
| `metric_comparison` | `trainingRunIds`, `metric`, `partition` | stored metrics |
| `optimization_history` | `orchestrationId` | Optuna trials and best so far |
| `confusion_matrix` | `trainingRunId` | classification predictions on the test rows |
| `residuals` | `trainingRunId` | regression predictions on the test rows |
| `roc_curve`, `precision_recall`, `anomaly_scores` | `trainingRunId` | anomaly scores and boolean labels on the test rows |
| `forecast` | `trainingRunId` | forecast from train and validation history against the test rows |
| `drift`, `drift_trend` | `deploymentId` | persisted drift evaluations |

Each report stores its request, sources (record identities and hashes), computed summary, producer version and the image
in the PostgreSQL blob store; reports are append-only and belong to their tenant. ROC and precision-recall need scores,
which Meticule models expose only for anomaly detection, so classification models are refused rather than drawn from
labels. Prediction-based reports score at most 5,000 test rows.

## Security summary

Tenant isolation is the existing pattern: every new table has the `tenant_isolation` RLS policy and every query filters the
tenant. ONNX bytes and reports are re-verified against their checksums on every read. Runners receive only an allowlisted
environment (the MLflow token is added for that runner only), and their failures surface as fixed codes. No new code path
unpickles anything except the existing scikit-learn artifacts, under the existing compatibility check.

## Limitations

- ONNX covers scikit-learn classification and regression over numeric features, on the CPU provider.
- Optuna has no pruning; samplers are TPE and random.
- Serving is synchronous in-process inference: no separate serving process, traffic splitting, canary, autoscaling, remote
  host or GPU fleet.
- Drift uses binned PSI and Jensen-Shannon on tabular features only; the PSI bands are conventions.
- MLflow: one tracking server per deployment, metadata only, and no isolation of direct server access.
- Reports: ROC and precision-recall only where scores exist; at most 5,000 scored rows; byte-identical output only within
  one environment.
- Every artifact, ONNX export and report image is a PostgreSQL blob of at most 16 MB.
- Monitoring profiles are pruned only when an operator runs `ml_prune_monitoring`.
