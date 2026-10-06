# Optional classical ML V1

ML extends Meticule's existing PostgreSQL run queue through `kind='ml'`. The agent executor, JWT authentication, tenant transactions and existing RLS boundary remain authoritative. Backend adapters compute; they never schedule work or change lifecycle state.

The framework-neutral domain includes Dataset, immutable DatasetVersion and DatasetSchema, FeaturePipeline versions, SplitDefinition, TrainingJob, TrainingRun attempts, Experiment, Metric, ModelArtifact, ModelRegistryEntry, ModelVersion, InferenceEndpoint and PredictionRecord. Training freezes the dataset/pipeline versions, algorithm/backend, requested/resolved hyperparameters, random seed, split specification and actual indices, source revision when supplied, environment, timing, outcome, metrics and artifact references.

Training phases: QUEUED → PREPARING → TRAINING → EVALUATING → SAVING → COMPLETED, with failure/cancellation terminal states. Canonical leases own execution. Recovered ML work closes the abandoned attempt and starts a new attempt with the same frozen snapshot. This ML-specific recovery path does not change agent recovery policy. Model versions move REGISTERED → READY → RETIRED, or REGISTERED → REJECTED. READY is an explicit operator decision.

## Port decisions

| Reviewed source component | Public decision | Reason |
| --- | --- | --- |
| Domain entities/state policies and backend protocol | Port unchanged | Framework-neutral ML behavior |
| Dataset/training persistence and processor | Adapt | Reuse public runs, transactions, audit and usage records |
| Registry, version-pinned inference, comparison | Adapt | Same lifecycle with existing public authorization |
| Local filesystem ArtifactStore | Replace with bounded PostgreSQL adapter | Preserve useful public precursor storage without shared-volume assumptions |
| Python runner and pinned sklearn dependencies | Port to existing `ml/` paths | Working numeric classical ML backend; remains optional |
| ML table isolation | Adapt to existing public RLS | Public repository already ships tenancy and `durable_agent_api` |
| Private `ml_manager`/`ml_viewer` roles | Omit | Existing `tenant_admin` authorizes mutations; authenticated tenant members read/predict |
| Private UI, operational monitoring endpoints and deployment settings | Omit | Outside public ML V1 scope |
| Private tests | Adapt | Test public storage, authorization and actual HTTP/runtime behavior |

All public precursor files were inspected before replacement. Useful ideas retained: workload-filtered claims, nullable agent fields only for ML, nested transactions, bounded content-addressed PostgreSQL artifacts, and a subprocess adapter with pinned dependencies. Superseded: the combined untested ML service, matrix-only schemas, the four-table experiment, permissive model state changes, alternate backend protocol and duplicate inline API routes. The precursor lacked versioned datasets/pipelines, durable attempt phases and tested publication integrity. Its source was archived locally outside this checkout before editing; it is not part of the public release.

The public checkout already has tenant/auth infrastructure; this port does not introduce a second tenancy mechanism or identity issuer. No private environment files, credentials, datasets, generated artifacts, private UI or deployment configuration are included.

Publication and failure handling both check the current immutable attempt ID under the canonical run lock, including reclamation by a process using the same worker ID. Public artifact bytes live in a tenant-scoped `ml_blobs` table, addressed by SHA-256 and checked against the recorded hash and size before loading. Each blob is capped at 16 MB; larger data/checkpoints require a different storage adapter.

## Migration compatibility

Apply normal migrations before starting upgraded API/workers, and restart older workers before enabling ML. The uncommitted `002_ml.sql` precursor was never part of the released public migration chain. If it was manually applied locally, V1 migration fails explicitly rather than overwriting its incompatible tables or discarding data. Export that experimental data and migrate a fresh database; no automatic destructive conversion is performed.

## Enable ML

Normal API/worker startup needs no Python while ML is disabled. From the repository root:

```powershell
python -m venv .tmp/ml-venv
.tmp/ml-venv/Scripts/python.exe -m pip install -r ml/requirements.txt
$env:ML_ENABLED='true'
$env:ML_PYTHON=(Resolve-Path .tmp/ml-venv/Scripts/python.exe).Path
$env:ML_TIMEOUT_MS='120000'
# Set DATABASE_URL and JWT configuration using README.md's existing workflow.
npm run db:migrate
npm run dev:api
# Second terminal, same configuration:
npm run dev:worker
```

On Linux/macOS use `python3 -m venv .tmp/ml-venv` and `.tmp/ml-venv/bin/python`. No artifact-root setting or shared volume is required. The existing Node-only image does not install Python; containerized ML needs an operator-provided optional Python environment.

The local identity helper already grants `tenant_admin`. That existing role manages datasets, pipelines, experiments, training and model registration/promotion/endpoints. Authenticated tenant members may inspect ML metadata and predict. JWT claims still intersect persisted membership roles; existing tenant transactions/RLS apply to all HTTP database access. No private ML roles are introduced.

## Backend and provenance

Scikit-learn 1.7.2 supports numeric features with `ridge`, `logistic_regression`, `random_forest_classifier` and `random_forest_regressor`. Classification labels may be numbers/strings. Pipeline operations are `impute_median` and `standard_scale`. Random splits optionally stratify classification targets. Fractions must be positive and sum to one; unsuitable small/class-imbalanced partitions fail durably during preparation. Hyperparameters are allowlisted/bounded and CPU threads fixed to one. Arbitrary Python/imports, client paths and artifact uploads are not accepted.

Preprocessing fits exclusively on training rows and is saved with the estimator. Train/validation/test metrics persist separately. The snapshot/environment records actual split indices, requested/resolved parameters, optional source revision, Python/library/platform versions and adapter source checksum. Model loading requires that recorded environment. Metadata supports reconstruction, not a promise of bit-identical results across hardware.

Models use pickle internally. Only generated, hash/size-verified artifacts are loaded. Database/artifact writers and the Python runtime are trusted; checksums detect corruption, not malicious administrator replacement. Prediction history retains input hashes/counts and outcomes, without raw input rows; training datasets are deliberately persisted. Backend inference errors are generic.

## Public API

All `/ml/*` routes use existing authentication. Disabled ML returns 503. IDs are UUIDs. Version documents use camelCase; lifecycle/registry responses follow existing snake_case row conventions. Lists return at most 100 records. No UI or ML monitoring endpoint is added.

| Method | Path | Behavior |
| --- | --- | --- |
| GET/POST | `/ml/datasets` | List / ingest `{name,schema,rows}` |
| GET/POST | `/ml/datasets/:id/versions` | Inspect / append immutable versions |
| GET/POST | `/ml/pipelines` | List versions / create `{name,definition}` |
| POST | `/ml/pipelines/:logicalId/versions` | Append a pipeline version |
| GET/POST | `/ml/experiments` | List / create `{name}` |
| GET | `/ml/experiments/:id` | Experiment and jobs |
| POST | `/ml/training-jobs` | Queue `{experimentId,spec}`, returning 202 |
| GET | `/ml/training-jobs/:id` | Job, attempts, metrics and artifact references |
| POST | `/ml/training-jobs/:id/cancel` | Canonical cancellation |
| GET | `/ml/training-runs/:baseline/compare/:candidate` | Metric deltas/regression flags |
| GET/POST | `/ml/registry` | List / create `{name}` |
| GET/POST | `/ml/registry/:id/versions` | List / register `{trainingRunId}` |
| GET | `/ml/model-versions` and `/ml/model-versions/:id` | Inspect versions and provenance |
| POST | `/ml/model-versions/:id/transition` | `{status:"READY"}`, `RETIRED` or `REJECTED` |
| GET/POST | `/ml/endpoints` | List / create `{name,modelVersionId}` for a READY version |
| POST | `/ml/endpoints/:id/disable` | Disable a pinned endpoint |
| POST | `/ml/endpoints/:id/predict` | `{rows:[...]}`; synchronous inference |
| GET | `/ml/endpoints/:id/predictions` | Last 100 outcomes |

Prediction batches are capped at 1,000 and must follow the original dataset schema with its target omitted. Malformed schema definitions return 400, incompatible data/inference rows 409, backend inference failures 422 with a durable FAILED record, and management-role failures 403. Retiring a version disables its endpoints atomically.

Registering the same completed attempt under the same entry is idempotent; a different attempt creates the next version, initially REGISTERED. Promoting a new version never retargets old endpoints. Comparisons require identical dataset content/schema, feature definition, seed, split specification and actual row indices, with matching metric names/directions. Algorithms/hyperparameters may differ. Comparison flags never promote automatically.

## Request shapes

Dataset input is `{name,schema:{id,columns:[{name,type,nullable}],target},rows:[...]}`. For numeric `x` predicting numeric `y`, columns are `[{"name":"x","type":"number"},{"name":"y","type":"number"}]`, target is `"y"`, and rows are objects such as `{"x":10,"y":21}`. Use a newly generated schema UUID and enough rows for all partitions. Schema IDs cannot be reused for different definitions.

Pipeline input is `{ "name":"numeric", "definition":{ "features":["x"], "steps":[{ "operation":"standard_scale" }] } }`. Create an experiment with `{ "name":"linear comparison" }`. Queue training using returned immutable IDs:

```json
{
  "experimentId": "11111111-1111-4111-8111-111111111111",
  "spec": {
    "datasetVersionId": "22222222-2222-4222-8222-222222222222",
    "featurePipelineId": "33333333-3333-4333-8333-333333333333",
    "backend": "sklearn",
    "algorithm": "ridge",
    "hyperparameters": { "alpha": 0.01 },
    "seed": 42,
    "split": { "id":"44444444-4444-4444-8444-444444444444", "strategy":"random", "train":0.6, "validation":0.2, "test":0.2 },
    "source": null
  }
}
```

Poll the returned job ID until COMPLETED/FAILED/CANCELLED. Create a registry entry with `{name}`, register `{trainingRunId}` using the successful attempt ID, explicitly promote `{status:"READY"}`, create `{name,modelVersionId}`, then predict `{rows:[{x:10}]}`. READY records an operator decision, not a production-quality claim. Later dataset/pipeline revisions do not alter frozen training snapshots.

## Verification

```powershell
npm run build
npm run test:unit
$env:ML_TEST_PYTHON=(Resolve-Path .tmp/ml-venv/Scripts/python.exe).Path
# DATABASE_URL and TEST_DATABASE_URL must target a dedicated migrated test database.
npm run test:integration
```

Real backend checks run only with `ML_TEST_PYTHON`; normal unit tests need no Python. Tests cover actual PostgreSQL storage, JWT/RLS, attempts, cancellation/recovery, stale workers, atomic publication, all four algorithms, training-only preprocessing, registry/version pins, inference schema errors, failed predictions, comparison and migration safety. Existing integration tests are rerun to check agent behavior. These repository suites truncate tables and must not target production data.

## Deferred scope

Other framework adapters, richer data formats, streaming/large artifacts, resumable backend checkpoints, high-throughput serving, GPU scheduling, AutoML, distributed training, fine-tuning and drift monitoring are deferred. This task ports only classical ML V1.

Synchronous inference commits its outcome with the HTTP tenant transaction; terminated requests are not resumable inference jobs. Standalone callers can leave PENDING evidence after a crash. Successful training usage records measure wall time with zero token/cost counts; failed/recovered attempts remain in history, not estimated billing. Blob retention/GC and pagination remain deferred.
