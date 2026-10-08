# Optional classical ML (V1 lifecycle and Phase 2)

Phase 2 (model catalog, cross-validation, benchmarks, hyperparameter search, bounded AutoML, ranking, richer preprocessing and metrics, explicit artifact compatibility) is described in [Phase 2](#phase-2-models-cross-validation-and-experiments). It extends the V1 lifecycle below; nothing in V1 was replaced.

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

Scikit-learn 1.7.2 runs the model catalog described under Phase 2; V1 specs (`ridge`, `logistic_regression`, `random_forest_classifier`, `random_forest_regressor` with `impute_median` and `standard_scale`) run unchanged. Classification labels may be numbers/strings. Random splits optionally stratify classification targets. Fractions must be positive and sum to one; unsuitable small/class-imbalanced partitions fail durably during preparation. Hyperparameters are allowlisted and bounded; arbitrary imports, estimators and process arguments are not accepted. CPU threads are fixed to one.

Preprocessing fits exclusively on training rows and is saved with the estimator. Train/validation/test metrics persist separately. The snapshot/environment records actual split indices, requested/resolved parameters, optional source revision, Python/library/platform versions and adapter source checksum. Inference requires a compatible artifact environment (see [Artifact compatibility](#artifact-compatibility-and-provenance)). Metadata supports reconstruction, not a promise of bit-identical results across hardware.

Models use pickle internally. Only generated, hash/size-verified artifacts are loaded. Database/artifact writers and the Python runtime are trusted; checksums detect corruption, not malicious administrator replacement. Prediction history retains input hashes/counts and outcomes, without raw input rows; training datasets are deliberately persisted. Backend inference errors are generic.

## Phase 2: models, cross-validation and experiments

Phase 2 is an extension of the lifecycle above, not a second ML system. Every benchmark, search or AutoML candidate is an ordinary V1 training job: one `kind='ml'` run with the same leases, heartbeat, cancellation, lease recovery, attempts, usage record, audit events, PostgreSQL artifact and tenant isolation. The experiment tables only record which jobs belong together and the persisted outcome. There is no second scheduler. Authorization follows the public rules: `tenant_admin` creates and cancels; authenticated tenant members read.

| Step | What it is | Registers a model? | Promotes / deploys? |
| --- | --- | --- | --- |
| Training | One job fits and evaluates one model | No | No |
| Benchmark | Listed configurations trained on identical data, features, split/CV, seed and metrics, then ranked | No | No |
| Search | The runtime generates grid or random candidates for one model; each is a job; results ranked | No | No |
| AutoML V1 | A bounded plan of one benchmark plus up to two searches, ranked together | No | No |
| Registration | You register one completed attempt (`POST /ml/registry/:id/versions`) | Yes, explicitly | No |
| Promotion | You move the version to `READY`, then create a version-pinned endpoint | n/a | Yes, explicitly |

A ranking or AutoML recommendation means "best candidate under this experiment's objective". It is never registered, promoted or deployed automatically.

### Models

`GET /ml/models` lists the catalog and metric registry. Each `ModelDefinition` (`src/ml/model-catalog.ts`) carries `id`, `backend`, `taskType`, an allowlisted bounded parameter table (which also generates the validator), `defaults`, `capabilities` and an optional cross-parameter rule. Unknown or out-of-range hyperparameters (for example `n_jobs`) are rejected at queueing, before every backend call and when search candidates are generated. Backends register definitions with `registerModels`, so another backend can be added without touching the core.

| Task | Model ids (backend `sklearn`) |
| --- | --- |
| Classification | `logistic_regression`, `random_forest_classifier`, `gradient_boosting_classifier`, `svc`, `k_neighbors_classifier`, `decision_tree_classifier` |
| Regression | `linear_regression`, `ridge`, `lasso`, `elastic_net`, `random_forest_regressor`, `gradient_boosting_regressor`, `svr`, `k_neighbors_regressor`, `decision_tree_regressor` |

Catalog defaults equal scikit-learn's defaults for the exposed parameters (a test compares them with the installed library). `svc` exposes decision scores, so it supports binary ROC AUC but not log loss. XGBoost and LightGBM are not included.

### Preprocessing

Operations are fitted on training rows only (the training partition, or each CV fold's training rows) and saved with the model.

| Operation | Applies to | Parameters |
| --- | --- | --- |
| `impute_median`, `impute_mean`, `standard_scale`, `min_max_scale` | number features | none |
| `impute_most_frequent`, `one_hot_encode` | string / boolean features | none (unknown categories encode as zeros) |
| `ordinal_encode` | string / boolean features | explicit `categories` order for every categorical feature (never inferred) |
| `variance_threshold` | assembled columns | `threshold` (default 0) |

Categorical features require exactly one encoder, `impute_most_frequent` must precede it, each operation appears at most once, and there is at most one numeric imputer and one scaler. The `features` list is the explicit inclusion list and can never contain the target. Inference rows are validated against the dataset schema.

### Metrics

Direction is declared once in the metric registry, never inferred from a name. `higher` means maximize and `lower` minimize.

| Task | Metric | Direction |
| --- | --- | --- |
| Classification | `accuracy`, `precision`, `recall`, `f1` (support-weighted), `roc_auc`, `log_loss` | higher, except `log_loss` lower |
| Regression | `mae`, `mse`, `rmse`; `r2` | lower; higher |

`evaluation.metrics` selects the set (default: V1's `accuracy`, or `mae`+`mse`). Metrics a model cannot produce are rejected up front. Stored partitions are `train`, `validation`, `test` and, with CV, `cv` (mean with population `std`) and `cv_fold` (one row per fold). No confusion matrix is produced. Measured `fitSeconds`, `predictSeconds` (scoring the test partition) and `cvFitSeconds` are stored per attempt; CPU and RAM are not recorded.

### Cross-validation

Add `evaluation.cv = {"strategy":"kfold"|"stratified_kfold","folds":2..20}` to a training spec (stratified is classification only). CV runs over the training partition only; validation and test stay held out and are still scored. Folds are shuffled with the job seed, computed once, persisted as `split_indices.fold_N`, and verified by the processor to partition the training indices exactly. Each fold refits a fresh preprocessing + estimator pipeline on the other folds, so nothing is preprocessed before CV. Persisted: strategy and fold count (frozen snapshot), seed, fold indices, per-fold metrics, mean, std and the model identity.

### Experiments

- `POST /ml/benchmarks`: `name`, `datasetVersionId`, `featurePipelineId`, `taskType`, `seed`, optional `split`, `evaluation {primaryMetric, secondaryMetrics, cv}`, 2-20 `candidates {algorithm, hyperparameters, label}`; optional `experimentId`, `idempotencyKey`. All candidates share dataset, pipeline, split/CV, seed and metrics by construction, and creation is atomic: any invalid candidate creates nothing.
- `POST /ml/searches`: one `algorithm`, `fixedHyperparameters`, a `space` of `choice`, `int` (optional `step`) or `float` (optional `log`, `steps` for grid) dimensions, `strategy` `grid` or `random`, `maxCandidates` 1-50. Grids above the cap are rejected rather than truncated; random search is seeded and samples without replacement; combinations the model rejects are dropped. Unsupported parameters, invalid ranges and empty spaces are rejected. Bayesian search is not implemented.
- `POST /ml/automl`: `datasetVersionId`, `target`, `taskType`, `metric`, `seed`, `cvFolds`, `budget {maxCandidateRuns 2-40, maxSearchCandidatesPerModel 1-10}`. The plan is deterministic: one derived shared pipeline, one benchmark child of default configurations of up to half the budget, and random-search children for random forest and gradient boosting within the remaining budget. The budget is an upper bound; there is no execution-time budget and no adaptive or multi-stage search.

Inspect with `GET /ml/benchmarks|searches|automl[/:id]`; `GET /ml/experiments/:id/ranking`; cancel with `POST /ml/experiments/:id/cancel`. Responses list every candidate (job, run, attempt, requested and resolved hyperparameters, metrics, timings, artifact size, failure, lineage from dataset version through pipeline, split/CV, seed and artifact), the persisted ranking configuration and ranking, and a recommendation.

**Ranking.** Candidates are ordered by the primary metric on the `cv` partition when CV is configured, otherwise `validation`, in the metric's declared direction; test data is never used. Ties break on lower fold std, secondary metrics in order, lower `fitSeconds`, then the training-run ID. The ranking configuration is stored. Before ranking, every candidate must share one cohort key (dataset content, schema, pipeline, split, seed, CV definition and actual indices); otherwise nothing is ranked.

**Failure semantics.** A failed candidate does not discard the others. When every candidate is terminal the experiment becomes `COMPLETED`, `PARTIALLY_COMPLETED`, `FAILED` or `CANCELLED` and its ranking is stored. Parent cancel cancels unfinished candidates through canonical run cancellation and keeps finished ones ranked. Stale workers rely on the existing lease recovery (a new attempt starts; the experiment stays running). The worker finalizes after the last job; a `tenant_admin` read also finalizes a complete experiment; other reads never write. Any number of workers may run candidates concurrently and the ranking is unchanged.

**Guarantees.** Same dataset version, configuration, seed and budget give the same folds, candidates, plan and ranking configuration; metrics are materially equal on the same library versions (not byte-identical artifacts). Leakage is tested against the real runner: final fits and every CV fold see only their own training rows, perturbing validation/test rows changes no fitted state or CV score, and inference replays the saved transformation independent of the batch.

### Artifact compatibility and provenance

Provenance identity is not artifact compatibility. The recorded environment (`ml_training_runs.environment`, also embedded in the artifact) has two roles:

| Field | Role | Decides inference? |
| --- | --- | --- |
| `backend`, `artifact_format`, `protocol`, `artifact_schema` | What the artifact is | Yes: unknown values are refused |
| Python major.minor, `sklearn`, `numpy`, `scipy`, `joblib`, `threadpoolctl` | Pickle/numeric compatibility | Yes: must equal the runtime |
| `adapter_sha256`, `platform`, Python patch | Exact provenance | No: recorded, never compared |

The recorded environment is checked before any bytes are unpickled and again from inside the artifact; content hashes, tenant-scoped blobs and the stored-format check are unchanged. Bump `artifact_schema` only when an artifact written before a change could no longer be used after it (payload layout, meaning of the feature matrix); editing runner code changes the checksum, which is provenance. V1 artifacts (protocol `meticule-sklearn-v1`, no schema field) are schema 1: tests run the unmodified V1 public runner, train a real artifact, and show the current runner predicts and evaluates it identically. Upgrading Python or any listed library intentionally makes existing artifacts unloadable and requires retraining.

### Example: benchmark, choose, register, promote, serve

1. `POST /ml/datasets` and `POST /ml/pipelines` as above (use `impute_median`, `standard_scale`, `impute_most_frequent`, `one_hot_encode` for mixed features).
2. `POST /ml/benchmarks` with four classifiers, `evaluation: {primaryMetric:"f1", secondaryMetrics:["accuracy"], cv:{strategy:"stratified_kfold", folds:5}}`.
3. Run ML workers; poll `GET /ml/benchmarks/:id` until the status leaves `RUNNING`.
4. Read `ranking.entries` and `recommendation.trainingRunId`. Nothing is registered yet.
5. Register explicitly: `POST /ml/registry` `{name}`, then `POST /ml/registry/:id/versions` `{trainingRunId}`.
6. Promote explicitly: `POST /ml/model-versions/:id/transition` `{status:"READY"}`, `POST /ml/endpoints` `{name,modelVersionId}`, then `POST /ml/endpoints/:id/predict`.

### Limits

One backend (`sklearn`); no neural networks, GPU or distributed training; no repeated CV, Bayesian or adaptive search, confusion matrix, CPU/RAM telemetry or execution-time budget; lists return the 100 newest rows. Stratified CV needs every class at least `folds` times in the training partition and `roc_auc` needs every class in each scored partition, otherwise the job fails durably. Artifacts remain capped at 16 MB in PostgreSQL.

## Public API

All `/ml/*` routes use existing authentication. Disabled ML returns 503. IDs are UUIDs. Version documents use camelCase; lifecycle/registry responses follow existing snake_case row conventions. Lists return at most 100 records. No UI or ML monitoring endpoint is added; Phase 2 adds only the model catalog and experiment routes below.

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
| GET | `/ml/models` | Model catalog and metric registry |
| GET/POST | `/ml/benchmarks`, `/ml/searches`, `/ml/automl` | List / create experiments (202) |
| GET | `/ml/benchmarks/:id`, `/ml/searches/:id`, `/ml/automl/:id` | Candidates, lineage, ranking, recommendation (404 for another kind) |
| GET | `/ml/experiments/:id/ranking` | Ranking and finality |
| POST | `/ml/experiments/:id/cancel` | Cancel unfinished candidates (`tenant_admin`) |

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

Real backend checks run when `ML_TEST_PYTHON` is set; normal unit tests need no Python. The GitHub Actions workflow installs Python 3.13 and the pinned `ml/requirements.txt` and always sets it, and a CI-only test fails the job if it is missing, so the real runner is exercised on every push and pull request (a local run without the variable skips those tests). Tests cover PostgreSQL storage, JWT/RLS, attempts, cancellation/recovery, stale workers, atomic publication, all fifteen models, preprocessing, K-fold and stratified K-fold, fold-local fitting and leakage, reproducibility, benchmark/search/AutoML planning and execution, ranking direction and tie-breaks, partial failure, concurrency, artifact compatibility with a real V1 artifact, registry/version pins, inference schema errors, failed predictions, comparison, public authorization and tenant isolation, and migration safety. Existing integration tests are rerun to check agent behavior.

## Deferred scope

Other framework adapters (including XGBoost/LightGBM), richer data formats, streaming/large artifacts, resumable backend checkpoints, high-throughput serving, GPU scheduling, repeated CV, Bayesian or multi-stage AutoML, distributed training, fine-tuning and drift monitoring are deferred.

Synchronous inference commits its outcome with the HTTP tenant transaction; terminated requests are not resumable inference jobs. Standalone callers can leave PENDING evidence after a crash. Successful training usage records measure wall time with zero token/cost counts; failed/recovered attempts remain in history, not estimated billing. Blob retention/GC and pagination remain deferred.
