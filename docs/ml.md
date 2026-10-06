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

## Migration compatibility

Apply normal migrations before starting upgraded API/workers, and restart older workers before enabling ML. The uncommitted `002_ml.sql` precursor was never part of the released public migration chain. If it was manually applied locally, V1 migration fails explicitly rather than overwriting its incompatible tables or discarding data. Export that experimental data and migrate a fresh database; no automatic destructive conversion is performed.

## Deferred scope

Other framework adapters, richer data formats, streaming/large artifacts, resumable backend checkpoints, high-throughput serving, GPU scheduling, AutoML, distributed training, fine-tuning and drift monitoring are deferred. This task ports only classical ML V1.
