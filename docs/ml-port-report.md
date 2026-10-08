# ML V1 public port review

The four specified source commits were reviewed locally: `732e65f`, `d7ae278`, `282574d`, `592ad5a`. No cherry-pick was used. The public checkout's seven modified tracked files and nine untracked ML files were all read before replacement; no unrelated changes were found. Their original contents were archived outside this repository.

## Source components

| Component | Result |
| --- | --- |
| `src/ml/domain.ts`, `src/ml/backend.ts` | Unchanged framework-neutral domain/protocol |
| Python computation runner and dependency pins | Unchanged computation protocol; retained public `ml/` filenames |
| `src/ml/sklearn-backend.ts` | Adapter path adapted to public runner |
| `src/ml/persistence.ts`, `training-processor.ts` | Public authorization, PostgreSQL artifacts, attempt-ID failure fencing |
| `src/ml/model-service.ts`, `routes.ts` | Public authorization/request-scoped storage; idempotent registration; monitoring endpoint omitted |
| Local filesystem artifact store/configuration | Replaced with bounded PostgreSQL storage and public opt-in configuration |
| ML migrations | Public storage and permissions; ML-only recovery; registration uniqueness; precursor migration guard |
| Runtime/database integration | Existing queue, leases, transactions, usage and audit reused; agent recovery policy unchanged |
| Domain/backend/storage/lifecycle tests | Adapted to actual public storage and signed HTTP behavior; all four algorithms covered |
| Private documentation/configuration | Public setup/port report written separately; no private environment/deployment files copied |

Private ML roles, the monitoring endpoint, private product/deployment assumptions and broader stages were deliberately omitted. Public tenancy/JWT/RLS were retained because they already exist in the released architecture; no new identity or tenant-policy system was introduced.

## Existing public precursor classification

| Reviewed public path | Classification and disposition |
| --- | --- |
| `src/api/app.ts` | Conflicting API contract; replaced duplicate inline ML routes with the V1 module |
| `src/db/repositories.ts` | Useful workload/null-agent changes retained; paused/manual ML recovery superseded by tested attempt recovery |
| `src/db/tenant-session.ts` | Useful nested-transaction export retained |
| `src/domain/schemas.ts` | Useful workload/null-agent fields retained |
| `src/execution/agent-loop.ts` | Useful guard preventing ML dispatch to agents retained |
| `src/worker/worker.ts` | Useful workload filtering retained |
| `src/worker/main.ts` | Useful canonical ML worker pattern adapted to the V1 processor |
| `migrations/002_ml.sql` | Obsolete uncommitted four-table experiment removed; incompatible applied copies fail explicitly without dropping data |
| `src/ml/backend.ts` | Incompatible precursor protocol superseded by the completed neutral protocol |
| `src/ml/artifact-store.ts` | Useful bounded PostgreSQL content-addressing retained/adapted |
| `src/ml/configuration.ts` | Useful optional-backend idea retained; timeout/flags adapted |
| `src/ml/sklearn-backend.ts` | Superseded by complete bounded adapter with four algorithms/environment matching |
| `src/ml/schema.ts` | Obsolete matrix-only contracts replaced by versioned lifecycle entities |
| `src/ml/ml-service.ts` | Superseded monolithic service; useful idempotent registration preserved |
| `ml/sklearn_runner.py` | Superseded combined train/predict protocol; public path retained |
| `ml/requirements.txt` | Useful pinned versions retained unchanged |

## Capability comparison

Public V1 retains the complete classical path: validation, immutable datasets/pipelines, seeded splits with actual indices, durable attempts/phases, separate evaluation metrics, experiment provenance, artifact integrity, explicit registry/promotion, pinned inference, prediction outcomes and reproducible regression comparison. It uses the same canonical lifecycle without importing a parallel scheduler.

Public-specific differences: PostgreSQL blobs capped at 16 MB instead of a shared artifact directory; existing `tenant_admin` management and authenticated tenant reads/predictions instead of private ML roles; existing public RLS and minimal column grants for locks; idempotent registry insertion; additional stale-failure fencing; no monitoring endpoint. UI, deployment infrastructure and later ML stages are unchanged.

## Verification

- Baseline public unit suite: 50 passed before the port.
- Final production build: passed.
- Final combined suite on a fresh PostgreSQL 16/pgvector test database: 180 passed across 34 files (60 unit, 120 integration).
- Public ML integration: 14 passed, including real sklearn, signed JWT/membership/RLS, schema rejection, version behavior, stale workers, failure atomicity and migration safety.
- Real adapter checks cover all four algorithms and inspect fitted scaler state to verify training-only preprocessing.
- Full `tsc --noEmit --pretty false`: the same 22 pre-existing diagnostics in unrelated test files; no new diagnostics.
- Staged diffs and whitespace checks were reviewed before every commit.

The first signed HTTP run exposed a source permission gap for registry row locks. Public-specific minimum UPDATE column grants fixed it; dataset and pipeline revision APIs were tested through the same JWT/RLS path. No unresolved port-caused failure remains.

Deferred: other framework implementations, richer data formats, large/streaming storage, resumable checkpoints, async serving, pagination/retention, GPUs, distributed training, fine-tuning, AutoML and drift monitoring. The synchronous inference and storage limits are documented in [ML setup and APIs](ml.md).

# ML Phase 2 public port review

Reviewed private revisions `e0ec4d9` through `c43f226` (Phase 2 plus the artifact-compatibility and CI hardening) on top of the V1 baseline `d3cea57`. No private commit was cherry-picked. Shared V1 files were three-way merged against the exact V1 baseline so the public adaptations (PostgreSQL blob storage, `tenant_admin` authorization, attempt-ID failure fencing, idempotent registration, `ml/` paths) were preserved and every conflict was resolved by hand. The private repository was read-only throughout.

## Component mapping

| Private Phase 2 component | Public disposition |
| --- | --- |
| `model-catalog.ts`, `metrics.ts`, `preprocessing.ts`, `ranking.ts`, `search.ts`, `automl.ts` | Ported unchanged (framework-neutral) |
| `domain.ts`, `backend.ts` | Merged unchanged (CV spec, partitions `cv`/`cv_fold`, `cohortKey`, `evaluateDetailed`, `artifactFormats`) |
| `python/ml_backend.py` | Ported to `ml/sklearn_runner.py` (15 models, preprocessing, CV, compatibility contract) |
| `sklearn-backend.ts`, `training-processor.ts`, `persistence.ts`, `model-service.ts`, `routes.ts` | Merged into the public versions: runner path, public authorization helper, public failure fencing, no monitor route |
| `experiments.ts`, `experiment-routes.ts` | Adapted: `authorizeMl` (writes need `tenant_admin`), request-scoped `PostgresArtifactStore`, `tenant_admin` reads finalize |
| `migrations/004`, `005` | Copied as additive migrations after reading the public schema; they depend only on objects the public chain already provides (`durable_agent_api`, `reject_evaluation_evidence_update`, ML V1 tables) |
| Private roles `ml_manager`/`ml_viewer` | Omitted; a test asserts a token carrying a private-style role gets no ML write access |
| `GET /ml/endpoints/:id/monitor` | Omitted (private operational endpoint); a test asserts 404 |
| Local filesystem artifact store and `ML_ARTIFACT_ROOT` | Not used; artifacts stay in tenant-scoped `ml_blobs` (16 MB cap) |
| Tests | Ported and adapted to PostgreSQL blobs, JWT membership and RLS; tamper test disables the append-only trigger as table owner; Phase 1 fixture is the unmodified public V1 runner |
| CI | Same job extended (Python 3.13, pinned `ml/requirements.txt`, `ML_TEST_PYTHON`, 40-minute timeout) |
| Docs | Folded into `docs/ml.md`; private `ml-platform.md`/`ml-experiments.md` not copied |

One existing public V1 test inspected the pickled fitted scaler by position (`steps[0][1]`); it now reads the same scaler from the Phase 2 pipeline structure and still asserts it equals the training-partition mean.

## Verification

- Baseline before porting (fresh PostgreSQL database): migrations, readiness, build and the full suite green with a working ML interpreter: 34 files, 180 tests (60 unit, 120 integration); `tsc --noEmit` 22 diagnostics in unrelated test files (`context-planning`, `evaluation`, `orchestration`, `product` integration; `connectors`, `retrieval` unit). The repository's own `.ml-venv` has no scikit-learn installed, so all runs used an interpreter with the pinned requirements (Python 3.13.12, scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2).
- After the port (fresh database, migrations 001-005 from empty, `db:check` 5 of 5, `CI=true` so no real-backend test can skip): production build passed; 45 files, 237 tests passed (86 unit, 151 integration); `tsc --noEmit` reports the identical 22 diagnostics and none in ML files.
- The Python suites (compatibility, leakage) also pass on Linux with the pinned requirements (`python:3.13-slim`). The GitHub-hosted workflow has not been run from here.
- Phase 1 regression: all 14 V1 ML integration tests and the five V1 backend tests pass; a real V1 runner artifact is served by the Phase 2 runner.
- Package inspection: no private role, hosted, or environment content; ML Python files and migrations are the only additions. `ml/__pycache__` can appear in the tarball after local runs (gitignored; delete before `npm pack`).

