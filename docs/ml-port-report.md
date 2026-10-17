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

# ML Phase 3 (3A–3F) public port review

Ported from private revision `cc837e8`: Phases 3A–3F, the five independent-audit corrections (`cb4c086`, `6734635`, `d9778cd`, `2a5e443`, `cc837e8`) and the conformance hardening. The public base was `a9ce951`. No private commit was cherry-picked. Shared files were three-way merged against the private pre-Phase-3 revision `c43f226`, which the public Phase 2 code matched apart from its public adaptations. Every conflict was resolved by hand, and every clean merge was re-read for private names. New files were adapted to public paths, storage and authorization. The private repository was read-only throughout.

Uncommitted local work found in this checkout (an endpoint-monitoring change to `docs/ml.md`, `src/ml/model-service.ts`, `src/ml/routes.ts` and `tests/integration/ml.test.ts`) was not part of the port. It was backed up, set aside before porting and restored unchanged afterwards. It is not included in any port commit.

## Classification

| Capability | Classification | Public disposition |
| --- | --- | --- |
| ML Backend Contract v1: version, envelope, capability model, failure taxonomy, shared runner | PUBLIC-CORE | Ported (`src/ml/contract.ts`, `runner-process.ts`, `runner-backend.ts`, `ml/ml_runtime_common.py`) |
| PyTorch backend, CPU and explicit CUDA, mixed precision | PUBLIC-CORE | Ported (`ml/pytorch_runner.py`, `src/ml/pytorch-*.ts`, `ml/requirements-pytorch.txt`) |
| TensorFlow/Keras CPU backend | PUBLIC-CORE | Ported (`ml/tensorflow_runner.py`, `src/ml/tensorflow-*.ts`, `ml/requirements-tensorflow.txt`) |
| Shared neural models, preprocessing and limits | PUBLIC-CORE | Ported (`src/ml/neural-*.ts`, `ml/neural_common.py`) |
| Iterative training, durable checkpoints and resume, early stopping, best-epoch export | PUBLIC-CORE | Ported (`training-policy.ts`, processor, persistence) |
| Per-candidate backends in benchmarks and searches; catalog availability | PUBLIC-CORE | Ported |
| Backend conformance suite and cross-framework matrix | PUBLIC-CORE | Ported, run against public storage and authorization |
| Artifact and checkpoint storage | PUBLIC-ADAPTATION | Kept the PostgreSQL `ml_blobs` store (16 MB per blob). Oversized blobs now fail as `output_limit_exceeded` with a message naming the limit |
| Training and checkpoint persistence | PUBLIC-ADAPTATION | Public `fail()` keeps its stale-attempt guard and now records a category and a safe message |
| Authorization and RLS | PUBLIC-ADAPTATION | `tenant_admin` writes, authenticated tenant reads; new tables use the existing `durable_agent_api` RLS policy |
| Migration `006_ml_durable_training` | PUBLIC-ADAPTATION | Additive; depends only on objects the public chain already provides |
| Runner file names | PUBLIC-ADAPTATION | Public `ml/*_runner.py` names; docs and tests use them |
| Private roles `ml_manager` and `ml_viewer`, filesystem artifact store, `ML_ARTIFACT_ROOT` | PRIVATE-ONLY | Not used |
| Private monitoring endpoint and operational settings | PRIVATE-ONLY | Not ported (the restored local monitoring work is separate) |
| Managed GPU fleets, distributed or remote execution, managed serving, enterprise governance, proprietary strategies | PRIVATE-ONLY | Not present in the private Phase 3 code either; nothing to exclude |

No ambiguous component was found. Phase 3 contains no commercially sensitive scheduling, serving or governance logic.

## Audit corrections

| Private commit | Public behaviour | Public evidence |
| --- | --- | --- |
| `cb4c086` | Persisted failures carry only Meticule-written text | Integration test with a backend error naming a server path. The original test deleted a file, but public blobs are append-only, so the invariant is tested directly. Checked by mutation |
| `6734635` | Resumed PyTorch chunks reapply determinism settings | `check_checkpoints.py`: `restored_run_enforces_determinism` (19 checks) |
| `d9778cd` | Undeclared checkpoint formats refused; corrupt checkpoints are `invalid_checkpoint` | Durable integration tests. Corruption is applied beneath the append-only blob table and caught by the store's hash |
| `2a5e443` | A worker whose lease expired before recovery cannot write progress | Durable integration test |
| `cc837e8` | Every declared preprocessing operation and score type is verified against real output | Conformance suite. Mutations (a falsely declared operation; SVC claiming probabilities) fail it |

## Verification

**Baseline before the port.** Fresh PostgreSQL database (Docker Compose), migrations 001–005, all real-backend interpreters set:

- unit: 23 files, 86 tests;
- non-ML integration: 17 files, 106 tests;
- ML integration: 5 files, 45 tests;
- 237 tests in total, all passing;
- `tsc --noEmit`: 22 diagnostics, all in unrelated test files.

**After the port, final code, fresh database.** Migrations 001–006 applied from empty; `db:check` 6 of 6. 61 files and 438 tests passed, with none skipped:

| Suite | Files | Tests |
| --- | --- | --- |
| Unit | 30 | 123 |
| Non-ML integration | 17 | 106 |
| ML integration (lifecycle, durable, devices, compatibility, CV, experiments, authorization, PyTorch, TensorFlow, Phase 3E matrix) | 10 | 76 |
| Conformance: sklearn | 1 | 47 |
| Conformance: PyTorch | 1 | 41 |
| Conformance: TensorFlow | 1 | 40 |
| Conformance: cross-backend | 1 | 5 |

The production build passes. `tsc --noEmit` reports the identical 22 diagnostics and none in ML files.

**Upgrade from Phase 2.** A Phase 2 database (001–005, populated by the Phase 2 suite) migrated to 006 with every ML row count unchanged, and `db:check` reported 6 of 6. In a second database, the Phase 2 code (`a9ce951`) trained a real scikit-learn model on the Phase 2 schema. After upgrading to 006, the Phase 3 code registered, promoted and served it (x = 10 predicts 21.01).

**Mutation checks**, each restored afterwards. Each made its test fail:

- storing raw error text;
- an unclassified storage-limit error;
- a neural backend declaring a preprocessing operation its runner ignores;
- SVC claiming probabilities.

**Interpreters.** The repository's own `.ml-venv` has no scikit-learn installed, so all runs used environments with the pinned requirements: Python 3.13.12, scikit-learn 1.7.2, PyTorch 2.9.1 (CPU), and TensorFlow 2.20.0 with Keras 3.15.1.

**Not verified here:**

- Real CUDA execution: PyTorch was a CPU build, so the CUDA checks took their explicit `device_unavailable` refusal branch.
- The hosted GitHub Actions workflow was not run from this machine.
- The Phase 3E matrix and the cross-backend pairs that include TensorFlow run only where all three frameworks are installed together. They passed locally and are not part of hosted CI.

# ML Phase 4 (M1–M9) public port review

The port comes from private revision `85f81bf`, the verified end of Phase 4 (M1–M9). The private report for that revision records 846 passing tests and 3 CI-only skips, with no hosted CI or GPU run. Those are private results; the public results are below.

The public base was `1b4118f`, which already contained the Phase 3 port and the public endpoint-monitoring commit. The checkout was clean, so no local work had to be set aside. The private repository was read-only throughout.

The method was the same as for Phase 3, applied per milestone:

- Every private file changed in the milestone was translated to public names: `ml/*_runner.py`, `ml/requirements.txt`, `docs/ml.md`, `authorizeMl`, `tenant_admin` and public `monitoring()`.
- Each file was then three-way merged against the private revision before that milestone, so the public adaptations survived. Conflicts were resolved by hand.
- New integration tests were moved from the filesystem artifact store to `PostgresArtifactStore`.
- Each milestone was typechecked, and its unit and integration files were run before it was committed.

The translation missed two runner names. Tests and the leakage scan found them, and they are fixed (see below).

## Classification

| Capability | Classification | Public disposition |
| --- | --- | --- |
| Contract v2 and vocabulary (M1): the task registry, `text`/`timestamp`/`image`/`audio` columns, media and timestamp validation in both languages, the metric registry and refusal codes | PUBLIC-CORE | Ported. v1 requests keep their meaning |
| Native JAX MLPs with durable chunked training (M2) | PUBLIC-CORE | Ported, CPU only |
| Native XGBoost and LightGBM (M3) | PUBLIC-CORE | Ported |
| Hugging Face Transformers for text and vision (M4); wav2vec2 audio classification and CTC speech recognition (M8) | PUBLIC-CORE | Ported. The manifest, hash, allowlist, offline and `trust_remote_code=False` rules are unchanged |
| NumPy/SciPy audio features (M5) | PUBLIC-CORE | Ported |
| pgvector embedding indexes and similarity, migration 007 (M6) | PUBLIC-ADAPTATION | Routes use the public request-scoped store. Writes need `tenant_admin`; members read and search. A new public JWT/RLS HTTP test covers this |
| Forecasting, the LightGBM forecaster and anomaly detection (M7) | PUBLIC-CORE | Ported |
| Artifacts and checkpoints for every new backend | PUBLIC-ADAPTATION | Kept in PostgreSQL `ml_blobs` at 16 MB per blob, unchanged. Oversized artifacts fail as `output_limit_exceeded` |
| `ML_TRANSFORMERS_MODEL_ROOT` | PUBLIC-CORE | Kept. It is a read-only library of operator-provisioned models, not artifact storage, and must be an existing absolute directory at startup |
| Configuration | PUBLIC-ADAPTATION | One flag pair per backend, with the public platform-aware default interpreter (`python` or `python3`) |
| CI, `.env.example`, packaging and docs (M9) | PUBLIC-ADAPTATION | See below |
| Private roles, the filesystem artifact store, `ML_ARTIFACT_ROOT` and the private `monitor()` route | PRIVATE-ONLY | Not ported (as in Phases 2 and 3) |
| Private planning and verification documents: `phase-4-handoff.md`, `phase-4-report.md`, `ml-platform.md`, `ml-experiments.md` | PRIVATE-ONLY | Not copied. The public facts are written into `docs/ml.md` |
| Managed GPU fleets, hosted orchestration and enterprise governance | — | Not present in private Phase 4 either, so there was nothing to exclude |

Nothing was deferred: every implemented Phase 4 capability is ported.

## Public-specific changes

- **Embedding routes.** `src/ml/embedding-routes.ts` builds `new PostgresArtifactStore(c.get("database"))` per request, like the other public ML routes.
- **Configuration.** `src/ml/configuration.ts` keeps the public style and default interpreter.
- **Embedding HTTP test.** `tests/integration/ml-embeddings.test.ts` adds a test through signed JWT membership:
  - a member gets 403 on create and build, and 200 on search and similarity;
  - another tenant gets 404 on get, search and similarity, and an empty list.
- **Monitoring assertions.** Private `monitor()` assertions map to public `monitoring()`, with `failures` renamed to `failed`.
- **Roles in tests.** A private-style viewer becomes a member with no roles.
- **Legacy guard.** `tests/python/legacy/phase3_neural/` holds frozen copies of `ml/pytorch_runner.py` and `ml/neural_common.py` from public `1b4118f`, checked against their public blob ids, so the compatibility guard tests the public Phase 3 runner.
- **Anomaly check import.** `tests/python/check_anomaly.py` imports the scikit-learn runner through the `ml_backend` alias in public `support.py`.
- **Packaging.** `package.json` ships `ml/*.py` and `ml/*.txt` instead of the whole `ml/` directory. Local `__pycache__` therefore no longer reaches the tarball, which was previously a manual cleanup step. It also adds `docs/ml-backend-contract.md`, which `docs/ml.md` links to.
- **Runner names.** Two names left over from the private sources were fixed: the runner path in the GBDT check and the runner path in the forecasting conformance fixture.

## Capability matrix (public)

| Backend | Models | Tasks | Inputs | Device | Artifact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| scikit-learn | Phase 2 catalog, plus isolation forest, LOF and robust covariance | classification, regression, anomaly detection | tabular | CPU | `sklearn-pickle-v1` (anomaly: schema 2) | Implemented and verified |
| PyTorch (Phase 3) | `mlp_classifier`, `mlp_regressor` | classification, regression | tabular | CPU; CUDA when requested | `pytorch-state-dict-v1` | CPU verified; CUDA not hardware-verified |
| TensorFlow/Keras (Phase 3) | `mlp_classifier`, `mlp_regressor` | classification, regression | tabular | CPU | `tensorflow-variables-v1` | Implemented and verified |
| JAX | `mlp_classifier`, `mlp_regressor` | classification, regression | tabular | CPU (CUDA refused by design) | `jax-params-v1` / `jax-checkpoint-v1` | Implemented and verified |
| XGBoost | classifier, regressor | classification, regression | tabular; missing numbers as NaN | CPU | `xgboost-json-v1` | Implemented and verified |
| LightGBM | classifier, regressor, forecaster | classification, regression, forecasting | tabular; timestamp for the forecaster | CPU | `lightgbm-text-v1` | Implemented and verified |
| Transformers | text, image and audio classifiers; text and image embedders; two frozen-encoder heads; speech recognizer | classification, embedding, transcription | one text, image or audio column | CPU, fp32 | `transformers-reference-v1` | Execution path verified with tiny offline models; real-model quality not measured |
| Audio | `audio_features`, `audio_mfcc_classifier` | embedding, classification | one audio column | CPU | `audio-npz-v1` | Implemented and verified |
| Forecasting | seasonal naive, ridge | forecasting (univariate) | timestamp and numeric target | CPU | `forecast-npz-v1` | Implemented and verified |
| Embedding indexes | (any embedding endpoint) | exact search, pairwise similarity | embedding endpoint rows | PostgreSQL/pgvector | none (vectors in tables) | Implemented and verified |

Not supported by design:

- encoder fine-tuning and PEFT/LoRA;
- beam search;
- multivariate or covariate forecasting;
- approximate vector indexes;
- model downloads;
- GPU execution for the Phase 4 backends.

## Migration evidence

**Fresh database.** Migrations `001`–`007` applied from empty, and `db:check` reported 7 of 7. The number 007 was free in the public chain. The migration depends only on objects the chain already provides: `durable_agent_api`, `reject_evaluation_evidence_update` and the `vector` extension created in 001.

**Populated upgrade.** The public Phase 3 code (`1b4118f`) first built a database with migrations 001–006 and left it in this state:

- a real scikit-learn ridge model, trained, registered, promoted and served (x = 10 predicts 21.010737758030174);
- a real PyTorch job with an epoch-2 checkpoint whose lease expired, recovered to QUEUED.

The database was then migrated to 007 and checked with the Phase 4 code. All five checks passed:

- the row counts of all 18 ML tables were unchanged;
- the embedding tables were empty;
- the Phase 3 endpoint returned the identical prediction;
- the Phase 3 checkpoint resumed from epoch 2 to completion, with epochs 1–6 each recorded once;
- the resumed model registered and served.

`db:check` reported 7 of 7.

## Verification

**Baseline before the port** (public `1b4118f`, fresh database, migrations 001–006, every real-framework interpreter set, one file per process, sequentially):

- 61 files: 435 passed, 2 failed, 1 skipped.
- The two failures were 15-second hook and test timeouts in `ml-durable.test.ts`, caused by port work running alongside. Re-run alone, that file passed 10 of 10.
- The skip is the CI-only TensorFlow runtime check.
- The effective baseline is therefore 437 passed plus 1 CI-only skip, which matches the 438 recorded after Phase 3.
- `tsc --noEmit` reported 22 diagnostics, all in unrelated test files.

**After the port** (final `main`, fresh database, migrations 001–007, every interpreter set, one file per process, sequentially): 101 files, 855 passed, 0 failed, 4 skipped.

| Suite | Files | Passed | Skipped |
| --- | ---: | ---: | ---: |
| Conformance (all backends, the three anomaly detectors, audio, forecasting, eight Transformers models, cross-backend) | 23 | 432 | 0 |
| ML integration (lifecycle, durable, devices, compatibility, CV, experiments and their authorization, matrix, PyTorch, TensorFlow, JAX, GBDT, Transformers, audio, forecasting, embeddings) | 16 | 106 | 0 |
| Non-ML integration | 17 | 106 | 0 |
| ML unit (including the real-framework check scripts) | 32 | 161 | 4 |
| Non-ML unit | 13 | 50 | 0 |

- **Skips.** The four skips are CI-only runtime-pin checks (TensorFlow, JAX, XGBoost/LightGBM, Transformers). They run only inside their own CI jobs. Every real-framework test ran.
- **Exit codes.** One file, `ml-embeddings.test.ts`, passed 6 of 6 but exited 1. The cause was the pre-existing tinypool worker-teardown error (`emitter.removeListener is not a function`) after its tests finished. Re-run alone, it passed 6 of 6 with exit code 0. Every other file exited 0.
- **Phase 1–3 regression.** All Phase 1–3 tests still pass, including:
  - the unmodified Phase 1 adapter artifact;
  - Phase 3 PyTorch artifacts and checkpoints written by a frozen copy of the public Phase 3 runner (`ml-legacy-neural`).
- **Build and types.** The production build passes. `tsc --noEmit` reports the same 22 diagnostics as the baseline after normalizing source locations, and none in ML files.

**Interpreters** (local venvs with the pinned requirements; Python 3.13.12 unless noted):

| Interpreter | Packages |
| --- | --- |
| scikit-learn, audio and forecasting | scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2, pandas 3.0.6 |
| PyTorch | Python 3.14, torch 2.9.1 CPU |
| TensorFlow | TensorFlow 2.20.0, Keras 3.15.1 |
| JAX | jax and jaxlib 0.11.2 |
| GBDT | XGBoost 3.4.1, LightGBM 4.7.0 |
| Transformers | torch 2.9.1 CPU, transformers 5.17.0 |

Transformers tests used tiny random models generated offline; nothing was downloaded.

## CI and packaging

- **CI coverage.** The CI workflow keeps the public `test` job and adds `jax`, `gbdt` and `transformers` jobs beside `tensorflow`. Every Phase 4 conformance file is listed in a job, or runs in `test` through `npm test` with the interpreter that job sets. This includes the LightGBM forecaster and the scikit-learn/GBDT cross-backend pairs, which the private M9 review had found missing.
- **CI-only pairs.** Cross-backend pairs that need TensorFlow, JAX or Transformers together with another framework run only locally, because CI has no shared environment for them. The same applies to the Transformers text-embedder index test.
- **Package contents.** `npm pack --dry-run` lists all 17 Python modules and 6 requirements files under `ml/`, all 7 migrations, `docs/ml.md` and `docs/ml-backend-contract.md`. It contains no tests, no `__pycache__`, no venvs and no environment files.
- **Installation.** Self-hosted installs run the API and worker from the package root, because the backends resolve `ml/*_runner.py` relative to the working directory. Optional dependencies stay optional: with a backend disabled, its models report `available: false` and requests for them are refused.
- **Docker.** The Docker image remains Node-only. Containerized ML still needs operator-provided Python environments, as documented.

## Security and compatibility review

One focused review covered the public-specific trust boundaries. It found no high or medium issues.

Checked and clean:

- **Tenant isolation.** Every embedding SQL statement filters by tenant and runs on the request's tenant session (`SET LOCAL ROLE durable_agent_api`), and artifact reads are bound to the caller's tenant.
- **Authorization.** Writes need `tenant_admin`, and reads call `authorizeMl`. No private roles appear in `src`.
- **Model loading.** Model names are pattern-checked and must resolve directly under the root. Only allowlisted files are loaded, links and `auto_map` are refused, and files are copied while hashing. Loading uses `local_files_only`, `trust_remote_code=False` and offline mode. The model root is never recorded.
- **Deserialization.** No pickle is used except the existing scikit-learn format. NumPy archives are loaded with `allow_pickle=False`, boosted trees are JSON or text, and PyTorch uses `weights_only=True`.
- **Failure text.** Failure records hold only Meticule-written text.
- **Configuration.** Startup refuses a relative or missing model root.

Low findings:

- **Commit messages** named the private source and one private hash. The nine local commits were reworded before they reached `main`. The trees are identical.
- **Media row caps (fixed after the port).** Index builds (256 rows per step) and pairwise similarity (64 rows per side, validated per side) were not held to the per-request row caps and media budget that prediction uses. Memory stayed bounded, and the source behaved the same way. The fix was made at the source first (private `745cd55`) and then ported here:
  - **Build steps** now embed the longest prefix that fits the embedding row cap for the model's feature types and the prediction media budget.
  - **Pairwise similarity** treats both sides as one request: a combined row cap checked before decoding, one media budget and one runner call.

  The tests cover a build step cut at 15 of 16 eight-second clips and a refused 9 + 8 or 128-second pairwise request. The public HTTP/JWT test also asserts the 16-row step and a member's 409. With the old service the new test fails.

A full-diff scan found no private repository names, developer paths, private planning documents, private commit references, private roles, filesystem-store settings or private runner file names.

## Not verified

- **GPU.** No GPU was tested. Every Phase 4 backend is CPU-only, JAX refuses CUDA, and the Phase 3 PyTorch CUDA path is still not hardware-verified.
- **Hosted CI.** The GitHub Actions workflow was not run from here. The workflow was edited, but no hosted pass is claimed.
- **Model quality.** Real pretrained-model quality was not measured: tests use tiny random offline models.

## Remaining limitations

- **Forecasting** is univariate and recursive.
- **Transformers** encoders are frozen, and speech recognition is greedy CTC.
- **Embedding search** is exact over at most 5,000 rows.
- **Storage.** Every blob is capped at 16 MB.
- **Inference** is synchronous.
- **Docker.** The image has no Python.

## Verdict

The Phase 4 (M1–M9) selective port is complete. Every implemented Phase 4 capability works in the public edition through its real framework, on public PostgreSQL storage and with public authorization. Contract v1 and Phase 1–3 behavior are preserved. The fresh and populated-upgrade migration paths pass. The full sequential regression passed with no failures. The work is nine feature and integration commits plus this report, all local; nothing was pushed.
