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

# ML Phase 5 (MLOps) public port review

The port comes from private revision `1631f40` on the private `main` branch. It is the verified end of Phase 5:

- milestones A `2549d38`, B `928dcaf`, C `df1eda7`, E `05f08a8`, F `e570c07`, G1 `0492932`, G2 `1f4b438` and D `93dd5da`;
- the audit corrections `2a2b58c`, including migration 016;
- the fixture isolation fix from `ce9a57a`.

The two commits after that (`dc2802b`, `1631f40`) change only the private verification report.

The private report for that revision records 896 passing tests across 112 files. Those are private results; the public results are below.

The public base was `f34e93e`: the Phase 4 port plus the port of the embedding row-cap fix. The checkout was clean, and `main` matched `origin/main`. The work was done on the branch `feature/ml-phase5-public` in a separate worktree. The private repository was read-only throughout.

The method was the same as for Phase 4. For each private milestone, every file it changed was translated to public names:

- runner paths: `ml/*_runner.py`, `ml/requirements*.txt`;
- `authorizeMl` instead of the private role check;
- a request-scoped `PostgresArtifactStore` instead of the configured filesystem store;
- `tenant_admin` instead of the private manager role.

Each file was then three-way merged against the private revision before that milestone. Conflicts were resolved by hand, and every clean merge was scanned for private names. Each milestone was typechecked, and its targeted tests were run before it was committed.

## Commits

| Public commit | Content |
| --- | --- |
| `0925cab` | A: experiment metadata, filtered listing, derived status, run history, comparison, reproduction, lineage (migration 008) |
| `834626c` | B: MLflow export and metadata import (migration 009) |
| `d21bc24` | C: Optuna studies over durable jobs (migration 010) |
| `71c76ff` | E: registry policies, approvals, separation of duties, lifecycle events (migration 011) |
| `047112d` | F: ONNX export, parity verification and ONNX Runtime inference (migration 012) |
| `54d6a90` | G1: deployments with version-pinned revisions (migration 013) |
| `ca7745c` | G2: reference profiles, drift, ground-truth feedback, deployment monitoring (migration 014) |
| `ba9dc5d` | D: Matplotlib reports and the `mlops` CI job (migration 015) |
| `636a852` | The private audit corrections (migration 016) and the fixture isolation fix |
| `4631a42` | Public JWT membership and RLS test of the MLOps routes |
| `a163f42` | Public documentation (`docs/ml-mlops.md`, `docs/ml.md`, README, package contents) |
| `6d5a21c` | Fixes from the public review (see Security review) |

## Classification

| Capability | Classification | Public disposition |
| --- | --- | --- |
| Experiment tracking, run history, comparison, reproduction and lineage | PUBLIC-CORE | Ported. `POST /ml/experiments` still accepts `{name}`. Lineage checks blob bytes through the public store |
| MLflow export and metadata import | PUBLIC-CORE | Ported. Operator-configured server only; no model artifacts move. Foreign-tenant runs are refused on import. Direct access to a shared server is not isolated, and the docs say so |
| Optuna studies (TPE, random) over durable jobs, worker sweep | PUBLIC-CORE | Ported. No second scheduler; no pruning |
| Registry governance: policies, thresholds, approvals, readiness, lifecycle events, governed endpoints | PUBLIC-ADAPTATION | Approvers are a `tenant_admin` other than the registrant. Service identities can never approve. Older versions recover their registrant from the audit log. A version without a registrant cannot be approved. Enforced by the service and the database |
| ONNX export (scikit-learn, numeric features), parity, runtime gate | PUBLIC-ADAPTATION | ONNX bytes are PostgreSQL blobs. An export over the 16 MB limit is recorded as `UNSUPPORTED` (`output_limit_exceeded`) |
| Deployments: revisions, validation, activation, update, rollback, deactivation | PUBLIC-CORE | Ported. Synchronous in-process serving through the existing endpoints |
| Drift (PSI, Jensen-Shannon), feedback, deployment monitoring, retention | PUBLIC-ADAPTATION | Merged with the existing public endpoint summary (`GET /ml/endpoints/:id/monitoring` gains `p50_latency_ms` and `failures_by_category`). The deployment monitor is `GET /ml/deployments/:id/monitoring` |
| Matplotlib reports | PUBLIC-ADAPTATION | Report images are PostgreSQL blobs |
| Configuration, CI, packaging | PUBLIC-ADAPTATION | One optional MLOps interpreter with four feature flags. Public default interpreter. The `mlops` CI job. The package ships the four runners, `ml/requirements-mlops.txt` and `docs/ml-mlops.md` |
| Private manager and viewer roles, filesystem artifact store, `ML_ARTIFACT_ROOT`, the private endpoint `/monitor` route | PRIVATE-ONLY | Not ported. A test asserts that both `/monitor` routes return 404 |
| Private planning and verification documents (`phase-5-report.md`, `ml-platform.md`) | PRIVATE-ONLY | Not copied. The public facts are in `docs/ml-mlops.md` |
| Managed serving, hosting, fleets, traffic splitting, canary rollout, autoscaling, GPU serving | — | Not present in private Phase 5 either, so there was nothing to exclude |

Nothing implemented in private Phase 5 was deferred.

## Public-specific changes

- **Route configuration.** `MlRouteConfiguration` drops the private `artifacts` key, and `createApp` accepts it, so the existing callers that pass `{enabled, backends}` keep compiling.
- **Registration is idempotent.** Public registration is idempotent per attempt, which the private repository does not enforce. Three private tests registered the same attempt twice to get a second version; they now train a second attempt.
- **Tests on PostgreSQL blobs.** Tests that deleted artifact files now remove the blob beneath the append-only trigger, as table owner.
- **Missing CI coverage.** The full-catalog ONNX parity check (`tests/unit/ml-onnx-runner.test.ts`) is listed in the `mlops` job. The private job did not list it. The private repository has since added it in its own commit.
- **Translation mistake.** Translating `python/` to `ml/` also rewrote `tests/python/check_onnx.py` to a non-existent `tests/ml/` path. The test found it, and it is fixed.

## Migrations

**Numbering.** The numbers 008–016 were free in the public chain. Every migration depends only on objects the public chain already provides: `durable_agent_api`, `reject_evaluation_evidence_update`, `tenant_memberships`, `audit_events` and the ML tables.

**Tenant isolation.** Each new table has the `tenant_isolation` RLS policy. Trigger functions are `SECURITY INVOKER`. `ml_prune_monitoring` is revoked from `PUBLIC`.

**Fresh database.** Migrations 001–016 applied from empty, and `db:check` reported 16 of 16.

**Populated upgrade.** The public Phase 4 code (`f34e93e`) first built a database with migrations 001–007, in this state:

- a real scikit-learn ridge model registered by user A, promoted and served (x = 10 predicts 21.010737758030174);
- a second version registered by user B and left `REGISTERED`.

The database was then migrated to 016 and checked with the Phase 5 code. All 21 assertions passed:

| Area | Assertions |
| --- | --- |
| Data preservation | All 20 ML tables kept their row counts. All three Phase 4 blobs read back and verified |
| Registrants | Backfilled from the audit log: A for the first version, B for the second. No events were invented |
| Serving | The old endpoint returns the identical prediction. Its reference profile was built on first use |
| Tracking and lineage | Experiment status is derived as `COMPLETED`. Lineage is clean with `verify=true` |
| Governance | A new approval policy holds the older versions. B cannot approve their own version, and A can. B's version reached `READY`. The tightened policy refuses the already-`READY` first version as a deployment revision |
| Deployments | The approved older version deploys, activates and serves |
| Database guards | Another tenant sees 0 rows in 10 ML tables under the API role. An invalid deployment transition, self-approval and changing the registrant are all refused by the database |
| Configuration | No filesystem store is configured |

`db:check` reported 16 of 16.

## Verification

**Baseline** (public `f34e93e`, migrations 001–007, every real-framework interpreter set, one file per process, sequentially):

- 101 files: 864 passed, 0 failed, 1 skipped; every process exited 0.
- The skip was the CI-only Transformers pin check, gated on `ML_TRANSFORMERS_JOB`. Rerun with that flag, its file passed 2 of 2.
- The run was interrupted twice by host memory pressure and resumed. Interrupted files were rerun, never counted as passed.

**After the port** (final branch `6d5a21c`, fresh database, migrations 001–016, every interpreter set):

- Set: `ML_TEST_MLOPS_PYTHON` with `mlflow==3.16.1` for a real local tracking server, and every CI job flag including `ML_TRANSFORMERS_JOB`.
- 114 files, one per process, sequentially: 917 passed, 0 failed, 0 skipped. Every process exited 0.

| Suite | Files | Passed |
| --- | ---: | ---: |
| MLOps integration (tracking, MLflow, Optuna, governance, ONNX, deployments, monitoring, reports, JWT/RLS) | 9 | 34 |
| MLOps unit (drift, report curves, MLOps runtime pins, full-catalog ONNX parity) | 4 | 10 |
| Conformance (all backends) | 23 | 445 |
| Phase 1–4 ML integration | 16 | 107 |
| Phase 1–4 ML unit | 32 | 165 |
| Non-ML integration | 17 | 106 |
| Non-ML unit | 13 | 50 |

The increase over the baseline:

- **44 tests:** the new MLOps tests.
- **9 tests:** the CI-gated Transformers checks, registered only when the flag is set. That is one per Transformers conformance file, plus the pin check.

**Interrupted file.** One file, `transformers-text-classifier` conformance, failed once with a refusal code of `undefined`. Its raw-envelope helper has a 120-second timeout, and under host memory pressure the runner took longer than that just to start. Phase 5 changes nothing on that path. Rerun alone, the file passed 11 of 11, and that rerun is the recorded result.

**Mutation checks.** Each public adaptation was broken on purpose, and each break made its test fail:

- removing the service-identity check from the service;
- removing it from the database trigger;
- removing the oversize-ONNX outcome;
- restoring the old study sweep.

**Build and types.** The production build passes. `tsc --noEmit` reports the same 22 diagnostics as the Phase 4 baseline, and none in Phase 5 files.

**Interpreters** (local venvs; Python 3.13.12 unless noted):

| Interpreter | Packages |
| --- | --- |
| MLOps | onnx 1.23.0, onnxruntime 1.30.0, skl2onnx 1.20.0, optuna 5.0.0, matplotlib 3.11.2, mlflow 3.16.1, with the scikit-learn pins of `ml/requirements.txt` |
| Others | As in Phase 4 |

## CI and packaging

- **CI.** The workflow has a new `mlops` job beside `test`, `tensorflow`, `jax`, `gbdt` and `transformers`. It installs `ml/requirements-mlops.txt` and `mlflow==3.16.1`, and runs every MLOps file, including the full-catalog ONNX check. `ML_MLOPS_JOB` makes a missing interpreter fail the job. The YAML parses.
- **Package contents.** `npm pack --dry-run` lists 206 files:
  - 28 under `ml/`, including the four MLOps runners and `requirements-mlops.txt`;
  - migrations 001–016;
  - `docs/ml-mlops.md`.

  It contains no tests, caches, venvs, environment files or private documents.
- **Optional features.** Every MLOps runner feature is off by default. A disabled feature answers 503 `ml_feature_not_configured`.

## Security review

**Process.** One focused independent review covered the public diff. It reproduced its database findings in a scratch database. It found no critical or high issues, and it found no private names, paths, roles or commit hashes in code, docs or commit messages.

**Fixed (`6d5a21c`):**

- **No registrant (low).** A version with no recorded registrant could be approved by anyone. The service and the trigger now refuse it.
- **Sweep starvation (low).** The worker sweep only ever looked at the 10 oldest running studies. It now picks only studies it can move (cancelled, past their deadline, or with no open trial), in random order. A test covers an older busy study.

**Documented, not changed:**

- **Approval is opt-in per registry entry (medium, inherited design).** Any `tenant_admin` can create an entry without a policy and promote a version there.
- **Training-value summaries (low, inherited).** Drift references summarize training values: numeric decile edges and the 20 most frequent values of string and boolean features. Every member can read them.
- **Identity-provider dependence.** Meticule cannot tell when one person holds two identities.

**Declined: approvers without a membership row (low).** The trigger also accepts an approver with no membership row. JWT authentication already requires an active membership. In development-header mode any identity can be asserted, so such a check would add nothing there.

**Checked and clean:**

- RLS on all 14 new tables;
- deserialization (pickle only for scikit-learn artifacts under the compatibility check; `allow_pickle=False` everywhere else);
- the ONNX gate;
- download of `VERIFIED` exports only;
- MLflow token isolation;
- Optuna cancellation and locking;
- deployment triggers;
- owner-only retention;
- inert SVG serving.

## Not verified

- **Hosted CI.** The GitHub Actions workflow, including the new `mlops` job, has not been run from here.
- **GPU.** No GPU was tested. Phase 5 adds no GPU code, and ONNX Runtime uses the CPU provider.

## Remaining limitations

- **ONNX** covers scikit-learn classification and regression over numeric features only.
- **Optuna** has no pruning; samplers are TPE and random.
- **Serving** is synchronous in-process inference: no separate serving process, traffic splitting, canary or autoscaling.
- **Drift** uses binned PSI and Jensen-Shannon on tabular features; the PSI bands are conventions, not significance tests.
- **MLflow** uses one server per deployment and moves metadata only.
- **Reports** draw ROC and precision-recall only where scores exist, and score at most 5,000 rows.
- **Storage.** Every blob, ONNX export and report image is limited to 16 MB.
- **Retention.** Monitoring profiles are pruned only when an operator runs `ml_prune_monitoring`.

## Verdict

The Phase 5 selective port is complete. Every implemented private Phase 5 capability works in the public edition:

- through its real framework and runner;
- on public PostgreSQL blob storage;
- with public JWT membership, `tenant_admin` writes and RLS.

The private audit corrections are included, and the public review's confirmed findings are fixed. Earlier behavior is preserved, and the fresh and populated-upgrade migration paths pass. The full sequential regression passed with no failures and no skips.

## Selective Phase 6 Public Core port

This is an adapter-focused selection, not a repository synchronization. Starting Public branch: `main`,
commit `f8d1edb5b8d21e020ef3fe723a86c0657eff3650`. The source repository remained read-only. Existing Phase 1–5 implementations were inspected before
editing: Public already had datasets/preprocessing, nine backend IDs, durable neural training, experiments,
registry/serving, monitoring and its own MLOps/authorization architecture. These were preserved.

The [public/private capability matrix](ml.md#phase-6-public-core-local-adapters) records the selected boundary.
Added: local Llama LoRA and optional single-GPU NF4/FP4 QLoRA, safe adapter/checkpoint formats, approved local model
provisioning and registered adapter inference. Distillation/QLoRA distillation, merged exports, RL/DPO and Ray remain
Private. Distillation would require additional teacher-version and merged-publication lifecycle contracts; DQN/PPO
share a broader RL runtime that was not selected. These are deliberate product-scope decisions, not claims that the
algorithms are impossible to run locally. No private policy, governance, distributed transport or migration was copied.

### Adaptation and compatibility

- Reused Public's `NeuralRunnerBackend`, task/catalog validation, epoch loop, worker leases, cancellation, registry,
  explicit READY promotion, inference, monitoring and tenant-scoped transactional PostgreSQL blob store.
- Extended contract v2 with `tokens`, `language_modeling` and token CE/perplexity/accuracy; older vocabulary and
  artifact contracts remain valid. No schema migration was needed; all 16 existing migrations applied to a new
  disposable local PostgreSQL/pgvector database.
- Extracted the narrow shared LoRA/quantization/provisioning primitives and adapter-only runner. Distillation,
  merging and Ray code are absent. Quantized checkpoint restoration includes the current runtime/configuration
  compatibility checks. Numeric accumulation and RNG validation audit corrections are included.
- Kept the existing manifest/copy/hash boundary; only safe local Llama safetensors assets and matching tokenizers
  are loaded. No model-supplied Python, unsafe pickle ingestion, arbitrary imports or network/model downloads.
- Added an opt-in `ML_PEFT_ENABLED` / `ML_PEFT_PYTHON` configuration and separate pinned optional Python requirements.
  JavaScript dependencies and lockfile are unchanged. API/workers still start without Python when ML is disabled.
- Added a parent-death guard for the adapter interpreter, including Windows venv launchers. Real cancellation and
  environment-secret isolation tests pass. Public storage remains transactional blobs, not a copied filesystem store.

### Fresh numerical evidence

The fresh pre-edit baseline passed production compilation and reported 22 full-project TypeScript diagnostics.
Three targeted baseline unit files passed 23 cases; six real-Python wrappers were skipped because their interpreter
variables were unset in that initial command. They run in the final enabled-runtime unit sweep below. Historical
Phase 5 results above are preserved and are not counted as fresh Phase 6 verification.

Tests generate small models and counting-token datasets locally; no private dataset or historical private result
is published as Public evidence. CPU uses Python 3.13.12, Torch 2.9.1+cpu, Transformers 5.17.0, PEFT 0.21.2,
Accelerate 1.15.0 and Safetensors 0.8.0. CUDA uses Torch 2.9.1+cu128 / CUDA 12.8 and bitsandbytes 0.50.2 on an
RTX 4060 Laptop GPU with 8,188 MiB VRAM. These are bounded tiny-model measurements, not large-model validation.

| Check | Actual result |
| --- | --- |
| CPU LoRA foundation | 27 named checks pass; frozen base 21,664 parameters, trainable adapter 1,024, total 22,688 |
| CPU runner / fresh-process continuation | 28 checks pass; adapter and optimizer maximum difference 0; dropout/sampler RNG restored |
| Accumulation and checkpoint RNG audit | 20 checks pass, including unequal token counts and partial accumulation groups |
| Provisioned model/tokenizer and hostile assets | 23 checks pass, including real training, recovery and inference |
| Quantization matrix | 56 checks pass: 16 explicitly injected validation checks and 40 real GPU checks across all eight configurations |
| CUDA checkpoint continuation/inference/refusals | Eight checks pass, including exact full-state continuation, incompatible quantization-runtime rejection and corrupt checkpoint rejection |

The direct CPU foundation trained 200 optimizer steps: CE **4.27402735 → 0.06221641**; all eight adapter tensors
changed, base weights remained unchanged and no frozen tensor had a gradient. Separate-process adapter inference
matches within `1e-6`. The six-epoch runner fixture achieved validation CE **0.60538**, perplexity **1.83194**,
accuracy **0.89855**, and held-out test CE **0.57336**, perplexity **1.77422**, accuracy **0.91860**. Discarding AdamW
moments or dropout RNG changed the continuation, so recovery is tested beyond successful process exit.

The bounded NF4/bf16/double-quantized recovery fixture trained four optimizer steps over two epochs: training CE
**3.84538124 → 1.87532334**. Uninterrupted/resumed adapter, optimizer and RNG tensors were identical; export and
inference passed. Eight-configuration peak Torch allocation was **67,745,280 bytes (64.61 MiB)**, excluding
CUDA context/driver memory and other processes. This does not establish concurrent capacity or full-ceiling model fit.
Quantization leaves the mathematical CE objective unchanged; there is no teacher or distillation claim.

### Lifecycle verification record

The first six-case PostgreSQL/blob integration run had two passes and four failures. The interruption fixture
incorrectly drained subsequent worker ticks, allowing automatic recovery before inspecting the abandoned state.
The production worker behaved correctly. The helper now executes one tick at the injected lease-loss boundary;
all intended checkpoint/publication assertions remain. A targeted four-case rerun passed CPU, FP4 and corrupt
checkpoint handling; NF4's exact artifact hash and all metric values matched, but a test incorrectly compared
unordered SQL row positions. Metrics are now compared by partition/name with exact values, without changing a
numerical tolerance or learning threshold. Original failed reports are retained.

Cancellation and provisioned-base dependency passed in the initial run: hiding the approved base produces FAILED
with no output; restoring it serves again. The passing CPU/FP4 recovery cases additionally verify actual learning,
committed checkpoints, lease replacement, no stale checkpoint publication, immutable adapter identity, explicit
registration/promotion, independent endpoint inference, role refusals, cross-tenant registry/artifact/inference
isolation and tampered-artifact rejection. The final NF4 case passed in 320.89 seconds with exact artifact/value
equality and registered inference. Across these reports, all six original lifecycle cases now have passing evidence.
The first rerun selected four cases (three passed, one failed, two filtered); the final rerun selected NF4 only
(one passed, five filtered). These filtered cases are not missing hardware/prerequisite skips or additional passes;
their passing evidence is in the earlier reports.

The first complete unit run exercised all installed optional interpreters: **234 passed, one failed, zero skipped**
across 53 files (492.35 seconds). The failure was the older exhaustive Python metric-reference table, which still
listed only the pre-token metrics. It now includes the three declared language-model metrics and their label-free
semantics while retaining every original Phase 3/4 assertion. The corrected four-file targeted run passed all
16 cases. The initial failed unit report is retained.

A controlled subprocess regression then reproduced a cleanup defect: after the guarded interpreter was cancelled,
its temporary fixture asset remained. Adapter calls now receive an owned per-call temporary directory, and Node
waits for the aborted interpreter's pipes to close before removing it. The same regression passes, verifies the
actual interpreter PID is gone and its scratch directory is absent, and retains the original failing report.
This narrow runner option is opt-in for adapters; existing backend behavior and storage contracts are unchanged.
The two short real CPU/CUDA training-and-registered-inference cases passed (70.86 and 115.35 seconds) and left
no new per-call scratch directories. The fresh final complete unit run then passed **235 tests across 53 files,
zero failures and zero skips**, in 474.72 seconds. All available optional interpreter/CI assertions were enabled.
The selected regression batch exercised 173 cases across 28 files: **172 passed, one failed, zero skipped**
(1,626.92 seconds). Its monitoring test observed 108 rows after four completed 36-row batches instead of 144.
The existing service used an application-clock millisecond upper bound against PostgreSQL microsecond timestamps,
which could omit the latest completed batch. A real-PostgreSQL regression explicitly injected an application clock
rounded to an already stored profile timestamp: the old implementation counted zero of three completed rows.
Default windows now use the database clock, rounded up to a millisecond exclusive bound that the API can replay.
Explicit caller windows retain their half-open semantics. No drift threshold, expected row count or traffic fixture
was relaxed. The complete monitoring file, public MLOps authorization file and drift unit file passed **nine of nine**
after correction; this includes the original 144-row assertion and the new clock/replay/exclusive-bound case.
The original broad failure and controlled failing reproduction are retained.

Consequently **174 unique regression cases have passing evidence**: **163 PostgreSQL integration cases across
27 files plus 11 cross-backend conformance cases**. That is the original 173 cases plus one new clock regression,
not 172+9 independent passes. The corrective nine-case run also repeats four already counted unit cases.
This is selected integration coverage, not the entire historical Public integration/per-backend conformance matrix.

The additional approved provisioned-model CUDA check passed in 134.35 seconds: actual NF4/fp32 training,
approved model/tokenizer provenance, explicit READY registration, independent inference and scratch cleanup.
Training CE was **4.08427683 → 1.72982484**. A read-only PostgreSQL proof confirms `device_actual: cuda`,
`quantization: nf4`, `base_model: provisioned-llama`, `base_model_name: approved-llama`, adapter format,
READY version and successful prediction with output. The final CPU isolated-serving case passed again in
61.81 seconds. All **eight current adapter lifecycle cases** have passing evidence across the focused reports;
filtered rerun cases remain labelled as filtered, not as passed or unavailable hardware.

The CPU quantization-refusal entry point also passed all 16 checks without requesting GPU training. The existing
16 migrations reran idempotently over the populated disposable test database and readiness passed afterward.
No production database, existing evidence database, external service or provisioned customer model was used.
`.tmp/phase6-port/evidence.json` maps the actual reports and separates repeated checks from unique coverage.

Production compilation passes; the full-project compiler still reports exactly the original 22 test-file
diagnostics, with byte-identical output. A new test-only nullability diagnostic was corrected; no unrelated baseline
diagnostics were fixed. The offline package dry-run contains **216 files**, including all seven added ML
runtime/requirements files, both compiled adapter modules and this existing public verification report, with no
tests, caches, model weights, environment files, private checkout paths or private runtime modules. The CI YAML
parses with a new separate CPU `peft` job; hosted CI and Linux execution have not run from this workstation.

### Commands and evidence

Actual local orchestration commands (ignored helpers hold installed interpreter paths, disposable database selection,
stdout/stderr and JSON reports; they are not runtime/package dependencies):

```powershell
rtk proxy node .tmp/phase6-port/setup.mjs
rtk proxy node .tmp/phase6-port/database.mjs
rtk proxy node .tmp/phase6-port/numerical.mjs cpu
rtk proxy node .tmp/phase6-port/numerical.mjs cuda
rtk proxy node .tmp/phase6-port/recovery.mjs
rtk proxy node .tmp/phase6-port/run.mjs peft-integration tests/integration/ml-peft.test.ts
rtk proxy node .tmp/phase6-port/finish-verification.mjs
rtk proxy node .tmp/phase6-port/run.mjs temp-cleanup-red tests/unit/ml-peft-cancellation.test.ts
rtk proxy node .tmp/phase6-port/run.mjs temp-cleanup-green tests/unit/ml-peft-cancellation.test.ts tests/unit/ml-runner-protocol.test.ts tests/unit/ml-runner-environment.test.ts tests/unit/ml-python-runtime.test.ts
rtk proxy node .tmp/phase6-port/finish-verification.mjs --final
rtk proxy node .tmp/phase6-port/run.mjs monitoring-clock-red tests/integration/ml-monitoring.test.ts --testNamePattern="uses database time"
rtk proxy node .tmp/phase6-port/run.mjs monitoring-clock-green tests/integration/ml-monitoring.test.ts tests/integration/ml-mlops-auth.test.ts tests/unit/ml-drift.test.ts
rtk proxy node .tmp/phase6-port/run.mjs peft-provisioned-gpu tests/integration/ml-peft.test.ts --testNamePattern="CUDA NF4: real isolated scratch"
rtk proxy node .tmp/phase6-port/run.mjs peft-final-cpu tests/integration/ml-peft.test.ts --testNamePattern="CPU: real isolated scratch"
rtk proxy node .tmp/phase6-port/finalize.mjs
rtk proxy node .tmp/phase6-port/evidence.mjs
rtk proxy node .tmp/phase6-port/audit.mjs
rtk proxy node .tmp/phase6-port/package.mjs
```

To reproduce without those local helpers, set `DATABASE_URL` and `TEST_DATABASE_URL` to a disposable database,
apply the repository migrations, and set `ML_TEST_PEFT_PYTHON` / `ML_TEST_QLORA_PYTHON` to approved CPU/CUDA
interpreters. Run the tracked entry points directly:

```powershell
node node_modules/vitest/vitest.mjs run tests/unit/ml-peft.test.ts tests/unit/ml-tokens.test.ts tests/unit/ml-peft-cancellation.test.ts tests/unit/ml-peft-numerical.test.ts tests/integration/ml-peft.test.ts
& $env:ML_TEST_QLORA_PYTHON -B tests/python/check_qlora.py --gpu --matrix
& $env:ML_TEST_QLORA_PYTHON -B tests/python/check_qlora_runner.py
node node_modules/typescript/bin/tsc -p tsconfig.build.json
node node_modules/typescript/bin/tsc --noEmit
npm pack --dry-run --ignore-scripts --offline
```

Without the optional CUDA interpreter, GPU integration cases are explicitly skipped. The CPU CI job does not
claim GPU coverage. Full optional unit coverage also requires the existing `ML_TEST_*_PYTHON` variables and matching
`ML_*_JOB=true` CI assertions described in `docs/ml.md`; install nothing automatically. Use sequential test files on
memory-constrained hosts. Remote/multi-node, other operating systems, large pretrained models, production load,
power-loss/PITR durability and new private algorithms are outside this verification. Per-call scratch cleanup covers
normal completion and controlled interpreter termination while its Node owner remains alive. An abrupt host or Node
process death can leave an abandoned temporary directory; operator cleanup/reconciliation is not an automated Public
feature. No unrelated or historical temporary directories were swept.

### Changed files and local commits

The selection changes 48 unique files. No migrations, JavaScript dependencies or lockfile entries were added.
The optional Python additions are PEFT 0.21.2, Accelerate 1.15.0, psutil 7.2.2 and, for CUDA only,
bitsandbytes 0.50.2, reusing the existing pinned Transformers/PyTorch stack. These versions were already installed
for local verification; nothing was downloaded. `package.json` adds the public verification report to its file whitelist.

| Area | Files (relative to repository root) |
| --- | --- |
| Configuration/package/CI | `.env.example`, `package.json`, `.github/workflows/ci.yml` |
| Python runtime (`ml/`) | `lora_backend.py`, `lora_finetune.py`, `lora_provisioning.py`, `lora_quantization.py`, `ml_runtime_common.py`, `requirements-peft.txt`, `requirements-qlora.txt`, `runner_parent_guard.py`, `transformers_manifest.py` |
| Services/contracts (`src/ml/`) | `configuration.ts`, `contract.ts`, `domain.ts`, `metrics.ts`, `model-catalog.ts`, `monitoring.ts`, `neural-backend.ts`, `peft-backend.ts`, `peft-models.ts`, `runner-backend.ts`, `runner-process.ts`, `tasks.ts` |
| Integration tests (`tests/integration/`) | `ml-monitoring.test.ts`, `ml-peft.test.ts` |
| Python checks (`tests/python/`) | `check_lora.py`, `check_lora_audit.py`, `check_lora_backend.py`, `check_metrics.py`, `check_provisioned_lora.py`, `check_qlora.py`, `check_qlora_runner.py`, `lora_provisioning_fixtures.py` |
| Unit tests (`tests/unit/`) | `ml-catalog.test.ts`, `ml-domain.test.ts`, `ml-peft-cancellation.test.ts`, `ml-peft-numerical.test.ts`, `ml-peft.test.ts`, `ml-tasks.test.ts`, `ml-tokens.test.ts` |
| Documentation | `README.md`, `docs/configuration.md`, `docs/limitations.md`, `docs/ml-backend-contract.md`, `docs/ml-port-report.md`, `docs/ml.md`, `docs/roadmap.md` |

| Local commit | Files | Verified workstream |
| --- | ---: | --- |
| `92043ed` | 10 | Token/language-model contracts and backward-compatible defaults |
| `11ef43c` | 8 | Safe local LoRA and optional four-bit primitives |
| `dabe76d` | 14 | Public training/checkpoint/registry/inference adapter integration |
| `dd3cf15` | 1 | Exhaustive metric-reference fixture extended for token losses |
| `0100fe5` | 5 | Reproduced interpreter scratch-cleanup correction |
| `0293339` | 2 | Reproduced monitoring clock/window correction |
| `f7b6651` | 3 | Real adapter lifecycle tests and bounded CPU CI |

The companion documentation/package commit follows these verified workstreams; its hash and final working-tree
status are recorded in the completion response and Git history. Every commit stays below the 20-file limit.
The canonical agent roadmap is unchanged apart from a bounded ML capability note. The source repository remains
unchanged; no push, publication, production deployment or production database mutation occurred.

**Final status: COMPLETE for the selected local Public Core adapter release.** Real CPU/CUDA optimization,
frozen-base/trainable-adapter boundaries, recovery, immutable registration, independent inference, negative validation,
tenant isolation, relevant regressions, compilation and package integrity are verified. Distillation, RL/DPO, Ray,
merged artifacts and managed/enterprise capabilities were intentionally not ported. This is not full-platform,
hosted-CI, cross-platform, large-model or production-readiness certification.
