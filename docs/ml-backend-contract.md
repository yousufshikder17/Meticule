# Meticule ML Backend Contract

This is the internal contract between the Meticule ML platform and the execution backends behind it (scikit-learn, PyTorch
and TensorFlow today). It has two versions: `meticule-ml-backend/v1`, and `meticule-ml-backend/v2`, which is v1 plus the
Phase 4 modality vocabulary (section 19). It is **internal**: there is no public plugin SDK, no dynamic loading of third-party
code, and nothing here is promised to external authors. The contract exists so that adding an internal backend is a narrow,
testable change, and so that v1 and v2 coexist instead of being misread.

Read this with [ml.md](ml.md) (lifecycle, HTTP surface, experiments, the optional frameworks and operation). This
document owns the contract; ml.md links here rather than restating it.

## 1. Ownership

Meticule owns platform semantics. A backend owns only numerical computation behind a boundary it cannot cross.

| Meticule owns | A backend owns |
| --- | --- |
| Job lifecycle, scheduling, leases, retries, cancellation, stale-worker fencing | Tensor operations, autograd, optimizers |
| Checkpoint persistence and resume decisions, early stopping, the epoch loop | Framework-native models and their serialization primitives, behind validated metadata |
| Experiment, dataset, split and pipeline identity; evaluation semantics; metric direction | Device execution details and framework kernels |
| Registry, promotion, inference lifecycle, tenant isolation, provenance requirements | Framework RNG internals |
| Artifact and checkpoint compatibility policy; capability negotiation | |
| Error and refusal semantics; the failure taxonomy; resource policy | |

A backend never persists anything, never claims a job, never schedules work and never chooses a device. It receives a bounded
request, computes, and returns a bounded response. The TypeScript side (`src/ml/`) decides everything else.

## 2. Version

There are two versions, string constants in `src/ml/contract.ts`: `ML_BACKEND_CONTRACT` (`meticule-ml-backend/v1`) and
`ML_BACKEND_CONTRACT_V2` (`meticule-ml-backend/v2`), listed together as `ML_BACKEND_CONTRACTS`. `ml/ml_runtime_common.py`
mirrors them as `ML_BACKEND_CONTRACT`, `ML_BACKEND_CONTRACT_V2` and `SUPPORTED_CONTRACTS`. Each request speaks exactly one
version, chosen per request by `contractFor` (section 19.1); a Phase 1-3 snapshot always speaks v1. The version is checked in
four places:

- every runner request carries `contract`. A runner refuses a version it does not support (`unsupported_contract`), and a
  request its version does not cover (section 19.2), before doing work. A runner served without a task lookup (the frozen
  Phase 1 adapter in the tests) supports v1 only;
- every successful runner response echoes the request's `contract`, and the TypeScript client refuses a response that does
  not carry exactly the version it sent (`runtime_incompatible`);
- `tests/unit/ml-contract.test.ts` reads every `ml/*.py` file and fails if the two sides declare different versions or
  emit different refusal codes;
- the conformance suite sends raw requests: another contract is refused, both current contracts are accepted, and a request
  its contract does not cover is refused (section 11).

A change that alters the meaning of a field, the command set, the failure taxonomy or artifact semantics is a new version. v2
was introduced that way: a second constant, a client that selects one version per request, and runners that accept both.
A v1 request keeps its exact Phase 1-3 meaning.

## 3. Operations

Every backend extends `RunnerBackend` (`src/ml/runner-backend.ts`), which implements the required operations below. Neural
backends extend it again through `NeuralRunnerBackend` and add the iterative pair.

| Operation | Required | Input | Output | Cancellation | Errors |
| --- | --- | --- | --- | --- | --- |
| `validate(snapshot)` | Yes | A parsed training snapshot | nothing, or `ConflictError` | Not applicable (no runner) | `invalid_request` |
| `prepare(snapshot, rows, signal)` | Yes | Snapshot and dataset rows | Disjoint train, validation and test indices; optional CV folds; environment | Aborting kills the runner | Runner categories |
| `train(snapshot, rows, prepared, signal)` | Yes | As above, plus the prepared split | Artifact bytes, format, environment, resolved hyperparameters, measured timings | Aborting kills the runner | `training_failed` and others |
| `evaluate` / `evaluateDetailed` | Yes (`evaluate`); `evaluateDetailed` refines it | Snapshot, rows, prepared split, trained model | Metrics for every partition the snapshot asks for | Aborting kills the runner | `evaluation_failed` |
| `predict(snapshot, artifact, rows, environment, signal)` | Yes | Stored artifact bytes and their recorded environment | One JSON prediction per row, or `horizon` predictions for forecasting, in the task's output shape (section 19.8) | Aborting kills the runner | `prediction_failed` and `incompatible_artifact` |
| `trainChunk(snapshot, rows, prepared, state, epochs, signal)` | Capability-gated: `iterative` | The previous checkpoint, or none, and a bounded number of epochs | A new checkpoint, the epoch summaries it ran, and the total completed | As above | `invalid_checkpoint`, `incompatible_checkpoint` |
| `exportModel(snapshot, checkpoint, restoreBest, signal)` | Capability-gated: `iterative` | A checkpoint | An inference artifact built from the last or best-epoch weights | As above | `invalid_checkpoint` |

Every operation is validated first, against the declared capabilities and the catalog, before a runner starts. An operation the
backend does not declare is never called: the processor checks `capabilities.iterative` and the model's `iterative` flag, and
`exportModel` and `trainChunk` are present exactly when `iterative` is declared. The conformance suite asserts that agreement.

Operations are not renamed for this contract. The names are the ones the platform already used, and the decomposition is
already the cleanest one that works for one-shot, chunked and inference-only backends.

## 4. Semantics

These meanings belong to Meticule. A backend may not reinterpret them.

- **Preparation** fixes the split once per attempt. Partitions are disjoint and cover every row. Cross-validation folds
  partition the training indices only, so validation and test stay untouched. A resumed attempt must see exactly the split its
  checkpoint was trained on, or it fails.
- **Training** fits preprocessing and the model on training rows only. No other partition is used to fit anything.
- **A bounded training chunk** is the unit of durable progress. Meticule asks for `N` epochs at a time (`checkpointEveryEpochs`,
  forced to 1 under early stopping), persists each returned checkpoint under the lease fence, and then decides whether to continue.
- **A checkpoint** is opaque to the platform and versioned by its backend's own format. Meticule stores its format and schema
  alongside it and refuses anything it does not recognise.
- **Resumed training** continues from a checkpoint only when the checkpoint's fingerprint matches the configuration being run.
  The fingerprint covers the algorithm, hyperparameters, seed, precision, feature pipeline, target, and the training and
  validation indices. A mismatch is an explicit `incompatible_checkpoint`, never a silent restart.
- **An artifact** is the deployable product of a completed attempt: format, bytes, and environment. It is stored by content hash
  and its format is recorded before inference.
- **Evaluation** computes the metrics the snapshot asks for, on the partitions it names. The metric registry (tasks,
  direction, whether a metric needs labels, whether it is a diagnostic) is declared in `src/ml/metrics.ts` and mirrored in
  `ml/ml_runtime_common.py`; a unit test compares the two (section 19.7). Ranking reads the direction, never the metric
  name.
- **Prediction** is a pure function of the stored artifact and the submitted rows. It refuses an artifact it cannot read before
  it reads any bytes beyond the recorded environment.
- **Environment and provenance** are recorded for every attempt (section 9).
- **Determinism**: the same seed, split and configuration on the same runtime produce the same predictions. The conformance
  suite asserts this for each backend. Across chunking, a checkpointed run must equal an uninterrupted run.
- **Backend refusal** is an explicit, classified failure. There is no fallback to another device, precision or backend.
- **Incompatibility** is decided by compatibility fields only (section 9), never by provenance.
- **Cancellation** is never completion. A cancelled attempt is recorded as cancelled, and its runner is killed before the call returns.
- **Stale execution**: a worker whose lease expired cannot persist progress. Checkpoint and completion writes are fenced by the
  lease and the attempt; backends have no write access at all.

## 5. Capabilities

A backend declares what its runner can execute. The declaration is `BackendCapabilities` (`src/ml/backend.ts`):

| Field | Meaning |
| --- | --- |
| `artifactFormats` | Exactly one format per backend. Inference refuses any other stored format before reading the bytes. |
| `checkpointFormats` | Empty for one-shot backends; one format for iterative ones. |
| `devices` | Devices the runner executes on. Requested explicitly; none is chosen silently. |
| `precisions` | `fp32`, and `mixed` only where CUDA training is real. |
| `preprocessing` | Operations the runner interprets. Any other step is refused before a runner starts. |
| `iterative` | Whether the backend trains in bounded, checkpointable chunks. |
| `tasks` | Tasks the runner implements, from `TASK_TYPES` (`src/ml/tasks.ts`). |
| `inputs` | Column types the runner reads as features, from `COLUMN_TYPES` (`src/ml/domain.ts`). |

Models carry their own capabilities in the catalog (`ModelCapabilities`): scores, scale sensitivity, seeding, `iterative`,
and optionally a narrower `devices` list and `mixedPrecision`. A model definition also declares `inputs` (the feature column
types it reads and a minimum and maximum feature count; default 1 to 199 number, string or boolean columns) and `learning`
(`supervised`, `unsupervised` or `pretrained`; default `supervised`). The rules, enforced by `RunnerBackend.validate` and the
conformance suite:

- a model without its own device list inherits the backend's;
- a model that narrows the device list must stay inside the backend's;
- a model that is iterative requires an iterative backend;
- a model that supports mixed precision requires the backend to declare `mixed`;
- a training policy (checkpoints, early stopping) is refused on a non-iterative backend;
- early stopping needs a validation partition and an iterative model;
- under v2, the backend must declare the model's task, and every feature column type must be one both the model and the
  backend read (section 19.4);
- the backend's `tasks` and `inputs` are exactly the tasks and input types its registered models use (conformance suite).

The device and precision vocabulary is declared once, in `src/ml/domain.ts` (`DEVICES`, `PRECISIONS`), and the TypeScript types
are derived from it. The task vocabulary is declared once in `src/ml/tasks.ts`, the column types in `src/ml/domain.ts`.

Phase 4 added the vocabulary for text, timestamp, image and audio columns and for the forecasting, anomaly-detection and
embedding tasks (section 19), but **no backend implements them yet**: all three declare classification and regression over
number, string and boolean feature columns. Generative models, fine-tuning, PEFT, distributed or remote training and ONNX-style
inference are **not implemented**. They would be new optional fields on `BackendCapabilities` or on the model definition, and
the suite would learn to check them, except where section 14 names a blocking assumption.

## 6. Failure taxonomy

Backend and runner failures are classified into a bounded set of categories (`ML_FAILURE_CATEGORIES`, `src/ml/contract.ts`):

| Category | Meaning |
| --- | --- |
| `invalid_request` | A snapshot or request was refused by validation (`ConflictError`), or a runner refused the data (`undefined_metric`, `invalid_time_series`, `invalid_media`). |
| `unsupported_capability` | The backend does not declare the device, precision, operation, training mode or task requested. |
| `incompatible_artifact` | An artifact this runtime must not load. |
| `incompatible_checkpoint` | A checkpoint from another configuration, backend, format or runtime. |
| `invalid_checkpoint` | A checkpoint that is unreadable, malformed or corrupt. |
| `backend_unavailable` | The backend is not configured for this process. |
| `device_unavailable` | A requested device is not present at runtime (for example CUDA). |
| `runtime_incompatible` | A runner speaks another contract, or the runtime cannot be used. |
| `training_failed` / `evaluation_failed` / `prediction_failed` | The operation failed for a reason that is not one of the above. |
| `invalid_response` | A runner's output did not match its declared shape. |
| `output_limit_exceeded` | A runner produced more than 64 MiB of output and was killed. |
| `cancelled` | The call was aborted by lease loss or cancellation. |
| `timed_out` | The runner exceeded its time budget and was killed. |

The runner refusal codes are a closed set: `unsupported_contract`, `incompatible_artifact`, `incompatible_checkpoint`,
`invalid_checkpoint`, `cuda_unavailable`, `cuda_unsupported`, `mixed_requires_cuda` and `unsupported_device_or_precision`, plus
the four Phase 4 codes `unsupported_task`, `undefined_metric`, `invalid_time_series` and `invalid_media` (section 19.6). A
runner prints one of them on stderr as its last line, as JSON. The client reads only those codes; any other text is discarded. A runner
exit with no fixed code is classified by the operation that failed.

What is stored: a failed training attempt records `{ code: "ml_training_failed", category, message }` and a failed prediction
records `{ code: "ml_prediction_failed", category, message }`. The `code` is unchanged from earlier phases, so existing
consumers keep working; `category` is the taxonomy. The message is the runner's fixed text or a generic one.

What never crosses the platform boundary: source paths, stack traces, Python exception text, dataset values, labels, request
bodies, environment values and free-form stderr. A zod failure on a runner response becomes `invalid_response` with a fixed
message, because zod issue text can echo the output it rejected.

## 7. Runner protocol

Each runner is a fresh Python process per call. It is started with `shell: false`, reads one JSON object from stdin and writes one
JSON object to stdout.

Request envelope: `{ "command", "contract", "snapshot", "rows", ...command fields }`. A v2 request also carries `task`, the
algorithm's catalog task, which the runner checks against its own (section 19.2). Commands are `prepare`, `train`,
`train_chunk`, `export`, `evaluate` and `predict`. The sklearn runner also answers `catalog`. A backend may add its own request
fields through `RunnerBackend.requestExtras()` (none does yet); the protocol fields always win over them.

Response: the command's JSON with `contract` set to the request's contract. Stdout is capped at 64 MiB; a larger response kills
the runner. Nothing a command prints can corrupt the response: while it runs, Python-level output goes to stderr, and file
descriptor 1 is pointed at stderr for the rest of the process (native libraries write to it directly), while the response is
written once to a duplicate of the original stdout. Buffered Python and C-runtime output is flushed before the final line, so a
refusal stays the last stderr line. A library linked to its own private C runtime cannot be flushed this way; its output
still goes to stderr, but could land after the refusal line.

Timeouts (`ML_TIMEOUT_MS`, default 120 s) and cancellation kill the child. A child never outlives its call.

Environment: runners start from an allowlist (`src/ml/runner-process.ts`). They receive `PATH` and the system variables the
interpreter and native libraries need, plus the thread limits, `PYTHONDONTWRITEBYTECODE`, `PYTHONIOENCODING`, and the Hugging
Face offline settings (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`,
`TOKENIZERS_PARALLELISM=false`). They never receive the database URL, the JWT secret, provider keys, a Hugging Face token or
cache path, `PYTHONPATH` or `PYTHONHOME`. A unit test plants a secret and asserts the child cannot see it; another asserts the
offline settings and the absence of the token and cache variables.

Stdin and stdout are the only channels. A runner never reads a file it was not handed, never opens a socket, and never receives a
database handle.

## 8. Artifacts

An artifact is valid when its format is one the backend declares, its bytes match the content hash recorded when it was written,
and its recorded environment is compatible with the loading runtime. The checks run in this order:

1. The stored format is in `capabilities.artifactFormats`. Otherwise the bytes are never read.
2. The bytes are read from the tenant's rows in `ml_blobs` (PostgreSQL, row-level security), and their SHA-256 and size
   are verified against the stored reference.
3. The recorded environment is checked against the loading runtime **before** any unpickling or tensor loading.
4. The payload is loaded through a safe loader: PyTorch uses `weights_only=True`, TensorFlow reads NumPy archives with
   `allow_pickle=False`. Neural artifacts contain no pickled objects.
5. The payload's own recorded environment is checked again.

Unreadable or malformed bytes are `incompatible_artifact`, never a framework exception.

**Storage bound.** Every blob (dataset, model artifact or training checkpoint) is at most 16 MB in the public edition. A larger
one is refused with `output_limit_exceeded` and a message naming the limit; nothing is truncated and no other storage is used.
A neural checkpoint holds the weights, the optimizer state and the best-epoch weights, so very wide inputs after one-hot
encoding (roughly 3,000 or more columns at the maximum first-layer width) can reach the bound. The training failure then says so.

**Compatibility** is what the artifact is and what it needs to load: `backend`, `artifact_format`, `artifact_schema`, `protocol`,
the Python major.minor version, and the runtime library versions the runner declares (`runtime_keys`). A mismatch in any of
them is refused.

**Provenance** is how and where it was made: `adapter_sha256`, `platform`, the Python patch version, the device used, the CUDA
runtime, the hardware name and the training environment. It is recorded and never compared. A test changes every provenance
field and asserts the artifact still loads.

Schema evolution follows [ml.md](ml.md#artifact-compatibility-and-provenance). Bump `artifact_schema` only when a
previously written artifact can no longer be used. Old artifacts remain loadable through `SUPPORTED_SCHEMAS` and
`SUPPORTED_PROTOCOLS`, which are verified against real Phase 1 and Phase 2 artifacts by existing tests.

Cross-backend consumption is refused twice: the stored format rejects a foreign format before reading bytes, and each runner
rejects a recorded environment whose backend is not its own. The conformance cross-backend tests check both directions for every
pair the environment provides.

**Integrity boundary.** The runner trusts the artifact bytes it is handed, because the store has verified them. A runner does not
verify the hash itself. Tampering with bytes and their recorded hash together is therefore outside what a runner can detect; see
section 12.

## 9. Checkpoints

A checkpoint is not an artifact. It exists to resume training, is never served, and carries everything needed to continue
exactly. Each neural checkpoint records:

- `schema` (1) and `format` (`pytorch-checkpoint-v1` or `tensorflow-checkpoint-v1`), checked before anything is restored;
- `fingerprint`, the identity of the configuration that wrote it, checked against the configuration being resumed;
- `completedEpoch`, `elapsedSeconds`, the last train and validation losses, and the best epoch so far;
- the device and precision it ran with, and the device provenance;
- the model and optimizer state, the shuffling generator's state, the framework RNG state, and the best-epoch weights when early
  stopping keeps them;
- the gradient scaler state under mixed precision;
- the environment, checked with the same compatibility rules as an artifact.

Meticule decides the epoch loop, the checkpoint cadence and early stopping. Early stopping is rebuilt on resume from the persisted
epoch summaries, so a resumed attempt makes the same decision an uninterrupted one would.

Corrupt checkpoints are `invalid_checkpoint`. A checkpoint from another configuration, backend or runtime is
`incompatible_checkpoint`. Both are refused before any training step runs. The processor resumes from the latest persisted
checkpoint for the job. A checkpoint that cannot be read fails the attempt; the processor does not start over from scratch.

## 10. Backend registration

Adding an internal backend is a narrow path:

1. **Runner.** Write a Python script that handles the six commands and calls `ml_runtime_common.serve`. Declare its library
   versions, its contract, its runtime keys and its artifact and checkpoint formats in the same style as `ml/pytorch_runner.py`.
   Keep framework code inside that script. Pin its requirements in `ml/requirements-<name>.txt`.
2. **Models.** Register model definitions with `defineModel` and `registerModels` (`src/ml/model-catalog.ts`). Declare each
   model's devices and mixed-precision support explicitly. Put the backend's device list in one exported constant, as
   `pytorch-models.ts` does, and reuse it in the backend.
3. **Backend class.** Extend `RunnerBackend`, or `NeuralRunnerBackend` for an iterative neural backend. Declare
   `BackendCapabilities` once, as a constant. Pass the runner script path and the timeout. Write no validation: it is inherited.
4. **Configuration.** Register the backend in `mlConfiguration` (`src/ml/configuration.ts`) behind its own enable flag and
   interpreter variable, and add both to `.env.example`. This is the only place backends are constructed.
5. **Conformance.** Add one file under `tests/conformance/` that calls `describeContract` with the backend's interpreter, runner
   script and two quick configurations, and add it to the CI job that provides the interpreter. Add the pairs to
   `cross-backend.test.ts`.

Nothing else changes. Persistence, the processor, the registry, evaluation, routing and the HTTP layer are backend-blind.
Section 16 records the JAX backend, which put this claim to the test.

Not allowed: a public plugin loader, executing third-party code from a configuration value, or a backend that persists its own
state.

## 11. Conformance suite

`tests/conformance/contract-suite.ts` is one suite, written against platform semantics, and each internal backend runs it through
a thin file. It checks, where the backend's declared capabilities say a check applies:

- identity and declared capabilities; truthfulness about iteration and models; `tasks` and `inputs` equal to the tasks and
  input types the backend's registered models use;
- validation refusals before any runner starts: an unknown algorithm, a model from another backend, an uninterpreted operation,
  an undeclared device, an undeclared mixed precision, a policy on a non-iterative backend;
- preparation (disjoint, complete partitions), training, evaluation, prediction, determinism for the same seed and split, and a
  regression lifecycle;
- artifact format and provenance fields; refusal of a foreign runtime, corrupt bytes, and a provenance-only difference that must
  load;
- cancellation before a call starts; a runner killed at its time budget and reported as `timed_out`; a request from another
  contract refused, and the current contract accepted;
- raw envelopes sent straight to the runner: a v1 `prepare` or `predict` carrying Phase 4 vocabulary refused as
  `unsupported_contract`; a v2 request naming another task or an unknown algorithm refused as `unsupported_task`; a matching
  v2 request accepted with v2 echoed;
- for iterative backends: a checkpoint per chunk and resume to the next epochs; chunked training equal to uninterrupted training;
  corrupt and foreign checkpoints refused with the right category;
- for backends that declare CUDA: a real CUDA run with CUDA recorded as the actual device, or an explicit `device_unavailable`
  refusal. The check accepts exactly those two outcomes.

Every check is asserted. A check is not skipped by a flag. The only gate is whether the interpreter is configured; in CI, the
suite asserts that the interpreter is present, so a missing runtime fails the job rather than passing.

`tests/conformance/cross-backend.test.ts` checks that each framework refuses the other frameworks' artifacts and checkpoints. Each
pair runs when both interpreters are configured. The CI test job provides scikit-learn and PyTorch, so the sklearn and PyTorch pair
runs there. The TensorFlow job provides only TensorFlow, so the pairs that include TensorFlow run locally when all three are
configured. This is a real limit of the current CI layout and is recorded in section 13.

The suite needs no database. It exercises the backend contract directly. The lifecycle through the job processor, tenant isolation
and the endpoints are covered by the integration suites.

Evidence that the suite can fail: with TensorFlow falsely declaring CUDA, the CUDA check fails and nothing else changes.

## 12. Security boundaries

| Concern | Boundary | Status |
| --- | --- | --- |
| Arbitrary deserialization | PyTorch and TensorFlow artifacts and checkpoints load without pickle. Only scikit-learn artifacts are pickles (below). | Verified by tests |
| Unsafe pickle | sklearn artifacts are the only pickles. They are loaded only after their recorded environment is accepted, and only bytes whose hash was verified by the store reach the runner. | Accepted residual risk, see below |
| Malicious metadata | Metadata is parsed, checked against the contract, then used. Architecture limits are enforced again inside the runner. | Verified by tests |
| Malformed runner output | Every response is parsed at one boundary and becomes `invalid_response` on mismatch. | Verified by the conformance suite |
| Oversized responses | A response over 64 MiB kills the runner. Rows are capped at 50,000 by the domain schema. | Verified by code review |
| Stderr leakage | Only fixed refusal codes are read. Other stderr is discarded. | Verified by the conformance suite |
| Library output on the protocol channel | While a command runs, stdout is redirected to stderr at the Python and file-descriptor levels; the response goes once to a duplicate of the original stdout (section 7). | Verified by `tests/python/check_contract.py` |
| Malicious media cells | Image and audio cells pass strict base64 and bounded header walks before anything decodes them, and each request's total pixels and audio seconds are capped (section 19.5). The validators never decode pixels. | Verified by the shared vector file in both languages |
| Source-path and stack-trace leakage | Runner exceptions are reduced to a type name or a fixed code. Refusal messages are checked for paths and traces. | Verified by the conformance suite |
| Secret and environment leakage | Runners start from an allowlist. | Verified by a planted-secret test that fails on the previous behaviour |
| Artifact tampering | The store verifies size and SHA-256 before the runner sees bytes. | Store-level tests; not repeated at the runner |
| Checkpoint tampering | Checkpoints are content-addressed like artifacts, and the loader refuses malformed or corrupt payloads. | Verified by the conformance suite for corrupt input |
| Cross-tenant artifact access | Blobs are keyed by tenant in `ml_blobs` under row-level security, and every read names the tenant. | Integration tests (database required) |
| Cross-backend confusion | Stored format, recorded backend and runner environment checks. | Verified by the cross-backend tests |
| Downgrade and fallback | No fallback between devices, precisions, backends or contract versions. Each refusal is explicit. | Verified by the conformance suite |
| Unsafe device fallback | CUDA is never silently replaced by CPU. The TensorFlow runner refuses CUDA. The PyTorch runner refuses an unavailable CUDA device. | Verified by the conformance suite |
| Unbounded capability declarations | Capabilities are fixed constants reviewed in code; the suite checks models against them. | Verified by the conformance suite |

**Accepted residual risk: scikit-learn pickles.** Loading a pickle runs code. The protection is that the bytes were written by a
worker, their SHA-256 is recorded with the attempt, and the store verifies it before the runner sees them. Blobs are append-only by
trigger. Anyone with write access to the database as its owner (able to rewrite both a blob and the row that references it) can
run code in the runner. That risk predates this phase and is unchanged. The
environment allowlist limits what such code could read: it cannot read the database URL or the JWT secret. Removing the risk
entirely requires moving sklearn to a non-pickle format or signing artifacts. Both are recorded as future work; neither is a v1
change.

## 13. Current implementations

| | scikit-learn | PyTorch | TensorFlow | JAX | XGBoost | LightGBM | Transformers | Audio | Forecasting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Class | `SklearnBackend` | `PytorchBackend` | `TensorflowBackend` | `JaxBackend` | `XgboostBackend` | `LightgbmBackend` | `TransformersBackend` | `AudioBackend` | `ForecastingBackend` |
| Artifact format | `sklearn-pickle-v1` | `pytorch-state-dict-v1` | `tensorflow-variables-v1` | `jax-params-v1` | `xgboost-json-v1` | `lightgbm-text-v1` | `transformers-reference-v1` | `audio-npz-v1` | `forecast-npz-v1` |
| Checkpoint format | none (one-shot) | `pytorch-checkpoint-v1` | `tensorflow-checkpoint-v1` | `jax-checkpoint-v1` | none (one-shot) | none (one-shot) | none (one-shot) | none (one-shot) | none (one-shot) |
| Devices | `cpu` | `cpu`, `cuda` | `cpu` | `cpu` | `cpu` | `cpu` | `cpu` | `cpu` | `cpu` |
| Precisions | `fp32` | `fp32`, `mixed` (CUDA only) | `fp32` | `fp32` | `fp32` | `fp32` | `fp32` | `fp32` | `fp32` |
| Iterative | no | yes | yes | yes | no | no | no | no | no |
| Tasks | `classification`, `regression`, `anomaly_detection` | `classification`, `regression` | `classification`, `regression` | `classification`, `regression` | `classification`, `regression` | `classification`, `regression`, `forecasting` | `classification`, `embedding` | `embedding`, `classification` | `forecasting` |
| Feature column types | `number`, `string`, `boolean` | `number`, `string`, `boolean` | `number`, `string`, `boolean` | `number`, `string`, `boolean` | `number`, `string`, `boolean` | `number`, `string`, `boolean`; `timestamp` for `lightgbm_forecaster` | `text`, `image` (exactly one) | `audio` (exactly one) | `timestamp` (the split's time column) |
| Contracts accepted | v1, v2 | v1, v2 | v1, v2 | v1, v2 | v1, v2 | v1, v2 | v2 (its snapshots always use non-tabular columns) | v2 | v2 |
| Preprocessing | eight operations, including `variance_threshold` | seven numeric and categorical operations | seven numeric and categorical operations | seven numeric and categorical operations | the same seven; missing numbers pass through as NaN | the same seven; missing numbers pass through as NaN | none | none | none |
| Interpreter variable | `ML_PYTHON` | `ML_PYTORCH_PYTHON` | `ML_TENSORFLOW_PYTHON` | `ML_JAX_PYTHON` | `ML_XGBOOST_PYTHON` | `ML_LIGHTGBM_PYTHON` | `ML_TRANSFORMERS_PYTHON` (+ `ML_TRANSFORMERS_MODEL_ROOT`) | `ML_AUDIO_PYTHON` (the scikit-learn venv) | `ML_FORECASTING_PYTHON` (the scikit-learn venv) |
| Conformance file | `tests/conformance/sklearn.test.ts` | `tests/conformance/pytorch.test.ts` | `tests/conformance/tensorflow.test.ts` | `tests/conformance/jax.test.ts` | `tests/conformance/xgboost.test.ts` | `tests/conformance/lightgbm.test.ts` | `tests/conformance/transformers-*.test.ts` (task suite) | `tests/conformance/audio-*.test.ts` (task suite) | `tests/conformance/forecasting-*.test.ts` (task suite) |

Known CI limit. The test job provides scikit-learn and PyTorch. The TensorFlow job provides only TensorFlow, and it runs the
TensorFlow conformance file. The pairs involving TensorFlow in `cross-backend.test.ts` therefore run locally when all three
interpreters are configured, not in hosted CI. Similarly, `tests/integration/ml-matrix.test.ts` (the Phase 3E matrix) needs all three
interpreters and is not run by either CI job. Closing this needs one venv that hosts all three frameworks, which is a CI design
change and is not made in this phase.

### 13.1 Gradient boosting (Phase 4 M3)

XGBoost and LightGBM run through their native APIs (`xgb.train` on a `DMatrix`, `lgb.train` on a `Dataset`), never through
scikit-learn wrappers. `ml/gbdt_common.py` holds what the two runners share; each runner supplies only its framework's
parameter mapping, fit, raw prediction and model text. Preprocessing is the neural runners' JSON-state NumPy implementation,
fitted on training rows only. Missing numbers with no imputer pass through as NaN (`transform(..., missing="nan")`; the neural
default still refuses them), and both frameworks route NaN natively.

- **Artifacts.** `xgboost-json-v1` is the booster's own JSON (`save_raw("json")`). `lightgbm-text-v1` is its text model
  (`model_to_string`). Each sits in a JSON envelope `{metadata, model}`; the envelope keys, environment and metadata are checked
  before the framework parses any model text. No pickle. Runtime keys: the framework version plus the numeric stack; LightGBM
  also records `pandas`, which its forecasting model uses.
- **Determinism.** Fixed seeds, one thread, histogram trees (XGBoost); `deterministic`, `force_row_wise` (LightGBM). The same
  seed and split give a byte-identical model.
- **Framework-native early stopping is not Meticule's training policy.** `early_stopping_rounds` (0 = off, at most 200) monitors
  the validation partition. That partition only picks the stopping point; it never fits preprocessing or trees. The number of
  trees used is recorded as `best_iteration` (resolved hyperparameters) and `bestIteration` (performance), and every later
  prediction uses exactly that many: XGBoost cuts prediction at the count, and LightGBM saves only those trees. Both backends are
  one-shot for Meticule (`iterative: false`, no checkpoints): a Meticule training policy is refused, and recovery after a crash
  is Meticule retrying the attempt from scratch. Early stopping inside cross-validation is refused, not silently disabled,
  because a fold has no separate monitoring partition.
- **Bounds.** snake_case hyperparameters, bounded in the catalog and enforced again by the runner. Unknown names are refused,
  never forwarded to the framework.
- **Devices.** CPU only. CUDA is refused as `cuda_unsupported`.

### 13.2 Hugging Face Transformers (Phase 4 M4)

Backend `transformers` (CPU, fp32, one-shot): pretrained BERT/DistilBERT text models and ViT image models that an operator has
provisioned under `ML_TRANSFORMERS_MODEL_ROOT`, an absolute existing directory. The root is platform-wide: every tenant may use
every provisioned model. Each model is one directory holding the Hugging Face files and a `meticule-model.json` manifest
(`{schema: 1, name, kind, family, source: {repository, revision}, license?, files: {relpath: sha256}}`), written by
`ml/transformers_manifest.py <model_dir> <kind> <family> <repository> <revision> [license]`.

| Model | Task | Learning | Input | Kind / families |
| --- | --- | --- | --- | --- |
| `text_classifier` | classification | pretrained (binds its labels) | one `text` | text-classifier / bert, distilbert |
| `text_embedder` | embedding | pretrained | one `text` | text-encoder / bert, distilbert |
| `text_embedding_classifier` | classification | supervised: softmax head over the frozen encoder | one `text` | text-encoder / bert, distilbert |
| `image_classifier` | classification | pretrained (binds its labels) | one `image` | image-classifier / vit |
| `image_embedder` | embedding | pretrained | one `image` | image-encoder / vit |
| `image_embedding_classifier` | classification | supervised: softmax head over the frozen encoder | one `image` | image-encoder / vit |

- **Provisioning check (every command, before any framework code touches the files).**
  - The name matches `^[a-z0-9][a-z0-9._-]{0,63}$` and resolves inside the root.
  - Every entry is a regular file or directory: no symlink, junction or other reparse point.
  - The casefolded listing equals the manifest plus `meticule-model.json`, with no case-only duplicates and at most 64 files.
  - Manifest keys have no `.`/`..` segments, and only allowlisted filenames are accepted: safetensors weights, configs,
    tokenizer and processor files. No `.py`, `.bin`, `.pt`, `.pkl`, `.h5` or shard index.
  - `config.json` has no `auto_map` or `transformers_weights`, and its `model_type` equals the manifest family.
  - The total size is at most 2 GiB.
  - Every file is copied into a fresh private directory while it is hashed, and only that copy is loaded.
- **Loading.**
  - Concrete classes, `local_files_only`, `use_safetensors=True`, `trust_remote_code=False`, `dtype=float32`.
  - Encoders load without a pooling layer.
  - Any missing, mismatched or unexpected weight (beyond an unused pooler) is refused, never randomly initialised.
  - At most 400M parameters. The image processor is the PIL one. `HF_HOME`/`HF_HUB_CACHE` point at a per-call empty directory.
- **Inputs.** Tokenisation always truncates at min(tokenizer limit, position embeddings, `maxLength`). Images pass
  `media_common` first, then Pillow (PNG/JPEG only, a pixel cap, decompression-bomb warnings as errors), one at a time.
- **Embeddings** are mean-pooled over real tokens (or the first token), optionally L2-normalised, and their dimension is recorded
  as `embedding_dimension`. Pretrained models refuse cross-validation, because nothing is fitted.
- **Artifact `transformers-reference-v1`** is JSON. It holds the model reference (name, kind, family, repository, revision and the
  verified digest), labels, settings and any trained head (`ml/linear_heads.py`), never base weights. Every load re-verifies
  the provisioned model and refuses a changed digest. Environments record the model name, family and digest, never a path.
- **Tests** use tiny random models generated offline (`tests/python/hf_fixtures.py`), including a malicious variant for each
  provisioning rule. They prove the execution path and the safety rules, not semantic quality. Fine-tuning the encoder itself
  (and PEFT/LoRA) is not part of Phase 4.

### 13.3 Audio (Phase 4 M5)

Backend `audio` (NumPy and SciPy only, on the scikit-learn interpreter; CPU, one-shot). One `audio` column of WAV clips that
passed the media rules of section 19.5.

- **Front end** (`ml/audio_common.py`).
  - Decoding uses only `media_common`'s chunk walker (never `scipy.io.wavfile`).
  - Mono by the arithmetic mean of the channels.
  - Resampled to `sampleRate` with `scipy.signal.resample_poly` over the reduced ratio.
  - Frames of `frameMs` with hop `hopMs`, a periodic Hann window and `rfft` power.
  - An HTK mel filterbank of `nMels` bands from 0 Hz to Nyquist, then log(max(energy, 1e-10)).
  - MFCC via an orthonormal DCT-II keeping `nMfcc` coefficients.
  - The clip's vector is each coefficient's mean, then each coefficient's standard deviation (length 2 * nMfcc).
  - One clip at a time.
- **Models.** `audio_features` (embedding; a fixed transform, nothing is fitted) returns that vector and records
  `embedding_dimension`. `audio_mfcc_classifier` (classification) standardises the vectors and fits the shared softmax head
  (`ml/linear_heads.py`) on training rows only. Cross-validation fits a fresh head per fold.
- **Artifact `audio-npz-v1`.** A NumPy archive read with `allow_pickle=False`: settings, classes and environment as metadata,
  and the head's arrays with shapes checked against the metadata.
- **What the checks prove.** Decoder known answers for every PCM width and float32, stereo downmix, and resampling. The shared
  vector file gives identical verdicts in both languages. MFCC determinism, and the exact amplitude invariance: scaling a
  broadband clip by a moves only the first coefficient's mean, by 2 ln a * sqrt(nMels). The classifier separates
  low from high tones and noise from tone. These are signal-processing features, not a learned speech or audio-event model.

### 13.4 Retrieval and similarity are not backend operations (Phase 4 M6)

Backends only generate embeddings (`predict` on an embedding model). Storing vectors, building indexes, exact search and
pairwise similarity live in Meticule (`src/ml/embedding-service.ts`, pgvector in PostgreSQL), behind the tenant, role and
lifecycle rules of the platform. The backend contract therefore gains no retrieval command, and every embedding backend is
indexable as soon as it records `embedding_dimension`. See [ml-platform.md](ml.md#embedding-indexes-and-similarity-phase-4).

### 13.5 Forecasting and anomaly detection (Phase 4 M7)

**Forecasting** (task `forecasting`): backend `forecasting` (`seasonal_naive_forecaster`, `ridge_forecaster`) and LightGBM's
`lightgbm_forecaster`, all over `ml/forecasting_common.py`. The only feature is the chronological split's time column, and
the target is the numeric series (non-nullable). Random splits and cross-validation are refused before any runner starts.

- **Series.** Rows are ordered by their exact UTC instant (the timestamp rules of section 19.5). Duplicate instants, irregular
  spacing (every step must have the same length, so calendar months are refused), missing or non-finite targets, and a
  history shorter than the model needs are refused as `invalid_time_series`.
- **No leakage.** Features are target lags 1..`lags` and rolling means over `windows`, built with pandas `shift`/`rolling` so a
  row's features use strictly earlier targets only. Models are fitted on the training block alone. Forecasts are recursive:
  each step's prediction, never a future actual, feeds the next. The checks prove that changing test-block values changes
  neither the fitted model nor the validation metrics, and that shuffled input rows give identical results.
- **Evaluation.**
  - The training block scores the one-step in-sample fit.
  - Each later block uses up to 200 evenly spaced rolling origins. Each origin sees only actuals before it, forecasts
    `horizon` steps, and is scored only on targets inside the block.
  - Metrics are mae, rmse, smape (0/0 terms count 0) and mase. MASE is scaled by the in-sample seasonal-naive MAE of the
    training block with `seasonal_period`, and refused when that scale is zero.
- **Prediction.** Rows are the most recent history (timestamp and target), at least the model's context, at the training
  spacing. The output is exactly `horizon` items `{step, timestamp, value}`, with UTC ISO timestamps continuing the series.
- **Artifacts.** `forecast-npz-v1` NumPy archives (allow_pickle=False). The ridge model is stored as coefficients, never
  pickled. The LightGBM forecaster keeps `lightgbm-text-v1` with forecasting metadata. LightGBM's native early stopping is not
  offered for the forecaster, because the validation block belongs to evaluation.

**Anomaly detection** (task `anomaly_detection`) runs inside the scikit-learn runner: `isolation_forest`,
`local_outlier_factor` (novelty mode) and `robust_covariance` (elliptic envelope), over the existing tabular preprocessing.

- **Unsupervised.** A boolean target, when present, is never used for fitting: the checks fit with and without labels, and
  with flipped labels, and get identical scores.
- **Scores and threshold.** The score is `-score_samples` (higher means more anomalous). Local outlier factor's training scores
  come from `-negative_outlier_factor_`, since its `score_samples` must not be used on its own training rows. Every model stores
  its training scores. The threshold is the training-score quantile at 1 - `contamination` (numeric, default 0.1, in (0, 0.5]).
  A row is an anomaly when its score is above the threshold. The output is `{score, anomaly}` per row.
- **Evaluation.** Labelled data gets roc_auc, average_precision and anomaly precision, recall and F1. A partition without an
  anomaly is refused as `undefined_metric` (stratify the split). Unlabelled data gets only `threshold_calibration_error`
  (|flagged fraction - contamination|), a diagnostic of how the threshold transfers, not of detection quality.
- **Artifacts.** `sklearn-pickle-v1` with `artifact_schema` 2 (the payload adds the threshold and training scores). The adapter
  supports schemas {1, 2}, classification and regression stay schema 1, and the runtime keys are unchanged.

## 14. Framework-neutrality audit

Findings from a search of the ML subsystem for framework assumptions. Each is classified: **A** belongs in the generic contract,
**B** is correctly backend-specific, **C** was accidental coupling, **D** is a future concern to document.

| Finding | Class | Resolution |
| --- | --- | --- |
| `MlBackend` interface, training policy, artifact store, metric directions, failure taxonomy | A | Kept; the taxonomy and capabilities were added in this phase |
| Device and precision vocabulary, declared in domain zod enums and in backend types | A, duplicated | Fixed: one declaration in `domain.ts`, types derived from it |
| `validate` duplicated between the sklearn and neural backends, with a sklearn-only training-policy special case | C | Fixed: `RunnerBackend.validate` reads capabilities |
| `call`, runner-response parsing and the `prepare`/`train`/`evaluate`/`predict` methods duplicated | C | Fixed: one base class |
| `artifactFormats` declared as a separate field next to the capability model | C | Fixed: `capabilities.artifactFormats` |
| Each runner duplicated its `__main__` error handling and its `RunnerError` class | C | Fixed: `ml_runtime_common.serve` and `RunnerError` |
| Three copies of `requested_metrics` in the Python runners | C | Fixed: one helper in `ml_runtime_common` |
| Preprocessing refusal message named "sklearn", in code the generic pipeline validator owns | C | Fixed: generic message |
| Sklearn accepted any preprocessing operation and failed later inside Python | C | Fixed: declared `preprocessing` capability |
| Preprocessing operation registry in `preprocessing.ts` is shaped around tabular scopes (numeric, categorical, global) | D | Documented. Phase 4 keeps it tabular: only string and boolean features count as categorical (text does not), and under v2 preprocessing steps are refused for a model whose inputs are not all tabular. A text, image or time-series pipeline would need its own registry. |
| Phase 4 rules declared in both languages: contract selection, the metric registry, media and timestamp rules, partition sizes | A, mirrored | Each pair is compared by a cross-language test (section 19.10) |
| Iterative models locate the epoch count by the hyperparameter name `epochs` | D | Documented convention. An iterative model that counts differently would need the catalog to declare its epoch parameter. |
| Neural model definitions and the Python runners mirror each other's limits (`MLP_LIMITS`, `LIMITS`) | C, remaining | Kept in sync by a comment in both files. A cross-language check like the one for the contract version would close it. Not done here. |
| The two neural runners duplicate their evaluation and cross-validation loops (about 975 differing lines by `diff`, most of it framework code) | B for the framework code, D for the loops | Left alone. These loops compute with framework-specific predictions. Extract them when a third neural framework needs the same loops. |
| AutoML and experiments default their backend to `"sklearn"` | B, deliberate | Kept. The default is documented, and an explicit backend is always accepted. |
| Backend registration in `mlConfiguration` | B, the single registration point | Kept |
| Sklearn model definitions register under the `sklearn` backend | B | Kept |

## 15. Future phases

What the v1 contract already accommodates, what a new optional capability would add, and what would block:

| Category | Status under v1 |
| --- | --- |
| Embeddings, NLP text classification | Vocabulary only. v2 defines `text` columns, the `embedding` task, its output and its metrics (section 19), but no backend declares them yet. Preprocessing steps stay tabular (section 14). A classifier head over embeddings works as a neural backend. |
| Vision, audio, time series, anomaly detection | Vocabulary only. v2 carries images and audio as validated base64 cells and timestamps as ISO-8601 cells in the same JSON rows, plus the forecasting and anomaly-detection tasks, outputs and metrics (section 19). No backend implements them yet; only the chronological split already runs, for classification and regression. |
| Hugging Face-style models, XGBoost, LightGBM | A one-shot backend is supported as sklearn is. An iterative one is supported as the neural backends are. Their weights need the artifact format to be opaque bytes, which it already is. |
| Richer MLOps, deployment targets, ONNX | Inference artifacts are opaque bytes behind a declared format, so an ONNX format is an additional format and runner. The inference lifecycle is outside the backend contract. |
| Lineage, monitoring, promotion and rollback, advanced HPO | Outside the backend contract by design. The registry and experiments already own them. |
| LoRA and PEFT, foundation-model fine-tuning | Needs a checkpoint that stores adapter weights and a base model reference. The checkpoint is opaque and the fingerprint covers the configuration, so this is an optional capability plus a new checkpoint format. No blocking assumption found. |
| Distributed and remote training | Blocking for the runner boundary. A runner is one local process with no shared state; a remote or distributed backend needs an execution transport and a lease model for its workers. This is a v2 concern. |
| Model routing, distillation, reinforcement learning | Outside the single-attempt model. Routing belongs to the inference layer; distillation and RL need multi-attempt state the platform does not yet model. |

**What required contract v2:** non-tabular inputs, which v2 carries as typed cells in the same JSON rows (section 19). **What would
still need a new version:** a remote or distributed execution transport; a training loop that is not expressed as epochs over rows.

## 16. JAX backend (Phase 4 M2)

The thought experiment that stood here is now an implementation, and it held: adding JAX touched no persistence, processor,
registry, evaluation or HTTP code. It needed a runner (`ml/jax_runner.py`), a requirements file
(`ml/requirements-jax.txt`), model definitions (`src/ml/jax-models.ts`, the shared `mlpDefinitions` under `jax`), a backend
class, a configuration entry (`ML_JAX_ENABLED`, `ML_JAX_PYTHON`) and tests.

- **Real JAX.** The network, `jax.value_and_grad` and a `jax.jit` training step are JAX. sgd, adam and adamw (decoupled weight
  decay 0.01; PyTorch's constants) are hand-written pure functions over parameter and optimizer-state pytrees. A check compares
  the gradients with central finite differences of an independent float64 NumPy implementation.
- **Explicit randomness.** `key(seed)` splits into init, shuffle and dropout keys. Epoch `e` shuffles with
  `fold_in(shuffle, e)`, and batch `b` of epoch `e` drops out with `fold_in(fold_in(dropout, e), b)`. No random state is carried
  between chunks, so chunked training equals an uninterrupted run bit for bit. The checkpoint records the base key data, and a
  restore refuses key data that does not derive from the snapshot seed.
- **Formats.** `jax-params-v1` and `jax-checkpoint-v1` are NumPy archives read with `allow_pickle=False`. The names and shapes of
  every array follow from the recorded architecture (which is itself re-checked against the allowlist). Each entry's size is
  checked before it is decompressed. Non-finite parameters, extra arrays and object arrays are refused. Checkpoints carry the
  parameters, adam moments and step count, and the best-epoch parameters.
- **Devices.** CPU and fp32 only. `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=0` and single-threaded XLA flags are assigned before
  JAX is imported. There is no CUDA jaxlib for native Windows; a CUDA request is refused as `cuda_unsupported` and any other
  device or precision as `unsupported_device_or_precision` (the TensorFlow runner's codes). The environment records
  `jax_platform` and `jax_devices` as provenance. No GPU capability is claimed.

## 17. Deferred

- A public plugin SDK, a plugin loader, marketplace distribution and third-party code execution. None of these is promised.
- Signed artifacts and a non-pickle sklearn format (section 12).
- A single CI environment that hosts all three frameworks (section 13).
- A cross-language check for the neural limits (section 14).
- The Phase 3E cross-framework matrix in hosted CI (section 13).
- Any change to the database schema. This phase needed none: every new field is carried in existing JSON columns and existing
  failure details, and the stored `code` is unchanged.

## 18. Readiness

The contract is ready for internal use: every operation is classified, every capability is checked against the same declarations the
platform uses, and the three backends pass one suite. It is not ready to be promised as an external SDK. The taxonomy and the
checkpoint format have been exercised by three backends, not by a stranger's backend; the runner protocol has no compatibility
policy beyond "each request names one contract the runner supports, and the response echoes it"; and the sklearn pickle boundary is a residual risk that an external author should
not inherit without a decision about signing.

## 19. Phase 4 modality extensions (contract v2)

This section records what Phase 4's first milestone (M1) built: contract v2, its vocabulary, its validation and the safety
rails later milestones rely on. M1 adds **no new backend and no new task implementation**. scikit-learn, PyTorch and
TensorFlow still declare classification and regression over number, string and boolean feature columns. Through them a
snapshot needs v2 only when it uses the chronological split, which every runner's `prepare` implements (`split_partitions`),
or when its dataset carries a column of a new type that is not a feature. Every Phase 1-3 snapshot is v1 and behaves as in
Phase 3. The other tasks and the new feature types are enforced and tested, but no backend executes them yet. Later
milestones extend this section as backends implement them.

### 19.1 Versions and selection

`contractFor(snapshot, task)` (`src/ml/contract.ts`) picks the version of every request; `task` is the algorithm's catalog
task. A request is **v1** only when all of these hold, and **v2** otherwise:

- the task is `classification` or `regression`;
- the split is `random`;
- the dataset declares a target;
- every column is `number`, `string` or `boolean`.

`uses_extensions(spec, task)` in `ml/ml_runtime_common.py` is the same rule (true exactly when v2 is needed). Every Phase 1-3
snapshot is v1, and a v1 snapshot keeps the exact Phase 3 rules: a stored model validates, serves and resumes as before,
including combinations v2 refuses, such as a ridge regression trained on a boolean target.

### 19.2 Where it is enforced

- **TypeScript, before a runner starts.** `RunnerBackend.validate` applies the v2 rules of section 19.4 only when
  `contractFor` returns v2. `call()` sends the selected contract and, on v2, `task`.
- **Python, before any work.** `serve(execute, task_of=...)` checks every request that carries a snapshot, except `catalog`,
  with `require_contract(request, task_of(algorithm))`:
  - a v1 request whose snapshot needs v2 is refused as `unsupported_contract` ("Contract v1 does not cover this request").
    An algorithm the runner does not know counts as needing v2, so a v1 request for one is refused the same way;
  - a v2 request whose `task` is not the runner's task for the algorithm, or whose algorithm the runner does not know, is
    refused as `unsupported_task`.

  The three runners pass `task_of` from their `MODELS` tables. `serve` without `task_of` accepts v1 only.
- **The response.** The runner echoes the request's contract, and `callRunner` refuses any other value
  (`runtime_incompatible`).
- **The training processor.** Before it reads the dataset, it applies the task's cross-validation rule whichever backend
  runs the attempt (`invalid_request`, "Cross-validation is not supported for this task and split"). After `prepare`, on a
  chronological split, it checks that every training instant is at or before every validation instant, and every
  validation instant at or before every test instant (`invalid_response`, "Backend returned a non-chronological split").
  After training, it fails an embedding attempt whose artifact does not record a valid `embedding_dimension` (section 19.8).

### 19.3 Vocabulary

**Tasks** (`TASK_TYPES`, `src/ml/tasks.ts`): `classification`, `regression`, `forecasting`, `anomaly_detection` and
`embedding`. `transcription` is not defined yet. The task registry `TASKS` is the single source of what a task means:

| Task | Target | Target types | Splits | Cross-validation | Prediction rows carry the target | Output | Rows per prediction request |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `classification` | required | number, string, boolean | random, chronological | random split only | no | one label per row | 1,000 |
| `regression` | required | number | random, chronological | random split only | no | one finite number per row | 1,000 |
| `forecasting` | required | number | chronological | no | yes (history rows) | `horizon` items `{ step, timestamp, value }` | 5,000 history rows |
| `anomaly_detection` | optional | boolean | random, chronological | no | no | `{ score, anomaly }` per row | 1,000 |
| `embedding` | optional | number, string, boolean | random, chronological | no | no | a vector of 1 to 4,096 finite numbers per row | 256 |

`maxPredictionRows(task, types)` lowers the cap when the prediction rows carry media: 32 with an image column, 16 with an
audio column, and 64 for an embedding model with a text column.

**Column types** (`COLUMN_TYPES`, `src/ml/domain.ts`): the tabular `number`, `string` and `boolean`
(`TABULAR_COLUMN_TYPES`), plus `text`, `timestamp`, `image` and `audio`, whose cells are JSON strings. `string` and `text`
cells hold at most 10,000 characters; timestamps, images and audio follow section 19.5.

**Datasets.** The target is optional. When present it must be a number, string, boolean or text column, and the dataset needs
another column; without a target one column is enough. No M1 task accepts a text target. A dataset version records
`datasetValidator(schema)`: `meticule-tabular-v1` when every column is tabular, as before, and `meticule-dataset-v2` once any
other column type appears.

**Splits.** `SplitDefinition` is `random` (unchanged; `stratify` defaults to false) or `chronological` (`timeColumn` and the
three fractions; no `stratify`, and an unknown key is refused). The fractions sum to one. `partitionSizes(split, n)` computes
the sizes the runners produce: random follows scikit-learn's two `train_test_split` calls (`floor(n * train)`, then
`floor(rest * validation / (validation + test))`); chronological takes `floor(n * train)` and `floor(n * validation)` and
leaves the rest to test.

**Catalog.** `ModelDefinition.inputs` and `learning` (section 5); `describeModel` returns both. The parameter kind `name` is a
local model reference: a bounded identifier (`^[a-z0-9][a-z0-9._-]{0,63}$`), never a path, nullable with default null.
Search refuses `name` and `intList` parameters explicitly, and AutoML plans only tabular feature columns.

**Preprocessing.** Only string and boolean features count as categorical; text does not.

### 19.4 Validation scope

Under v2, `RunnerBackend.validate` refuses, before any runner starts:

1. a task the backend does not declare;
2. a feature count outside the model's `inputs` bounds, or a feature column type the model or the backend does not read;
3. a missing target where the task requires one, or a target type the task does not accept;
4. a split strategy the task does not allow; cross-validation with a chronological split or for a task without it;
   stratification without a target;
5. for a chronological split: a time column that is not a non-nullable `timestamp`, or a partition that `partitionSizes`
   leaves empty for the dataset's row count. Forecasting also needs the time column as its only feature and a non-nullable
   target;
6. an embedding partition over 5,000 rows;
7. preprocessing steps for a model whose inputs are not all tabular;
8. a forecasting model without an integer `horizon` from 1 to 100 (hyperparameter or model default).

The rules both versions share follow (pipeline, preprocessing operations, policy, device, precision, metrics). Metrics are
checked with `labelled` set to whether the dataset has a target.

**Rows** (`validateRows`, at ingest and at prediction): exactly the schema's columns; prediction rows leave out the target
unless the task's prediction rows carry it (forecasting history). Each cell is checked against its column type, timestamps
and media by section 19.5, and media is charged to the request's budget. A prediction request's row cap (section 19.3) is
checked on the raw array before any cell is decoded. The `Rows` schema admits strings of up to 2,796,204 characters (a 2 MiB
base64 cell), but only image and audio cells may exceed 10,000; any other long string fails with the same validation error
at `[row, column]` as before.

**Runner side** (`split_partitions`, every runner's `prepare`): the random path is the Phase 3 code unchanged; an unlabelled
dataset splits as if every target were zero and cannot stratify. The chronological path sorts rows stably by the exact UTC
microsecond of the time column (ties keep row order), takes the same floor sizes, and refuses an empty partition as
`invalid_time_series`.

### 19.5 Media and timestamp rules

`src/ml/media.ts` (TypeScript, no dependencies) and `ml/media_common.py` (standard library and NumPy) apply the same rules;
Python's timestamp rules are `epoch_micros` and `parse_timestamp` in `ml/ml_runtime_common.py`. TypeScript refusals are
`ConflictError`s reading `Row i: invalid <column>: <fixed reason>`; Python refusals are `invalid_media` or
`invalid_time_series`. No message quotes the cell.

- **Base64:** the standard alphabet by the grammar `^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$` (no
  `data:` prefix, URL-safe alphabet, whitespace or inner padding), at most 2,796,204 characters and 2 MiB decoded.
- **Images:** PNG or JPEG, sized by a header walk; pixels are never decoded here. PNG: `IHDR` first with length 13, every
  chunk inside the buffer, the walk ends at `IEND` (bytes after it are never read), and a second `IHDR` is refused. JPEG:
  segments from SOI to the first SOS; `0xFF` fill bytes are allowed; every marker before SOS carries a length of at least 2
  inside the buffer (standalone markers are refused); exactly one start-of-frame marker (C0-C3, C5-C7, C9-CB, CD-CF), whose
  length is 8 + 3 × components (1 to 4). Each side 1 to 4,096, at most 16,777,216 pixels.
- **Audio:** WAV only. The RIFF size equals the byte length minus 8; every chunk lies inside the buffer, and an odd-sized
  chunk is followed by a pad byte (absent only at the very end); other chunk types are skipped; exactly one `fmt ` comes before
  exactly one `data`. PCM 8, 16, 24 or 32-bit, or IEEE float 32-bit, directly or as `WAVE_FORMAT_EXTENSIBLE` (cbSize 22, valid
  bits equal to bits, PCM or IEEE-float SubFormat); 1 or 2 channels; 8,000 to 48,000 Hz; `blockAlign` = channels × bits / 8
  and `byteRate` = rate × `blockAlign`; the data is whole frames lasting 0.1 to 30 seconds; float samples are finite.
  `read_wav` decodes a valid file to float32 samples shaped (frames, channels), PCM scaled to [-1, 1).
- **Walk bound:** a header walk visits at most 10,000 chunks or segments, so a crafted file cannot buy unbounded work.
- **Budgets per request:** at ingest, at most 268,435,456 pixels and 600 seconds of audio; at prediction, 67,108,864 pixels
  and 120 seconds. Audio is counted in whole microseconds, rounded up per clip, so the sum is exact. The `/ml` body limit
  (8,000,000 bytes, `src/ml/routes.ts`) stays the effective bound on what one request carries; the budgets bound the decoded
  size the headers declare.
- **Timestamps:** `YYYY-MM-DDTHH:MM`, optionally `:SS` and up to six fractional digits, then `Z` or `±HH:MM`; at most 40
  characters, ASCII digits only; a real proleptic Gregorian date; hour 0-23, minute and second 0-59; an offset of at most
  23:59; year 1900 to 2199. Naive timestamps are refused. Neither `Date.parse` nor `datetime.fromisoformat` is used, so the
  rules do not depend on the runtime version. `timestampMs` returns UTC epoch milliseconds, `epoch_micros` exact integer
  microseconds, which `split_partitions` sorts by.

`tests/fixtures/media-vectors.json` (generated by `tests/python/make_media_vectors.py`, including the attack cases) lists the
cases both sides must decide alike, verdict and facts.

In M1 no runner reads media yet: `media_common` is exercised only by its tests.

### 19.6 Refusal codes

| Code | Category | Raised by |
| --- | --- | --- |
| `unsupported_task` | `unsupported_capability` | `serve`: a v2 request naming another task, or an algorithm the runner does not know |
| `undefined_metric` | `invalid_request` | a metric with no defined value on the partition (section 19.7) |
| `invalid_time_series` | `invalid_request` | a timestamp that breaks the rules of section 19.5; a chronological split that leaves a partition empty |
| `invalid_media` | `invalid_request` | `media_common`: the base64, image header or WAV rules |
| `pretrained_model_unavailable` | `backend_unavailable` | transformers: the named model is not provisioned under the model root (or the root is unusable) |
| `pretrained_model_mismatch` | `incompatible_artifact` | transformers: the provisioned files break the manifest rules, or the model's recorded digest changed |
| `pretrained_label_mismatch` | `invalid_request` | transformers: the dataset's classes are not labels of the pretrained classifier |

`tests/unit/ml-contract.test.ts` checks that the Python sources emit exactly the TypeScript set.

### 19.7 Metric registry

`METRICS` (`src/ml/metrics.ts`) declares each metric's `tasks`, `direction`, `labels` (whether it needs the target), optional
`needs` (the score the estimator must expose) and optional `diagnostic`. `ml/ml_runtime_common.py` mirrors tasks,
direction, labels and diagnostic, and a unit test compares the TypeScript registry with the dump `tests/python/check_metrics.py`
prints.

| Metric | Tasks | Direction | Needs labels | Notes |
| --- | --- | --- | --- | --- |
| `accuracy`, `precision`, `recall`, `f1` | classification | higher | yes | as in Phase 3 |
| `roc_auc` | classification, anomaly_detection | higher | yes | needs scores; for anomaly detection, anomaly scores against the labels |
| `log_loss` | classification | lower | yes | needs probabilities |
| `mae`, `rmse` | regression, forecasting | lower | yes | |
| `mse` | regression | lower | yes | |
| `r2` | regression | higher | yes | |
| `smape` | forecasting | lower | yes | 0 to 200 percent; a term whose \|actual\| + \|forecast\| is 0 counts 0 |
| `mase` | forecasting | lower | yes | scaled by the training block's seasonal-naive error |
| `average_precision` | anomaly_detection | higher | yes | needs scores |
| `anomaly_precision`, `anomaly_recall`, `anomaly_f1` | anomaly_detection | higher | yes | positive class = anomaly; zero division counts 0 |
| `threshold_calibration_error` | anomaly_detection | lower | no | diagnostic: \|flagged fraction - contamination\| |
| `retrieval_precision_at_1`, `retrieval_mrr` | embedding | higher | yes | leave-one-out cosine neighbours within the partition; relevant = same label; ties rank against the model |
| `mean_pairwise_cosine` | embedding | lower | no | diagnostic: anisotropy |

**Defaults** (`defaultMetrics(task, labelled)`, for a snapshot without `evaluation.metrics`): classification `accuracy`;
regression `mae` and `mse` (the Phase 1 defaults, still exported as `DEFAULT_METRICS`); forecasting `mae` and `rmse`; anomaly
detection `average_precision`, or `threshold_calibration_error` without labels; embedding `retrieval_precision_at_1`, or
`mean_pairwise_cosine` without labels. A task that always has a target has no unlabelled default.

**Refusals.** `assertMetricsSupported(model, metrics, labelled)` refuses an unknown metric, a metric for another task, one that
needs labels on a dataset without a target, and one that needs probabilities or scores the model cannot produce. A
diagnostic describes a model's outputs, not its quality, so experiments, AutoML and ranking refuse it
(`assertRankingMetrics`). AutoML and experiments also refuse a dataset without a target.

**Undefined values** are refused as `undefined_metric`, never reported as NaN: MASE with an empty or zero seasonal-naive
scale; the anomaly metrics on a partition without both normal and anomalous rows; the embedding metrics with fewer than two
rows or a zero or non-finite vector; the retrieval metrics when no row shares its label with another row.

The new metrics are pure Python functions (`smape`, `mase`, `anomaly_roc_auc`, `average_precision`, `anomaly_scores`,
`threshold_calibration_error`, `retrieval_scores`, `mean_pairwise_cosine`) with hand-computed tests. No M1 runner computes
them yet, and `measure_predictions` is unchanged.

### 19.8 Outputs

`validatePredictions(task, predictions, count, environment)` (`src/ml/tasks.ts`) checks every prediction response, in
`RunnerBackend.predict` and again in `MlModelService.predict`, which also covers backends that do not extend `RunnerBackend`.
A mismatch is `invalid_response`, and the prediction record fails.

- **Count:** one prediction per row, or for forecasting exactly `horizon` items whose `step`s run from 1 to `horizon` in
  order and whose `timestamp`s follow section 19.5.
- **Shape:** each item matches the task's output (section 19.3). A classification label is a string of at most 10,000
  characters, a finite number or a boolean.
- **Embeddings:** each vector has exactly `embedding_dimension` numbers. The value is read from the artifact's recorded
  environment and must be spelled as a decimal integer from 1 to 4,096 (`^[1-9][0-9]{0,3}$`); it is never coerced. Training
  fails an embedding attempt whose artifact does not record a valid one.
- **Embedding vectors are not stored.** The prediction record's `output` is `{ digest, shape }` (the canonical-JSON SHA-256 of
  the vectors, and `[rows, dimension]`); only the HTTP response carries `vectors`, once.

### 19.9 Changes visible to existing callers

v1 requests keep their meaning, but these differences reach Phase 3 callers:

- `GET /ml/models` returns the registry in its new shape: each metric has `tasks` (a list) instead of `task`, plus `labels`,
  and `diagnostic` where set; `needs` is unchanged. The new metrics are listed too.
- A prediction batch over the row cap is refused before its cells are checked: an oversized batch with invalid cells now
  reports `Prediction batch exceeds N rows` (409), and a batch over 50,000 rows is a 409 where it was a 400 schema error.
- Prediction resolves the model's catalog definition before checking rows, so a READY model whose algorithm is no longer
  registered gets a 409 instead of a FAILED prediction record. Phase 4 removes no model, so current data cannot reach this.
- `RunnerBackend.predict` checks the count and the task's output shape itself ("Predictions did not match the task
  contract"), and classification outputs must be labels as above rather than any JSON value.
- A dataset schema without a target may have a single column. A chronological split refuses unknown keys, `stratify`
  included, and a random split on a dataset without a target cannot stratify ("Stratification requires a target").

### 19.10 Tests

- `tests/unit/ml-contract.test.ts`: both version constants, and the refusal-code sets equal across the languages.
- `tests/unit/ml-tasks.test.ts`: the task registry, and `contractFor` over `tests/fixtures/contract-table.json`. With
  `ML_TEST_PYTHON`, `tests/python/contract_table.py` has Python judge the snapshots TypeScript serialises and raw envelopes,
  and compares partition sizes with scikit-learn and the chronological rule.
- `tests/unit/ml-media.test.ts`: every shared vector, and the TypeScript-only rules (limits, budgets).
- `tests/unit/ml-python-runtime.test.ts`: the named checks of `tests/python/check_contract.py` (`serve` semantics and stdout
  protection), `check_partitions.py`, `check_metrics.py` (every new metric and edge case, and the registry equality) and
  `check_media.py` (the shared vectors through `media_common` and `parse_timestamp`).
- `tests/unit/ml-legacy-neural.test.ts`: a frozen copy of the Phase 3 PyTorch runner trains and checkpoints; the current
  runner predicts identically from its artifact and resumes its checkpoint.
- `tests/unit/ml-backend.test.ts`, `ml-v2-backend.test.ts` and `ml-runner-protocol.test.ts`: the v2-only refusals, the
  unchanged v1 envelope, a stored boolean-target ridge that still validates and predicts, and the echo rule.
- `tests/integration/ml.test.ts`: the processor and prediction paths for forecasting and embedding, and the row caps, against a
  test backend that declares those tasks.
- The conformance suite (section 11): truthful `tasks` and `inputs`, and the raw-envelope checks.

### 19.11 Not built in M1

M1 built the vocabulary only. Later milestones add the backends that use it; section 13 lists them as they land. At the end of
M1, no backend implemented a task other than classification and regression or took a non-tabular feature column, and no runner
decoded media or computed the new metrics. `transcription` and its metrics are not defined yet. No database migration was
needed: the new snapshot fields and the embedding digest fit the existing JSON columns.
