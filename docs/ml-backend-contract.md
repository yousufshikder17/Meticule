# Meticule ML Backend Contract v1

This is the internal contract between the Meticule ML platform and the execution backends behind it (scikit-learn, PyTorch
and TensorFlow today). It is version `meticule-ml-backend/v1`. It is **internal**: there is no public plugin SDK, no dynamic
loading of third-party code, and nothing here is promised to external authors. The contract exists so that adding an internal
backend is a narrow, testable change, and so that a future v2 can coexist with v1 instead of being misread.

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

The version is a string constant, `ML_BACKEND_CONTRACT` in `src/ml/contract.ts`, mirrored as `ML_BACKEND_CONTRACT` in
`ml/ml_runtime_common.py`. It is checked in four places:

- every runner request carries `contract` and a runner refuses any other value (`unsupported_contract`) before doing work;
- every successful runner response echoes `contract`, and the TypeScript client refuses a response that does not carry it
  (`runtime_incompatible`);
- `tests/unit/ml-contract.test.ts` fails if the two sides declare different versions or emit different refusal codes;
- the conformance suite sends a request from another contract and asserts the refusal.

A change that alters the meaning of a field, the command set, the failure taxonomy or artifact semantics is a new version. A
v2 would be introduced by adding a constant and teaching the client to accept both; nothing in v1 is written to make that
impossible.

## 3. Operations

Every backend extends `RunnerBackend` (`src/ml/runner-backend.ts`), which implements the required operations below. Neural
backends extend it again through `NeuralRunnerBackend` and add the iterative pair.

| Operation | Required | Input | Output | Cancellation | Errors |
| --- | --- | --- | --- | --- | --- |
| `validate(snapshot)` | Yes | A parsed training snapshot | nothing, or `ConflictError` | Not applicable (no runner) | `invalid_request` |
| `prepare(snapshot, rows, signal)` | Yes | Snapshot and dataset rows | Disjoint train, validation and test indices; optional CV folds; environment | Aborting kills the runner | Runner categories |
| `train(snapshot, rows, prepared, signal)` | Yes | As above, plus the prepared split | Artifact bytes, format, environment, resolved hyperparameters, measured timings | Aborting kills the runner | `training_failed` and others |
| `evaluate` / `evaluateDetailed` | Yes (`evaluate`); `evaluateDetailed` refines it | Snapshot, rows, prepared split, trained model | Metrics for every partition the snapshot asks for | Aborting kills the runner | `evaluation_failed` |
| `predict(snapshot, artifact, rows, environment, signal)` | Yes | Stored artifact bytes and their recorded environment | One JSON prediction per row | Aborting kills the runner | `prediction_failed` and `incompatible_artifact` |
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
- **Evaluation** computes the metrics the snapshot asks for, on the partitions it names, with directions declared once in
  `ml/ml_runtime_common.py` and mirrored in `src/ml/metrics.ts`. Ranking reads the direction, never the metric name.
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

Models carry their own capabilities in the catalog (`ModelCapabilities`): scores, scale sensitivity, seeding, `iterative`,
and optionally a narrower `devices` list and `mixedPrecision`. The rules, enforced by `RunnerBackend.validate` and the
conformance suite:

- a model without its own device list inherits the backend's;
- a model that narrows the device list must stay inside the backend's;
- a model that is iterative requires an iterative backend;
- a model that supports mixed precision requires the backend to declare `mixed`;
- a training policy (checkpoints, early stopping) is refused on a non-iterative backend;
- early stopping needs a validation partition and an iterative model.

The device and precision vocabulary is declared once, in `src/ml/domain.ts` (`DEVICES`, `PRECISIONS`), and the TypeScript types
are derived from it.

Future capabilities (text, vision, audio, time series, embeddings, generative models, fine-tuning, PEFT, distributed or remote
training, ONNX-style inference) are **not implemented**. They would be new optional fields on `BackendCapabilities` or on the
model definition, and the suite would learn to check them. Nothing in v1 needs to change for them to be added, except where
section 14 names a blocking assumption.

## 6. Failure taxonomy

Backend and runner failures are classified into a bounded set of categories (`ML_FAILURE_CATEGORIES`, `src/ml/contract.ts`):

| Category | Meaning |
| --- | --- |
| `invalid_request` | A snapshot or request was refused by validation (`ConflictError`). |
| `unsupported_capability` | The backend does not declare the device, precision, operation or training mode requested. |
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
`invalid_checkpoint`, `cuda_unavailable`, `cuda_unsupported`, `mixed_requires_cuda` and `unsupported_device_or_precision`. A runner
prints one of them on stderr as its last line, as JSON. The client reads only those codes; any other text is discarded. A runner
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

Request envelope: `{ "command", "contract", "snapshot", "rows", ...command fields }`. Commands are `prepare`, `train`,
`train_chunk`, `export`, `evaluate` and `predict`. The sklearn runner also answers `catalog`.

Response: the command's JSON with `contract` set to the version. Stdout is capped at 64 MiB; a larger response kills the runner.

Timeouts (`ML_TIMEOUT_MS`, default 120 s) and cancellation kill the child. A child never outlives its call.

Environment: runners start from an allowlist (`src/ml/runner-process.ts`). They receive `PATH` and the system variables the
interpreter and native libraries need, plus the thread limits and `PYTHONDONTWRITEBYTECODE`. They never receive the database URL,
the JWT secret, provider keys, `PYTHONPATH` or `PYTHONHOME`. A unit test plants a secret and asserts the child cannot see it.

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
Section 14 records the JAX thought experiment that tests this claim.

Not allowed: a public plugin loader, executing third-party code from a configuration value, or a backend that persists its own
state.

## 11. Conformance suite

`tests/conformance/contract-suite.ts` is one suite, written against platform semantics, and each internal backend runs it through
a thin file. It checks, where the backend's declared capabilities say a check applies:

- identity and declared capabilities; truthfulness about iteration and models;
- validation refusals before any runner starts: an unknown algorithm, a model from another backend, an uninterpreted operation,
  an undeclared device, an undeclared mixed precision, a policy on a non-iterative backend;
- preparation (disjoint, complete partitions), training, evaluation, prediction, determinism for the same seed and split, and a
  regression lifecycle;
- artifact format and provenance fields; refusal of a foreign runtime, corrupt bytes, and a provenance-only difference that must
  load;
- cancellation before a call starts; a runner killed at its time budget and reported as `timed_out`; a request from another
  contract refused, and the current contract accepted;
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

| | scikit-learn | PyTorch | TensorFlow |
| --- | --- | --- | --- |
| Class | `SklearnBackend` | `PytorchBackend` | `TensorflowBackend` |
| Artifact format | `sklearn-pickle-v1` | `pytorch-state-dict-v1` | `tensorflow-variables-v1` |
| Checkpoint format | none (one-shot) | `pytorch-checkpoint-v1` | `tensorflow-checkpoint-v1` |
| Devices | `cpu` | `cpu`, `cuda` | `cpu` |
| Precisions | `fp32` | `fp32`, `mixed` (CUDA only) | `fp32` |
| Iterative | no | yes | yes |
| Preprocessing | eight operations, including `variance_threshold` | seven numeric and categorical operations | seven numeric and categorical operations |
| Interpreter variable | `ML_PYTHON` | `ML_PYTORCH_PYTHON` | `ML_TENSORFLOW_PYTHON` |
| Conformance file | `tests/conformance/sklearn.test.ts` | `tests/conformance/pytorch.test.ts` | `tests/conformance/tensorflow.test.ts` |

Known CI limit. The test job provides scikit-learn and PyTorch. The TensorFlow job provides only TensorFlow, and it runs the
TensorFlow conformance file. The pairs involving TensorFlow in `cross-backend.test.ts` therefore run locally when all three
interpreters are configured, not in hosted CI. Similarly, `tests/integration/ml-matrix.test.ts` (the Phase 3E matrix) needs all three
interpreters and is not run by either CI job. Closing this needs one venv that hosts all three frameworks, which is a CI design
change and is not made in this phase.

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
| Preprocessing operation registry in `preprocessing.ts` is shaped around tabular scopes (numeric, categorical, global) | D | Documented. A text, image or time-series backend would need a different registry. This is the largest v2 candidate. |
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
| Embeddings, NLP text classification | Partly. A model with a text feature needs a text preprocessing vocabulary, which is a blocking registry assumption (section 14). A classifier head over embeddings works as a neural backend. |
| Vision, audio, time series, anomaly detection | Blocking for the preprocessing registry. The runner protocol carries JSON rows and a single numeric matrix; image and audio inputs need a new input shape, which is a v2 change. |
| Hugging Face-style models, XGBoost, LightGBM | A one-shot backend is supported as sklearn is. An iterative one is supported as the neural backends are. Their weights need the artifact format to be opaque bytes, which it already is. |
| Richer MLOps, deployment targets, ONNX | Inference artifacts are opaque bytes behind a declared format, so an ONNX format is an additional format and runner. The inference lifecycle is outside the backend contract. |
| Lineage, monitoring, promotion and rollback, advanced HPO | Outside the backend contract by design. The registry and experiments already own them. |
| LoRA and PEFT, foundation-model fine-tuning | Needs a checkpoint that stores adapter weights and a base model reference. The checkpoint is opaque and the fingerprint covers the configuration, so this is an optional capability plus a new checkpoint format. No blocking assumption found. |
| Distributed and remote training | Blocking for the runner boundary. A runner is one local process with no shared state; a remote or distributed backend needs an execution transport and a lease model for its workers. This is a v2 concern. |
| Model routing, distillation, reinforcement learning | Outside the single-attempt model. Routing belongs to the inference layer; distillation and RL need multi-attempt state the platform does not yet model. |

**What likely requires a contract v2:** a non-tabular input shape; a remote or distributed execution transport; a training loop that
is not expressed as epochs over rows.

## 16. Adding JAX (thought experiment only)

JAX was used as a design test. Nothing was installed or implemented.

A JAX backend would need, following section 10:

- a Python runner that implements the six commands with JAX computation and calls `serve`;
- a JAX requirements file;
- a set of model definitions and a device list, and a capability declaration (`jax` would likely be `cpu`, with GPU as an
  explicit device when a CUDA-enabled install is present);
- a backend class, a configuration entry and a conformance file.

It would not require edits to persistence, the training processor, the registry, evaluation, routing, the HTTP layer or unrelated
core code. Its checkpoint would be an opaque, versioned format; its artifact would be an opaque, versioned format. Those are the
two things the platform never inspects.

The honest caveat is that JAX's functional state (parameters and optimizer state held as pytrees) fits the checkpoint format
without change, but its random-number handling is explicit in a way the current runners do not exercise, so the determinism check
would need a JAX-specific expectation. That is a difference in the backend, expressed as a check, not a contract change.

Conclusion: the architecture is healthy for this case.

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
policy beyond "the contract constant must match"; and the sklearn pickle boundary is a residual risk that an external author should
not inherit without a decision about signing.
