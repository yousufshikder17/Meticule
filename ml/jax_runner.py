"""JAX CPU backend: bounded feed-forward tabular networks written directly in JAX, over the Meticule runner protocol.

No database, scheduler or client paths. Commands match pytorch_runner.py (prepare, train, train_chunk, export, evaluate,
predict). The network, its gradients (jax.value_and_grad), the jitted training step and the optimizers are plain JAX;
preprocessing and targets come from neural_common, so its fitted state is JSON. Artifacts and checkpoints are NumPy archives
read with allow_pickle=False.

Randomness is explicit: every key derives from the snapshot seed. Epoch e shuffles with fold_in(shuffle, e), and batch b of
epoch e drops out with fold_in(fold_in(dropout, e), b). No random state is carried between chunks, so a run split into
chunks equals an uninterrupted run by construction.
"""
import os

# CPU only and float32, single-threaded XLA so results are deterministic. Set before jax is imported; assigned rather than
# defaulted, so an inherited variable can never move this backend to another platform or precision.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "0"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import base64
import io
import json
import sys
import time
import zipfile

import jax
import jax.numpy as jnp
import jaxlib
import joblib
import numpy as np
import scipy
import scipy.special
import sklearn
import threadpoolctl

import ml_runtime_common as common
import neural_common
from neural_common import LIMITS, MODELS, TASK_LOSS, fingerprint, fit_preprocessing, resolve_hyperparameters, target_vector, transform

BACKEND = "jax"
ARTIFACT_FORMAT = "jax-params-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-jax-v1"
SUPPORTED_PROTOCOLS = {PROTOCOL: 1}
SUPPORTED_SCHEMAS = {1}
RUNTIME_KEYS = ("jax", "jaxlib", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": SUPPORTED_PROTOCOLS,
            "schemas": SUPPORTED_SCHEMAS, "runtime_keys": RUNTIME_KEYS}
CHECKPOINT_FORMAT = "jax-checkpoint-v1"
CHECKPOINT_SCHEMA = 1
# Optimizer constants (PyTorch defaults, so the three neural backends mean the same thing by "adam" and "adamw").
BETA1, BETA2, EPSILON, ADAMW_DECAY = 0.9, 0.999, 1e-8, 0.01
MAX_METADATA_BYTES = 16 * 1024 * 1024

RunnerError = common.RunnerError
ACTIVATIONS = {"relu": jax.nn.relu, "gelu": lambda v: jax.nn.gelu(v, approximate=False), "tanh": jnp.tanh}


def environment():
    return common.base_environment([__file__, common.__file__, neural_common.__file__], BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                    "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def check_compatible(recorded, current):
    common.check_compatible(recorded, current, CONTRACT)


def resolve_device(spec):
    """CPU only (no CUDA jaxlib exists for native Windows). A CUDA request is refused, never run on CPU instead."""
    training = spec.get("training") or {}
    requested, precision = training.get("device", "cpu"), training.get("precision", "fp32")
    if requested == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported by the JAX backend")
    if requested != "cpu" or precision != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")
    return requested, precision


def device_fields():
    """Where the run executed, as JAX reports it. Recorded for audit; deliberately not part of compatibility."""
    return {"device_requested": "cpu", "device_actual": jax.default_backend(), "cuda_runtime": "none", "device_name": "none",
            "device_capability": "none", "jax_platform": jax.default_backend(), "jax_devices": str(len(jax.devices("cpu")))}


# ---- network: parameters are a list of {"w", "b"} layers; everything below is a pure function ------------------------

def layer_shapes(architecture, input_dim, output_dim):
    widths = [input_dim, *architecture["hiddenLayers"], output_dim]
    return [((widths[i], widths[i + 1]), (widths[i + 1],)) for i in range(len(widths) - 1)]


def init_params(key, architecture, input_dim, output_dim):
    """Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases, the PyTorch nn.Linear default."""
    params = []
    for i, (w_shape, b_shape) in enumerate(layer_shapes(architecture, input_dim, output_dim)):
        bound = 1.0 / np.sqrt(w_shape[0]) if w_shape[0] else 0.0
        kw, kb = jax.random.split(jax.random.fold_in(key, i))
        params.append({"w": jax.random.uniform(kw, w_shape, jnp.float32, -bound, bound),
                       "b": jax.random.uniform(kb, b_shape, jnp.float32, -bound, bound)})
    return params


def forward(params, x, activation, dropout, key=None):
    """Hidden layers apply the activation and (when `key` is given, i.e. training) inverted dropout."""
    for i, layer in enumerate(params[:-1]):
        x = ACTIVATIONS[activation](x @ layer["w"] + layer["b"])
        if key is not None and dropout > 0.0:
            keep = jax.random.bernoulli(jax.random.fold_in(key, i), 1.0 - dropout, x.shape)
            x = jnp.where(keep, x / (1.0 - dropout), 0.0)
    return x @ params[-1]["w"] + params[-1]["b"]


def loss_value(name, output, y):
    if name == "cross_entropy":
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(output), y[:, None], axis=1))
    if name == "mse":
        return jnp.mean((output - y) ** 2)
    return jnp.mean(jnp.abs(output - y))


def init_optimizer(name, params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"step": 0} if name == "sgd" else {"step": 0, "m": zeros, "v": jax.tree.map(jnp.zeros_like, params)}


def optimizer_update(name, lr, params, grads, state):
    """Hand-written sgd, adam and adamw (decoupled weight decay) over parameter and state pytrees."""
    step = state["step"] + 1
    if name == "sgd":
        return jax.tree.map(lambda p, g: p - lr * g, params, grads), {"step": step}
    m = jax.tree.map(lambda m, g: BETA1 * m + (1.0 - BETA1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: BETA2 * v + (1.0 - BETA2) * g * g, state["v"], grads)
    c1, c2 = 1.0 - BETA1 ** step, 1.0 - BETA2 ** step
    if name == "adamw":
        params = jax.tree.map(lambda p: p * (1.0 - lr * ADAMW_DECAY), params)
    params = jax.tree.map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + EPSILON), params, m, v)
    return params, {"step": step, "m": m, "v": v}


def make_step(hp):
    """The jitted training step for one configuration: loss, gradients by autodiff, and the optimizer update."""
    activation, dropout, loss_name, optimizer, lr = hp["activation"], float(hp["dropout"]), hp["loss"], hp["optimizer"], float(hp["learningRate"])

    def objective(params, x, y, key):
        return loss_value(loss_name, forward(params, x, activation, dropout, key), y)

    @jax.jit
    def step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(objective)(params, x, y, key)
        params, opt_state = optimizer_update(optimizer, lr, params, grads, opt_state)
        return params, opt_state, loss

    @jax.jit
    def evaluate_loss(params, x, y):
        return loss_value(loss_name, forward(params, x, activation, dropout), y)

    return step, evaluate_loss


def derive_keys(seed):
    base = jax.random.key(int(seed))
    init, shuffle, dropout = jax.random.split(base, 3)
    return base, init, shuffle, dropout


# ---- a training run: everything carried between chunks ---------------------------------------------------------------

class Run:
    def __init__(self, task, hp, params, opt_state, preprocessing, target_info, architecture, input_dim, output_dim, seed, x, y, val_x, val_y):
        self.task, self.hp, self.params, self.opt_state = task, hp, params, opt_state
        self.preprocessing, self.target_info, self.architecture = preprocessing, target_info, architecture
        self.input_dim, self.output_dim, self.seed = input_dim, output_dim, seed
        self.x, self.y, self.val_x, self.val_y = x, y, val_x, val_y
        self.completed, self.elapsed = 0, 0.0
        self.best = None  # {"epoch", "value", "params"}
        self.last_train_loss = self.last_validation_loss = None
        _, _, self.shuffle_key, self.dropout_key = derive_keys(seed)
        self.step, self.evaluate_loss = make_step(hp)


def as_targets(task, values):
    return values.astype(np.int32) if task == "classification" else values.astype(np.float32)


def build_data(task, spec, rows, prepared, preprocessing, target_info):
    target = spec["dataset"]["schema"]["target"]
    fit_idx, val_idx = prepared["indices"]["train"], prepared["indices"]["validation"]
    x = transform(preprocessing, [rows[i] for i in fit_idx])
    y = as_targets(task, target_vector(task, [rows[i][target] for i in fit_idx], target_info))
    if not val_idx:
        return x, y, None, None
    val_x = transform(preprocessing, [rows[i] for i in val_idx])
    val_y = as_targets(task, target_vector(task, [rows[i][target] for i in val_idx], target_info))
    return x, y, val_x, val_y


def fresh_run(spec, rows, prepared):
    """Fits preprocessing on training rows only and initialises parameters and optimizer state from the seed."""
    hp = resolve_hyperparameters(spec)
    task = MODELS[spec["algorithm"]]
    target = spec["dataset"]["schema"]["target"]
    fit_idx = prepared["indices"]["train"]
    if not fit_idx:
        raise ValueError("Training partition is empty")
    resolve_device(spec)
    preprocessing = fit_preprocessing(spec, rows, fit_idx)
    raw = [rows[i][target] for i in fit_idx]
    if task == "classification":
        classes = sorted(set(raw))
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes in the training partition")
        target_info, output_dim = {"classes": classes}, len(classes)
    else:
        values = np.array(raw, dtype=float)
        target_info, output_dim = {"mean": float(values.mean()), "std": float(values.std()) or 1.0}, 1
    architecture = {"hiddenLayers": hp["hiddenLayers"], "activation": hp["activation"], "dropout": float(hp["dropout"])}
    input_dim = int(transform(preprocessing, [rows[fit_idx[0]]]).shape[1])
    _, init_key, _, _ = derive_keys(spec["seed"])
    params = init_params(init_key, architecture, input_dim, output_dim)
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, preprocessing, target_info)
    return Run(task, hp, params, init_optimizer(hp["optimizer"], params), preprocessing, target_info, architecture, input_dim, output_dim,
               spec["seed"], x, y, val_x, val_y)


def run_epochs(run, count):
    """Runs `count` epochs. Shuffling and dropout depend only on (seed, absolute epoch, batch), never on earlier chunks."""
    summaries, n, size = [], len(run.x), run.hp["batchSize"]
    for _ in range(count):
        started = time.perf_counter()
        epoch = run.completed + 1
        order = np.asarray(jax.random.permutation(jax.random.fold_in(run.shuffle_key, epoch), n))
        epoch_dropout, total = jax.random.fold_in(run.dropout_key, epoch), 0.0
        for number, start in enumerate(range(0, n, size)):
            batch = order[start:start + size]
            run.params, run.opt_state, loss = run.step(run.params, run.opt_state, run.x[batch], run.y[batch], jax.random.fold_in(epoch_dropout, number))
            total += float(loss) * len(batch)
        train_loss = total / n
        validation_loss = None if run.val_x is None else float(run.evaluate_loss(run.params, run.val_x, run.val_y))
        elapsed = time.perf_counter() - started
        run.completed, run.elapsed = epoch, run.elapsed + elapsed
        run.last_train_loss, run.last_validation_loss = train_loss, validation_loss
        if not (np.isfinite(train_loss) and (validation_loss is None or np.isfinite(validation_loss))):
            raise ValueError("Training diverged to a non-finite loss")
        if validation_loss is not None and (run.best is None or validation_loss < run.best["value"]):
            run.best = {"epoch": epoch, "value": validation_loss, "params": params_to_numpy(run.params)}
        summaries.append({"epoch": epoch, "trainLoss": train_loss, "validationLoss": validation_loss, "elapsedSeconds": elapsed})
    return summaries


# ---- NumPy archives: named float32 arrays whose names and shapes are fixed by the recorded architecture ----------------

def params_to_numpy(params):
    return [{"w": np.asarray(layer["w"], dtype=np.float32), "b": np.asarray(layer["b"], dtype=np.float32)} for layer in params]


def params_from_numpy(layers):
    return [{"w": jnp.asarray(layer["w"]), "b": jnp.asarray(layer["b"])} for layer in layers]


def tree_arrays(prefix, layers):
    return {f"{prefix}/{i}/{part}": np.asarray(layer[part], dtype=np.float32) for i, layer in enumerate(layers) for part in ("w", "b")}


def write_archive(metadata, arrays):
    buffer = io.BytesIO()
    np.savez(buffer, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


def open_archive(encoded, failure):
    """Decodes a NumPy archive and its metadata. Object arrays (pickles) are refused by allow_pickle=False."""
    try:
        archive = np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile) or "metadata" not in archive.files:
            raise ValueError
        if archive.zip.getinfo("metadata.npy").file_size > MAX_METADATA_BYTES:
            raise ValueError
        raw = archive["metadata"]
        if raw.dtype.kind != "U" or raw.shape != ():
            raise ValueError
        metadata = json.loads(str(raw))
        if not isinstance(metadata, dict):
            raise ValueError
        return archive, metadata
    except Exception:
        raise failure("unreadable archive") from None


def read_layers(archive, prefix, shapes, failure):
    """Reads exactly the arrays the architecture defines, checking each entry's size before it is decompressed."""
    layers = []
    for i, (w_shape, b_shape) in enumerate(shapes):
        layer = {}
        for part, shape in (("w", w_shape), ("b", b_shape)):
            name = f"{prefix}/{i}/{part}"
            try:
                if archive.zip.getinfo(f"{name}.npy").file_size > int(np.prod(shape)) * 4 + 1024:
                    raise ValueError
                value = archive[name]
            except Exception:
                raise failure("unreadable archive") from None
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise failure("archive arrays do not match the recorded architecture")
            layer[part] = value
        layers.append(layer)
    return layers


def expected_names(prefixes, shapes):
    return {"metadata"} | {f"{p}/{i}/{part}" for p in prefixes for i in range(len(shapes)) for part in ("w", "b")}


def checked_architecture(metadata, failure):
    """The recorded architecture must itself be within the allowlist before it sizes anything."""
    try:
        architecture = metadata["architecture"]
        resolve_hyperparameters({"algorithm": "mlp_classifier", "hyperparameters": {"hiddenLayers": architecture["hiddenLayers"],
                                 "activation": architecture["activation"], "dropout": architecture["dropout"]}})
        input_dim, output_dim = int(metadata["inputDim"]), int(metadata["outputDim"])
        if not 0 <= input_dim <= 1_000_000 or not 1 <= output_dim <= 100_000:
            raise ValueError
        return architecture, input_dim, output_dim
    except Exception:
        raise failure("archive architecture is invalid") from None


# ---- checkpoints -------------------------------------------------------------------------------------------------------

def save_run(run, spec, prepared, env):
    metadata = {"schema": CHECKPOINT_SCHEMA, "format": CHECKPOINT_FORMAT, "backend": BACKEND, "algorithm": spec["algorithm"], "task": run.task,
                "fingerprint": fingerprint(spec, run.hp, prepared), "completedEpoch": run.completed, "device": "cpu", "precision": "fp32",
                "architecture": run.architecture, "inputDim": run.input_dim, "outputDim": run.output_dim, "target": run.target_info,
                "preprocessing": run.preprocessing, "hyperparameters": run.hp, "optimizerStep": int(run.opt_state["step"]),
                "elapsedSeconds": run.elapsed, "lastTrainLoss": run.last_train_loss, "lastValidationLoss": run.last_validation_loss,
                "best": None if run.best is None else {"epoch": run.best["epoch"], "value": run.best["value"]}, "environment": env}
    numpy_params = params_to_numpy(run.params)
    arrays = {**tree_arrays("params", numpy_params), "key_data": np.asarray(jax.random.key_data(derive_keys(run.seed)[0]), dtype=np.uint32)}
    if run.hp["optimizer"] != "sgd":
        arrays.update(tree_arrays("opt_m", params_to_numpy(run.opt_state["m"])))
        arrays.update(tree_arrays("opt_v", params_to_numpy(run.opt_state["v"])))
    if run.best is not None:
        arrays.update(tree_arrays("best", run.best["params"]))
    return write_archive(metadata, arrays)


def load_checkpoint(checkpoint, current_env):
    """Validates a checkpoint's format, environment and every array before a run is rebuilt from it."""
    archive, meta = open_archive(checkpoint, common.InvalidCheckpoint)
    if meta.get("format") != CHECKPOINT_FORMAT or meta.get("schema") != CHECKPOINT_SCHEMA or meta.get("backend") != BACKEND:
        raise common.IncompatibleCheckpoint("unsupported checkpoint format or schema")
    try:
        common.check_compatible(meta.get("environment"), current_env, CONTRACT)
    except common.IncompatibleArtifact as error:
        raise common.IncompatibleCheckpoint(str(error)) from None
    architecture, input_dim, output_dim = checked_architecture(meta, common.InvalidCheckpoint)
    shapes = layer_shapes(architecture, input_dim, output_dim)
    hp = meta.get("hyperparameters") or {}
    prefixes = ["params"] + (["opt_m", "opt_v"] if hp.get("optimizer") != "sgd" else []) + (["best"] if meta.get("best") is not None else [])
    if set(archive.files) != expected_names(prefixes, shapes) | {"key_data"}:
        raise common.InvalidCheckpoint("malformed checkpoint")
    layers = {prefix: read_layers(archive, prefix, shapes, common.InvalidCheckpoint) for prefix in prefixes}
    key_data = archive["key_data"]
    if key_data.dtype != np.uint32 or key_data.shape != (2,):
        raise common.InvalidCheckpoint("malformed checkpoint")
    return layers, key_data, meta


def restore_run(checkpoint, spec, rows, prepared, current_env):
    layers, key_data, meta = load_checkpoint(checkpoint, current_env)
    hp = resolve_hyperparameters(spec)
    if meta["fingerprint"] != fingerprint(spec, hp, prepared):
        raise common.IncompatibleCheckpoint("Checkpoint belongs to a different training configuration")
    if not np.array_equal(key_data, np.asarray(jax.random.key_data(derive_keys(spec["seed"])[0]))):
        raise common.IncompatibleCheckpoint("Checkpoint keys do not derive from the snapshot seed")
    resolve_device(spec)
    task = MODELS[spec["algorithm"]]
    params = params_from_numpy(layers["params"])
    opt_state = {"step": int(meta["optimizerStep"])}
    if hp["optimizer"] != "sgd":
        opt_state.update({"m": params_from_numpy(layers["opt_m"]), "v": params_from_numpy(layers["opt_v"])})
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, meta["preprocessing"], meta["target"])
    run = Run(task, hp, params, opt_state, meta["preprocessing"], meta["target"], meta["architecture"], meta["inputDim"], meta["outputDim"],
              spec["seed"], x, y, val_x, val_y)
    run.completed, run.elapsed = meta["completedEpoch"], meta["elapsedSeconds"]
    run.last_train_loss, run.last_validation_loss = meta["lastTrainLoss"], meta["lastValidationLoss"]
    if meta["best"] is not None:
        run.best = {"epoch": meta["best"]["epoch"], "value": meta["best"]["value"], "params": layers["best"]}
    return run


# ---- inference artifacts ---------------------------------------------------------------------------------------------

def bundle_from_run(run, params=None):
    return {"params": params_to_numpy(run.params) if params is None else params, "preprocessing": run.preprocessing, "task": run.task,
            "target": run.target_info, "architecture": run.architecture, "input_dim": run.input_dim, "output_dim": run.output_dim}


def serialize(bundle, env):
    metadata = {"schema": ARTIFACT_SCHEMA, "backend": BACKEND, "task": bundle["task"], "architecture": bundle["architecture"],
                "inputDim": bundle["input_dim"], "outputDim": bundle["output_dim"], "target": bundle["target"],
                "preprocessing": bundle["preprocessing"], "environment": env}
    return write_archive(metadata, tree_arrays("params", bundle["params"]))


def load_bundle(artifact, current_env):
    """Checks the environment and the recorded architecture before any parameter array is read."""
    archive, metadata = open_archive(artifact, common.IncompatibleArtifact)
    check_compatible(metadata.get("environment"), current_env)
    if metadata.get("task") not in TASK_LOSS or metadata.get("backend") != BACKEND or metadata.get("schema") != ARTIFACT_SCHEMA:
        raise common.IncompatibleArtifact("unsupported artifact task")
    architecture, input_dim, output_dim = checked_architecture(metadata, common.IncompatibleArtifact)
    shapes = layer_shapes(architecture, input_dim, output_dim)
    if set(archive.files) != expected_names(["params"], shapes):
        raise common.IncompatibleArtifact("malformed artifact")
    return {"params": read_layers(archive, "params", shapes, common.IncompatibleArtifact), "preprocessing": metadata["preprocessing"],
            "task": metadata["task"], "target": metadata["target"], "architecture": architecture, "input_dim": input_dim, "output_dim": output_dim}


def predict_bundle(bundle, rows):
    """Normalised output shared with the other runners: predictions, class probabilities and training labels."""
    if not rows:
        empty = np.zeros((0, bundle["output_dim"]))
        if bundle["task"] == "classification":
            return {"prediction": np.array([]), "proba": empty, "classes": np.array(bundle["target"]["classes"])}
        return {"prediction": np.zeros(0), "proba": None, "classes": None}
    x = jnp.asarray(transform(bundle["preprocessing"], rows))
    output = forward(params_from_numpy(bundle["params"]), x, bundle["architecture"]["activation"], 0.0)
    if bundle["task"] == "classification":
        # Softmax in float64 from the float32 logits, so each row sums to one at the precision metrics check.
        probabilities = scipy.special.softmax(np.asarray(output, dtype=np.float64), axis=1)
        classes = bundle["target"]["classes"]
        return {"prediction": np.array([classes[i] for i in probabilities.argmax(axis=1)]), "proba": probabilities, "classes": np.array(classes)}
    values = np.asarray(output, dtype=float).reshape(-1) * bundle["target"]["std"] + bundle["target"]["mean"]
    return {"prediction": values, "proba": None, "classes": None}


def train_network(spec, rows, fit_indices):
    """One-shot training for cross-validation folds: a fresh run trained for every requested epoch."""
    run = fresh_run(spec, rows, {"indices": {"train": fit_indices, "validation": []}})
    run_epochs(run, run.hp["epochs"])
    return bundle_from_run(run)


def measure(names, y, output):
    if output["proba"] is not None:
        return common.measure_predictions(names, y, output["prediction"], proba=output["proba"], scores=output["proba"], classes=output["classes"])
    return common.measure_predictions(names, y, output["prediction"])


def execute(request):
    env = environment()
    spec = request["snapshot"]
    rows = request["rows"]
    if request.get("expectedEnvironment") is not None:
        check_compatible(request["expectedEnvironment"], env)
    command = request["command"]
    target = spec["dataset"]["schema"]["target"]
    if command == "prepare":
        resolve_device(spec)
        y = np.array([row[target] for row in rows])
        return {**common.split_partitions(spec, rows, y), "environment": env}
    if command == "predict":
        return {"predictions": predict_bundle(load_bundle(request["artifact"], env), rows)["prediction"].tolist()}
    if command == "train":
        started = time.perf_counter()
        run = fresh_run(spec, rows, {"indices": request["prepared"]["indices"]})
        run_epochs(run, run.hp["epochs"])
        env_run = {**env, **device_fields(), "precision": "fp32"}
        performance = {"fitSeconds": time.perf_counter() - started, "epochsCompleted": float(run.completed), "finalTrainLoss": float(run.last_train_loss)}
        if run.last_validation_loss is not None:
            performance["finalValidationLoss"] = run.last_validation_loss
        return {"artifact": serialize(bundle_from_run(run), env_run), "format": ARTIFACT_FORMAT, "environment": env_run,
                "resolvedHyperparameters": run.hp, "performance": performance}
    if command == "train_chunk":
        # One bounded chunk of epochs. Meticule decides when to call again and what to persist.
        prepared = {"indices": request["prepared"]["indices"]}
        epochs = int(request["epochs"])
        if not 1 <= epochs <= LIMITS["maxEpochs"]:
            raise ValueError("Chunk size out of range")
        run = fresh_run(spec, rows, prepared) if request.get("state") is None else restore_run(request["state"], spec, rows, prepared, env)
        summaries = run_epochs(run, epochs)
        env_run = {**env, **device_fields(), "precision": "fp32"}
        return {"checkpoint": save_run(run, spec, prepared, env_run), "checkpointFormat": CHECKPOINT_FORMAT, "checkpointSchema": CHECKPOINT_SCHEMA,
                "completedEpoch": run.completed, "epochs": summaries, "environment": env_run}
    if command == "export":
        # The inference artifact from a checkpoint, from either the last or the best-epoch parameters.
        layers, _, meta = load_checkpoint(request["checkpoint"], env)
        if meta["algorithm"] != spec["algorithm"] or meta["target"] is None:
            raise common.IncompatibleCheckpoint("Checkpoint does not match the snapshot")
        params = layers["best"] if request.get("restoreBest") and "best" in layers else layers["params"]
        bundle = {"params": params, "preprocessing": meta["preprocessing"], "task": meta["task"], "target": meta["target"],
                  "architecture": meta["architecture"], "input_dim": meta["inputDim"], "output_dim": meta["outputDim"]}
        performance = {"fitSeconds": float(meta["elapsedSeconds"]), "epochsCompleted": float(meta["completedEpoch"]), "finalTrainLoss": float(meta["lastTrainLoss"])}
        if meta["lastValidationLoss"] is not None:
            performance["finalValidationLoss"] = meta["lastValidationLoss"]
        if meta["best"] is not None:
            performance["bestEpoch"] = float(meta["best"]["epoch"])
        env_export = {**env, **device_fields(), "precision": "fp32"}
        return {"artifact": serialize(bundle, env_export), "format": ARTIFACT_FORMAT, "environment": env_export,
                "resolvedHyperparameters": meta["hyperparameters"], "performance": performance}
    if command == "evaluate":
        bundle = load_bundle(request["artifact"], env)
        names = common.requested_metrics(spec, MODELS[spec["algorithm"]])
        y = np.array([row[target] for row in rows])
        prepared = request["prepared"]["indices"]
        metrics, performance = [], {}

        def emit(partition, values, **extra):
            metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n], **extra} for n, v in values.items())

        for partition in ("train", "validation", "test"):
            indices = prepared[partition]
            started = time.perf_counter()
            values = measure(names, y[indices], predict_bundle(bundle, [rows[i] for i in indices]))
            if partition == "test":
                performance["predictSeconds"] = time.perf_counter() - started
            emit(partition, values)
        cv = (spec.get("evaluation") or {}).get("cv")
        if cv:
            pool = np.array(prepared["train"])
            per_fold, fit_total = [], 0.0
            for number, held_out in enumerate(request["prepared"]["folds"]):
                fit_rows = [int(i) for i in pool[~np.isin(pool, held_out)]]
                started = time.perf_counter()
                fold_bundle = train_network(spec, rows, fit_rows)  # fresh parameters and preprocessing for every fold
                fit_total += time.perf_counter() - started
                values = measure(names, y[held_out], predict_bundle(fold_bundle, [rows[i] for i in held_out]))
                per_fold.append(values)
                emit("cv_fold", values, fold=number)
            for n in names:
                series = np.array([f[n] for f in per_fold])
                metrics.append({"name": n, "partition": "cv", "value": float(series.mean()), "direction": common.DIRECTIONS[n], "std": float(series.std())})
            performance["cvFitSeconds"] = fit_total
            performance["cvModelsBuilt"] = float(len(per_fold))
        return {"metrics": metrics, "performance": performance}
    raise ValueError("Unknown backend command")


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
