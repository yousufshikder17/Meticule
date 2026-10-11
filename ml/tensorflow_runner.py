"""TensorFlow CPU backend: bounded feed-forward tabular networks (Keras 3) over the Meticule runner protocol.

No database, scheduler or client paths. Commands match the PyTorch runner (prepare, train, train_chunk, export, evaluate,
predict). Preprocessing, limits, targets and fingerprints come from neural_common, so both neural backends behave alike.
Native Windows TensorFlow is CPU-only, so this backend has no CUDA path; a CUDA request is refused, never redirected.

Artifacts and checkpoints are NumPy archives read with allow_pickle=False. Nothing in them is executable.
"""
import base64
import io
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # native TensorFlow logs are noise on stderr; errors still surface
import joblib
import keras
import numpy as np
import scipy
import sklearn
import tensorflow as tf
import threadpoolctl

import ml_runtime_common as common
import neural_common as nc
from neural_common import (LIMITS, MODELS, TASK_LOSS, fingerprint, fit_preprocessing, resolve_hyperparameters,
                           target_vector, transform)

BACKEND = "tensorflow"
ARTIFACT_FORMAT = "tensorflow-variables-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-tensorflow-v1"
SUPPORTED_PROTOCOLS = {PROTOCOL: 1}
SUPPORTED_SCHEMAS = {1}
RUNTIME_KEYS = ("tensorflow", "keras", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": SUPPORTED_PROTOCOLS,
            "schemas": SUPPORTED_SCHEMAS, "runtime_keys": RUNTIME_KEYS}
CHECKPOINT_FORMAT = "tensorflow-checkpoint-v1"
CHECKPOINT_SCHEMA = 1


def configure_runtime():
    """Single-threaded, op-deterministic execution, so the same inputs give the same weights across processes."""
    for setter in (tf.config.threading.set_inter_op_parallelism_threads, tf.config.threading.set_intra_op_parallelism_threads):
        try:
            setter(1)
        except RuntimeError:
            pass  # TensorFlow already started its thread pools in this process; determinism below still applies
    tf.config.experimental.enable_op_determinism()


configure_runtime()


def environment():
    return common.base_environment([__file__, common.__file__, nc.__file__], BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
                                    "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                                    "threadpoolctl": threadpoolctl.__version__})


def check_compatible(recorded, current):
    common.check_compatible(recorded, current, CONTRACT)


RunnerError = common.RunnerError


def resolve_device(spec):
    """CPU only. A CUDA request is refused explicitly; it is never silently run on CPU."""
    training = spec.get("training") or {}
    requested, precision = training.get("device", "cpu"), training.get("precision", "fp32")
    if requested == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported by the TensorFlow backend")
    if requested != "cpu" or precision not in ("fp32",):
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")
    return requested, precision


def device_fields():
    """Recorded for audit. Deliberately not part of compatibility."""
    return {"device_requested": "cpu", "device_actual": "cpu", "cuda_runtime": "none", "device_name": "none", "device_capability": "none"}


# ---- network and training primitives -------------------------------------------------------------------------------

def build_network(architecture, input_dim, output_dim):
    layers = [keras.Input(shape=(input_dim,))]
    for hidden in architecture["hiddenLayers"]:
        layers += [keras.layers.Dense(hidden, activation=architecture["activation"]), keras.layers.Dropout(architecture["dropout"])]
    layers.append(keras.layers.Dense(output_dim))
    return keras.Sequential(layers)


def make_optimizer(name, learning_rate):
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    return keras.optimizers.AdamW(learning_rate=learning_rate)


def make_loss(name):
    if name == "cross_entropy":
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # matches the PyTorch CrossEntropyLoss on logits
    if name == "mse":
        return keras.losses.MeanSquaredError()
    return keras.losses.MeanAbsoluteError()


class Run:
    """Everything one training run carries between chunks. Built fresh or restored from a checkpoint."""

    def __init__(self, task, hp, model, optimizer, loss_fn, rng, preprocessing, target_info, architecture, input_dim, output_dim,
                 x, y, val_x, val_y):
        self.task, self.hp, self.model, self.optimizer, self.loss_fn, self.rng = task, hp, model, optimizer, loss_fn, rng
        self.preprocessing, self.target_info, self.architecture = preprocessing, target_info, architecture
        self.input_dim, self.output_dim = input_dim, output_dim
        self.x, self.y, self.val_x, self.val_y = x, y, val_x, val_y
        self.completed = 0
        self.elapsed = 0.0
        self.best = None  # {"epoch", "value", "state"}; state is a list of arrays matching model.variables
        self.last_train_loss = None
        self.last_validation_loss = None


def build_data(task, spec, rows, prepared, preprocessing, target_info):
    target = spec["dataset"]["schema"]["target"]
    fit_idx, val_idx = prepared["indices"]["train"], prepared["indices"]["validation"]
    x = transform(preprocessing, [rows[i] for i in fit_idx])
    y = target_vector(task, [rows[i][target] for i in fit_idx], target_info)
    if val_idx:
        val_x = transform(preprocessing, [rows[i] for i in val_idx])
        val_y = target_vector(task, [rows[i][target] for i in val_idx], target_info)
    else:
        val_x = val_y = None
    return x, y, val_x, val_y


def fresh_run(spec, rows, prepared):
    """Seeds, fits preprocessing on training rows only, and builds the network, optimizer and shuffling generator."""
    hp = resolve_hyperparameters(spec)
    task = MODELS[spec["algorithm"]]
    target = spec["dataset"]["schema"]["target"]
    fit_idx = prepared["indices"]["train"]
    if not fit_idx:
        raise ValueError("Training partition is empty")
    resolve_device(spec)
    keras.utils.set_random_seed(spec["seed"])
    preprocessing = fit_preprocessing(spec, rows, fit_idx)
    raw = [rows[i][target] for i in fit_idx]
    if task == "classification":
        classes = sorted(set(raw))
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes in the training partition")
        target_info, output_dim = {"classes": classes}, len(classes)
    else:
        values = np.array(raw, dtype=float)
        mean, std = float(values.mean()), float(values.std()) or 1.0
        target_info, output_dim = {"mean": mean, "std": std}, 1
    architecture = {"hiddenLayers": hp["hiddenLayers"], "activation": hp["activation"], "dropout": float(hp["dropout"])}
    input_dim = int(transform(preprocessing, [rows[fit_idx[0]]]).shape[1])
    model = build_network(architecture, input_dim, output_dim)
    optimizer = make_optimizer(hp["optimizer"], float(hp["learningRate"]))
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, preprocessing, target_info)
    return Run(task, hp, model, optimizer, make_loss(hp["loss"]), rng, preprocessing, target_info, architecture, input_dim, output_dim,
               x, y, val_x, val_y)


def train_step(run, batch_x, batch_y):
    with tf.GradientTape() as tape:
        loss = run.loss_fn(batch_y, run.model(batch_x, training=True))
    grads = tape.gradient(loss, run.model.trainable_variables)
    run.optimizer.apply(grads, run.model.trainable_variables)
    return float(loss)


def run_epochs(run, count):
    """Runs `count` epochs in order. Each epoch is shuffled by the run's own generator, so chunking never changes the result."""
    summaries = []
    for _ in range(count):
        started = time.perf_counter()
        order = run.rng.permutation(len(run.x))
        total = 0.0
        for start in range(0, len(order), run.hp["batchSize"]):
            batch = order[start:start + run.hp["batchSize"]]
            total += train_step(run, run.x[batch], run.y[batch]) * len(batch)
        train_loss = total / len(run.x)
        validation_loss = None
        if run.val_x is not None:
            validation_loss = float(run.loss_fn(run.val_y, run.model(run.val_x, training=False)))
        epoch = run.completed + 1
        elapsed = time.perf_counter() - started
        run.completed = epoch
        run.elapsed += elapsed
        run.last_train_loss, run.last_validation_loss = train_loss, validation_loss
        if validation_loss is not None and (run.best is None or validation_loss < run.best["value"]):
            run.best = {"epoch": epoch, "value": validation_loss, "state": [np.array(v.numpy(), copy=True) for v in run.model.variables]}
        summaries.append({"epoch": epoch, "trainLoss": train_loss, "validationLoss": validation_loss, "elapsedSeconds": elapsed})
    return summaries


def train_network(spec, rows, fit_indices, validation_indices=None):
    """One-shot training used by cross-validation folds: a fresh run trained for every requested epoch."""
    prepared = {"indices": {"train": fit_indices, "validation": validation_indices or []}}
    run = fresh_run(spec, rows, prepared)
    run_epochs(run, run.hp["epochs"])
    return bundle_from_network(run.model, run.preprocessing, run.task, run.target_info, run.architecture,
                               run.input_dim, run.output_dim, run.hp, run.last_train_loss, run.last_validation_loss)


def bundle_from_network(network, preprocessing, task, target_info, architecture, input_dim, output_dim, hp, train_loss, validation_loss):
    return {"network": network, "preprocessing": preprocessing, "task": task, "target": target_info,
            "architecture": architecture, "input_dim": input_dim, "output_dim": output_dim, "hyperparameters": hp,
            "final_train_loss": train_loss, "final_validation_loss": validation_loss}


# ---- checkpoints: NumPy archives of model and optimizer variables, validated before any array is used -----------------

def pack(metadata, groups):
    """groups maps a name to a list of arrays. The archive holds the metadata JSON plus one entry per array."""
    arrays = {f"{name}_{i}": array for name, values in groups.items() for i, array in enumerate(values)}
    buffer = io.BytesIO()
    np.savez(buffer, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


def unpack(encoded, refusal):
    """Returns (metadata, groups). Refuses anything whose entries do not match the counts its metadata declares.

    A corrupt or malformed checkpoint is InvalidCheckpoint; a malformed artifact stays IncompatibleArtifact (Phase 3D).
    """
    invalid = common.InvalidCheckpoint if refusal == "checkpoint" else common.IncompatibleArtifact
    try:
        raw = base64.b64decode(encoded, validate=True)
        archive = np.load(io.BytesIO(raw), allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        counts = metadata["counts"]
        expected = {"metadata"} | {f"{name}_{i}" for name, count in counts.items() for i in range(count)}
        if set(archive.files) != expected:
            raise invalid("malformed " + refusal)
        groups = {name: [archive[f"{name}_{i}"] for i in range(count)] for name, count in counts.items()}
    except common.IncompatibleArtifact:
        raise
    except Exception:
        raise invalid("unreadable " + refusal) from None
    return metadata, groups


def assign(variables, values, what, refusal=common.IncompatibleArtifact):
    if len(variables) != len(values):
        raise refusal(f"{what} does not match the architecture")
    for variable, value in zip(variables, values):
        variable.assign(value)


def save_run(run, spec, prepared, env):
    metadata = {"schema": CHECKPOINT_SCHEMA, "format": CHECKPOINT_FORMAT, "backend": BACKEND, "algorithm": spec["algorithm"],
                "task": run.task, "fingerprint": fingerprint(spec, run.hp, prepared), "completedEpoch": run.completed,
                "device": "cpu", "precision": "fp32", "deviceProvenance": device_fields(),
                "architecture": run.architecture, "inputDim": run.input_dim, "outputDim": run.output_dim,
                "target": run.target_info, "preprocessing": run.preprocessing, "hyperparameters": run.hp,
                "elapsedSeconds": run.elapsed, "lastTrainLoss": run.last_train_loss, "lastValidationLoss": run.last_validation_loss,
                "best": None if run.best is None else {"epoch": run.best["epoch"], "value": run.best["value"]},
                "generator": run.rng.bit_generator.state, "environment": env}
    groups = {"model": [v.numpy() for v in run.model.variables], "optimizer": [v.numpy() for v in run.optimizer.variables]}
    if run.best is not None:
        groups["best"] = run.best["state"]
    metadata["counts"] = {name: len(values) for name, values in groups.items()}
    return pack(metadata, groups)


def load_checkpoint(checkpoint, current_env):
    metadata, groups = unpack(checkpoint, "checkpoint")
    if metadata.get("format") != CHECKPOINT_FORMAT or metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise common.IncompatibleCheckpoint("unsupported checkpoint format or schema")
    try:
        common.check_compatible(metadata.get("environment"), current_env, CONTRACT)
    except common.IncompatibleArtifact as error:
        raise common.IncompatibleCheckpoint(str(error)) from None
    return metadata, groups


def restore_run(checkpoint, spec, rows, prepared, current_env):
    meta, groups = load_checkpoint(checkpoint, current_env)
    hp = resolve_hyperparameters(spec)
    if meta["fingerprint"] != fingerprint(spec, hp, prepared):
        raise common.IncompatibleCheckpoint("Checkpoint belongs to a different training configuration")
    task = MODELS[spec["algorithm"]]
    arch = meta["architecture"]
    resolve_device(spec)
    model = build_network(arch, meta["inputDim"], meta["outputDim"])
    assign(model.variables, groups["model"], "checkpoint weights", common.IncompatibleCheckpoint)
    optimizer = make_optimizer(hp["optimizer"], float(hp["learningRate"]))
    optimizer.build(model.trainable_variables)  # creates the slots the saved optimizer state is assigned into
    assign(optimizer.variables, groups["optimizer"], "checkpoint optimizer state", common.IncompatibleCheckpoint)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = meta["generator"]
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, meta["preprocessing"], meta["target"])
    run = Run(task, hp, model, optimizer, make_loss(hp["loss"]), rng, meta["preprocessing"], meta["target"], arch,
              meta["inputDim"], meta["outputDim"], x, y, val_x, val_y)
    run.completed = meta["completedEpoch"]
    run.elapsed = meta["elapsedSeconds"]
    run.last_train_loss, run.last_validation_loss = meta["lastTrainLoss"], meta["lastValidationLoss"]
    if meta["best"] is not None:
        run.best = {"epoch": meta["best"]["epoch"], "value": meta["best"]["value"], "state": groups["best"]}
    return run


# ---- artifacts and inference ---------------------------------------------------------------------------------------

def serialize(bundle, env):
    metadata = {"schema": ARTIFACT_SCHEMA, "backend": BACKEND, "task": bundle["task"], "architecture": bundle["architecture"],
                "inputDim": bundle["input_dim"], "outputDim": bundle["output_dim"], "target": bundle["target"],
                "preprocessing": bundle["preprocessing"], "environment": env}
    groups = {"model": [v.numpy() for v in bundle["network"].variables]}
    metadata["counts"] = {name: len(values) for name, values in groups.items()}
    return pack(metadata, groups)


def load_bundle(artifact, current_env):
    """Validates the artifact contract before any array is used. Allowlisted arrays only; no pickled objects."""
    metadata, groups = unpack(artifact, "artifact")
    check_compatible(metadata.get("environment"), current_env)
    if metadata.get("task") not in TASK_LOSS or metadata.get("backend") != BACKEND:
        raise common.IncompatibleArtifact("unsupported artifact task")
    architecture = metadata["architecture"]
    network = build_network({"hiddenLayers": architecture["hiddenLayers"], "activation": architecture["activation"],
                             "dropout": float(architecture["dropout"])}, metadata["inputDim"], metadata["outputDim"])
    assign(network.variables, groups["model"], "artifact weights")
    return {"network": network, "preprocessing": metadata["preprocessing"], "task": metadata["task"], "target": metadata["target"],
            "architecture": architecture, "input_dim": metadata["inputDim"], "output_dim": metadata["outputDim"]}


def predict_bundle(bundle, rows):
    """Normalised output shared with the sklearn and PyTorch runners: predictions, class probabilities and labels."""
    network = bundle["network"]
    x = transform(bundle["preprocessing"], rows)
    output = np.asarray(network(x, training=False), dtype=float) if len(rows) else np.zeros((0, bundle["output_dim"]))
    if bundle["task"] == "classification":
        shifted = output - output.max(axis=1, keepdims=True) if len(rows) else output
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True) if len(rows) else output
        classes = bundle["target"]["classes"]
        labels = [classes[i] for i in probabilities.argmax(axis=1)] if len(rows) else []
        return {"prediction": np.array(labels), "proba": probabilities, "classes": np.array(classes)}
    values = output.reshape(-1) * bundle["target"]["std"] + bundle["target"]["mean"] if len(rows) else np.zeros(0)
    return {"prediction": values, "proba": None, "classes": None}


def requested_metrics(spec):
    return common.requested_metrics(spec, MODELS[spec["algorithm"]])


def execute(request):
    env = environment()
    spec = request["snapshot"]
    rows = request["rows"]
    if request.get("expectedEnvironment") is not None:
        # Recorded in PostgreSQL and checked before any artifact bytes are read.
        check_compatible(request["expectedEnvironment"], env)
    command = request["command"]
    target = spec["dataset"]["schema"]["target"]
    if command == "prepare":
        y = np.array([row[target] for row in rows])
        return {**common.partitions(spec, y), "environment": env}
    if command == "predict":
        bundle = load_bundle(request["artifact"], env)
        return {"predictions": predict_bundle(bundle, rows)["prediction"].tolist()}
    if command == "train":
        prepared = request["prepared"]["indices"]
        started = time.perf_counter()
        run = fresh_run(spec, rows, {"indices": prepared})
        run_epochs(run, run.hp["epochs"])
        bundle = bundle_from_network(run.model, run.preprocessing, run.task, run.target_info, run.architecture,
                                     run.input_dim, run.output_dim, run.hp, run.last_train_loss, run.last_validation_loss)
        fit_seconds = time.perf_counter() - started
        artifact = serialize(bundle, {**env, **device_fields(), "precision": "fp32"})
        performance = {"fitSeconds": fit_seconds, "epochsCompleted": float(run.completed), "finalTrainLoss": float(run.last_train_loss)}
        if run.last_validation_loss is not None:
            performance["finalValidationLoss"] = run.last_validation_loss
        return {"artifact": artifact, "format": ARTIFACT_FORMAT, "environment": {**env, **device_fields(), "precision": "fp32"},
                "resolvedHyperparameters": run.hp, "performance": performance}
    if command == "train_chunk":
        # One bounded chunk of epochs. Meticule decides when to call again and what to persist.
        prepared = {"indices": request["prepared"]["indices"]}
        epochs = int(request["epochs"])
        if not 1 <= epochs <= LIMITS["maxEpochs"]:
            raise ValueError("Chunk size out of range")
        if request.get("state") is None:
            run = fresh_run(spec, rows, prepared)
        else:
            run = restore_run(request["state"], spec, rows, prepared, env)
        summaries = run_epochs(run, epochs)
        env_run = {**env, **device_fields(), "precision": "fp32"}
        return {"checkpoint": save_run(run, spec, prepared, env_run), "checkpointFormat": CHECKPOINT_FORMAT,
                "checkpointSchema": CHECKPOINT_SCHEMA, "completedEpoch": run.completed, "epochs": summaries, "environment": env_run}
    if command == "export":
        # Builds the normal inference artifact from a checkpoint, from either the last or the best-epoch weights.
        meta, groups = load_checkpoint(request["checkpoint"], env)
        if meta["algorithm"] != spec["algorithm"] or meta["target"] is None:
            raise common.IncompatibleCheckpoint("Checkpoint does not match the snapshot")
        weights = groups["best"] if request.get("restoreBest") and "best" in groups else groups["model"]
        network = build_network(meta["architecture"], meta["inputDim"], meta["outputDim"])
        assign(network.variables, weights, "checkpoint weights", common.IncompatibleCheckpoint)
        bundle = bundle_from_network(network, meta["preprocessing"], meta["task"], meta["target"], meta["architecture"],
                                     meta["inputDim"], meta["outputDim"], meta["hyperparameters"], meta["lastTrainLoss"], meta["lastValidationLoss"])
        performance = {"fitSeconds": float(meta["elapsedSeconds"]), "epochsCompleted": float(meta["completedEpoch"]),
                       "finalTrainLoss": float(meta["lastTrainLoss"])}
        if meta["lastValidationLoss"] is not None:
            performance["finalValidationLoss"] = meta["lastValidationLoss"]
        if meta["best"] is not None:
            performance["bestEpoch"] = float(meta["best"]["epoch"])
        env_export = {**env, **meta.get("deviceProvenance", device_fields()), "precision": "fp32"}
        return {"artifact": serialize(bundle, env_export), "format": ARTIFACT_FORMAT, "environment": env_export,
                "resolvedHyperparameters": meta["hyperparameters"], "performance": performance}
    if command == "evaluate":
        bundle = load_bundle(request["artifact"], env)
        names = requested_metrics(spec)
        y = np.array([row[target] for row in rows])
        prepared = request["prepared"]["indices"]
        metrics, performance = [], {}

        def emit(partition, values, **extra):
            metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n], **extra} for n, v in values.items())

        for partition in ("train", "validation", "test"):
            indices = prepared[partition]
            started = time.perf_counter()
            output = predict_bundle(bundle, [rows[i] for i in indices])
            values = common.measure_predictions(names, y[indices], output["prediction"], proba=output["proba"], scores=output["proba"], classes=output["classes"]) \
                if output["proba"] is not None else common.measure_predictions(names, y[indices], output["prediction"])
            if partition == "test":
                performance["predictSeconds"] = time.perf_counter() - started
            emit(partition, values)
        cv = (spec.get("evaluation") or {}).get("cv")
        if cv:
            pool = np.array(prepared["train"])
            per_fold, fit_total, built = [], 0.0, 0
            for number, held_out in enumerate(request["prepared"]["folds"]):
                fit_rows = [int(i) for i in pool[~np.isin(pool, held_out)]]
                started = time.perf_counter()
                fold_bundle = train_network(spec, rows, fit_rows)  # fresh network and preprocessing for every fold
                built += 1
                fit_total += time.perf_counter() - started
                fold_output = predict_bundle(fold_bundle, [rows[i] for i in held_out])
                values = common.measure_predictions(names, y[held_out], fold_output["prediction"], proba=fold_output["proba"], scores=fold_output["proba"], classes=fold_output["classes"]) \
                    if fold_output["proba"] is not None else common.measure_predictions(names, y[held_out], fold_output["prediction"])
                per_fold.append(values)
                emit("cv_fold", values, fold=number)
            for n in names:
                series = np.array([f[n] for f in per_fold])
                metrics.append({"name": n, "partition": "cv", "value": float(series.mean()), "direction": common.DIRECTIONS[n], "std": float(series.std())})
            performance["cvFitSeconds"] = fit_total
            performance["cvModelsBuilt"] = float(built)
        return {"metrics": metrics, "performance": performance}
    raise ValueError("Unknown backend command")


if __name__ == "__main__":
    sys.exit(common.serve(execute))
