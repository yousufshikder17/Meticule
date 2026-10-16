"""PyTorch CPU backend: bounded feed-forward tabular networks over the Meticule runner protocol.

No database, scheduler or client paths. Commands match sklearn_runner.py (prepare, train, evaluate, predict).
Preprocessing is implemented here in NumPy and its fitted state is stored as JSON, so PyTorch artifacts contain
no pickled sklearn objects. Splits and metric computation are shared with the sklearn runner (ml_runtime_common).
"""
import base64
import io
import os
from contextlib import nullcontext
import json
import random
import sys
import time

# cuBLAS needs a fixed workspace for deterministic kernels; it must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ml_runtime_common as common

BACKEND = "pytorch"
ARTIFACT_FORMAT = "pytorch-state-dict-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-pytorch-v1"
SUPPORTED_PROTOCOLS = {PROTOCOL: 1}
SUPPORTED_SCHEMAS = {1}
RUNTIME_KEYS = ("torch", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": SUPPORTED_PROTOCOLS,
            "schemas": SUPPORTED_SCHEMAS, "runtime_keys": RUNTIME_KEYS}

import neural_common
from neural_common import (DEFAULTS, LIMITS, MODELS, TASK_LOSS, fingerprint, fit_preprocessing,
                           resolve_hyperparameters, target_vector, transform)

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
OPTIMIZERS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss, "mse": nn.MSELoss, "l1": nn.L1Loss}


def environment():
    return common.base_environment([__file__, common.__file__, neural_common.__file__], BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                    "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                                    "threadpoolctl": threadpoolctl.__version__})


def check_compatible(recorded, current):
    common.check_compatible(recorded, current, CONTRACT)


RunnerError = common.RunnerError


def resolve_device(spec):
    """Explicit device and precision. A requested CUDA device that is unavailable fails; it is never replaced by CPU."""
    training = spec.get("training") or {}
    requested, precision = training.get("device", "cpu"), training.get("precision", "fp32")
    if requested not in ("cpu", "cuda") or precision not in ("fp32", "mixed"):
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")
    if precision == "mixed" and requested != "cuda":
        raise RunnerError("mixed_requires_cuda", "Mixed precision requires device cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RunnerError("cuda_unavailable", "CUDA requested but unavailable")
    return requested, precision


def device_provenance(requested, actual, name=None, capability=None, cuda_runtime=None):
    """Where a run actually executed. Recorded for audit; deliberately not part of compatibility."""
    cuda = actual == "cuda"
    if not cuda:
        return {"device_requested": requested, "device_actual": actual, "cuda_runtime": "none", "device_name": "none", "device_capability": "none"}
    return {"device_requested": requested, "device_actual": actual,
            "cuda_runtime": cuda_runtime or (torch.version.cuda or "none"),
            "device_name": name or torch.cuda.get_device_name(0),
            "device_capability": capability or ".".join(map(str, torch.cuda.get_device_capability(0)))}


def cpu_tree(value):
    """Checkpoints hold CPU tensors, so they load on any device."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def configure_determinism():
    """Process-wide execution settings every chunk needs. Each chunk runs in a new process, so fresh and restored runs
    must both apply them: a resumed chunk may not run under weaker determinism than the chunk before it."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def seed_all(seed):
    """Seed every source of randomness the runner uses and forbid nondeterministic CPU kernels where PyTorch can."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_determinism()


# ---- network ----------------------------------------------------------------------------------------------------

def build_network(architecture, input_dim, output_dim):
    layers, width = [], input_dim
    for hidden in architecture["hiddenLayers"]:
        layers += [nn.Linear(width, hidden), ACTIVATIONS[architecture["activation"]](), nn.Dropout(architecture["dropout"])]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


CHECKPOINT_FORMAT = "pytorch-checkpoint-v1"
CHECKPOINT_SCHEMA = 1




class Run:
    """Everything one training run carries between chunks. Built fresh or restored from a checkpoint."""

    def __init__(self, task, hp, network, optimizer, generator, preprocessing, target_info, architecture, input_dim, output_dim,
                 criterion, x, y, val_x, val_y):
        self.task, self.hp, self.network, self.optimizer, self.generator = task, hp, network, optimizer, generator
        self.preprocessing, self.target_info, self.architecture = preprocessing, target_info, architecture
        self.input_dim, self.output_dim, self.criterion = input_dim, output_dim, criterion
        self.x, self.y, self.val_x, self.val_y = x, y, val_x, val_y
        self.completed = 0
        self.elapsed = 0.0
        self.best = None  # {"epoch", "value", "state"}
        self.last_train_loss = None
        self.last_validation_loss = None


def build_data(task, spec, rows, prepared, preprocessing, target_info):
    target = spec["dataset"]["schema"]["target"]
    fit_idx, val_idx = prepared["indices"]["train"], prepared["indices"]["validation"]
    x = torch.from_numpy(transform(preprocessing, [rows[i] for i in fit_idx]))
    y = torch.from_numpy(target_vector(task, [rows[i][target] for i in fit_idx], target_info))
    if val_idx:
        val_x = torch.from_numpy(transform(preprocessing, [rows[i] for i in val_idx]))
        val_y = torch.from_numpy(target_vector(task, [rows[i][target] for i in val_idx], target_info))
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
    seed_all(spec["seed"])
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
    requested, precision = resolve_device(spec)
    network = build_network(architecture, input_dim, output_dim).to(torch.device(requested))
    optimizer = OPTIMIZERS[hp["optimizer"]](network.parameters(), lr=float(hp["learningRate"]))
    generator = torch.Generator().manual_seed(spec["seed"])
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, preprocessing, target_info)
    run = Run(task, hp, network, optimizer, generator, preprocessing, target_info, architecture, input_dim, output_dim,
              LOSSES[hp["loss"]](), x, y, val_x, val_y)
    attach_device(run, requested, precision)
    return run


def attach_device(run, requested, precision, scaler_state=None):
    """Binds a run to its device. Mixed precision gets a gradient scaler, whose state a checkpoint carries."""
    run.requested, run.precision, run.device = requested, precision, torch.device(requested)
    run.scaler = torch.amp.GradScaler("cuda") if precision == "mixed" else None
    if run.scaler is not None and scaler_state is not None:
        run.scaler.load_state_dict(scaler_state)


def save_run(run, spec, prepared, env):
    metadata = {"schema": CHECKPOINT_SCHEMA, "format": CHECKPOINT_FORMAT, "backend": BACKEND, "algorithm": spec["algorithm"],
                "task": run.task, "fingerprint": fingerprint(spec, run.hp, prepared), "completedEpoch": run.completed,
                "device": run.requested, "precision": run.precision,
                "deviceProvenance": device_provenance(run.requested, run.device.type, name=env.get("device_name"),
                                                      capability=env.get("device_capability"), cuda_runtime=env.get("cuda_runtime")),
                "architecture": run.architecture, "inputDim": run.input_dim, "outputDim": run.output_dim,
                "target": run.target_info, "preprocessing": run.preprocessing, "hyperparameters": run.hp,
                "elapsedSeconds": run.elapsed, "lastTrainLoss": run.last_train_loss, "lastValidationLoss": run.last_validation_loss,
                "best": None if run.best is None else {"epoch": run.best["epoch"], "value": run.best["value"]}, "environment": env}
    payload = {"metadata": json.dumps(metadata, allow_nan=False), "model": cpu_tree(run.network.state_dict()),
               "optimizer": cpu_tree(run.optimizer.state_dict()), "generator": run.generator.get_state(), "rng": torch.get_rng_state(),
               "best_model": None if run.best is None else run.best["state"],
               "scaler": None if run.scaler is None else run.scaler.state_dict(),
               "cuda_rng": torch.cuda.get_rng_state().cpu() if run.device.type == "cuda" else None}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return base64.b64encode(buffer.getvalue()).decode()


def load_checkpoint(checkpoint, current_env):
    """Validates a checkpoint before it is turned into a run. weights_only refuses arbitrary objects."""
    try:
        raw = base64.b64decode(checkpoint, validate=True)
        payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    except Exception:
        raise common.InvalidCheckpoint("unreadable checkpoint") from None
    expected = {"metadata", "model", "optimizer", "generator", "rng", "best_model", "scaler", "cuda_rng"}
    if not isinstance(payload, dict) or set(payload) != expected or not isinstance(payload["metadata"], str):
        raise common.InvalidCheckpoint("malformed checkpoint")
    metadata = json.loads(payload["metadata"])
    if metadata.get("format") != CHECKPOINT_FORMAT or metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise common.IncompatibleCheckpoint("unsupported checkpoint format or schema")
    try:
        common.check_compatible(metadata.get("environment"), current_env, CONTRACT)
    except common.IncompatibleArtifact as error:
        raise common.IncompatibleCheckpoint(str(error)) from None
    return payload, metadata


def restore_run(checkpoint, spec, rows, prepared, current_env):
    configure_determinism()
    payload, meta = load_checkpoint(checkpoint, current_env)
    hp = resolve_hyperparameters(spec)
    if meta["fingerprint"] != fingerprint(spec, hp, prepared):
        raise common.IncompatibleCheckpoint("Checkpoint belongs to a different training configuration")
    task = MODELS[spec["algorithm"]]
    arch = meta["architecture"]
    requested, precision = resolve_device(spec)
    network = build_network(arch, meta["inputDim"], meta["outputDim"])
    network.load_state_dict(payload["model"], strict=True)
    network.to(torch.device(requested))
    optimizer = OPTIMIZERS[hp["optimizer"]](network.parameters(), lr=float(hp["learningRate"]))
    optimizer.load_state_dict(payload["optimizer"])  # casts optimizer state to the parameters' device
    generator = torch.Generator()
    generator.set_state(payload["generator"])
    torch.set_rng_state(payload["rng"])
    if requested == "cuda" and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu())
    x, y, val_x, val_y = build_data(task, spec, rows, prepared, meta["preprocessing"], meta["target"])
    run = Run(task, hp, network, optimizer, generator, meta["preprocessing"], meta["target"], arch, meta["inputDim"], meta["outputDim"],
              LOSSES[hp["loss"]](), x, y, val_x, val_y)
    run.completed = meta["completedEpoch"]
    run.elapsed = meta["elapsedSeconds"]
    run.last_train_loss, run.last_validation_loss = meta["lastTrainLoss"], meta["lastValidationLoss"]
    if meta["best"] is not None:
        run.best = {"epoch": meta["best"]["epoch"], "value": meta["best"]["value"], "state": payload["best_model"]}
    attach_device(run, requested, precision, payload.get("scaler"))
    return run


def run_epochs(run, count):
    """Runs `count` epochs in order. Each epoch is shuffled by the run's own generator, so chunking never changes the result."""
    summaries = []
    for _ in range(count):
        started = time.perf_counter()
        loader = DataLoader(TensorDataset(run.x, run.y), batch_size=run.hp["batchSize"], shuffle=True, generator=run.generator)
        run.network.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(run.device), batch_y.to(run.device)
            run.optimizer.zero_grad()
            with (torch.autocast(device_type="cuda", dtype=torch.float16) if run.precision == "mixed" else nullcontext()):
                loss = run.criterion(run.network(batch_x), batch_y)
            if run.scaler is not None:
                run.scaler.scale(loss).backward()
                run.scaler.step(run.optimizer)
                run.scaler.update()
            else:
                loss.backward()
                run.optimizer.step()
            total += loss.item() * len(batch_x)
        train_loss = total / len(run.x)
        validation_loss = None
        if run.val_x is not None:
            run.network.eval()
            with torch.no_grad():
                validation_loss = float(run.criterion(run.network(run.val_x.to(run.device)), run.val_y.to(run.device)))
        epoch = run.completed + 1
        elapsed = time.perf_counter() - started
        run.completed = epoch
        run.elapsed += elapsed
        run.last_train_loss, run.last_validation_loss = train_loss, validation_loss
        if validation_loss is not None and (run.best is None or validation_loss < run.best["value"]):
            run.best = {"epoch": epoch, "value": validation_loss,
                        "state": {k: v.detach().cpu().clone() for k, v in run.network.state_dict().items()}}
        summaries.append({"epoch": epoch, "trainLoss": train_loss, "validationLoss": validation_loss, "elapsedSeconds": elapsed})
    return summaries


def train_network(spec, rows, fit_indices, validation_indices=None):
    """One-shot training used by cross-validation folds: a fresh run trained for every requested epoch."""
    prepared = {"indices": {"train": fit_indices, "validation": validation_indices or []}}
    run = fresh_run(spec, rows, prepared)
    run_epochs(run, run.hp["epochs"])
    return bundle_from_network(run.network, run.preprocessing, run.task, run.target_info, run.architecture,
                               run.input_dim, run.output_dim, run.hp, run.last_train_loss, run.last_validation_loss)


def bundle_from_network(network, preprocessing, task, target_info, architecture, input_dim, output_dim, hp, train_loss, validation_loss):
    return {"network": network, "preprocessing": preprocessing, "task": task, "target": target_info,
            "architecture": architecture, "input_dim": input_dim, "output_dim": output_dim, "hyperparameters": hp,
            "final_train_loss": train_loss, "final_validation_loss": validation_loss}


def predict_bundle(bundle, rows):
    """Normalised output shared with the sklearn runner: predictions, class probabilities and training labels."""
    network = bundle["network"]
    network.eval()
    x = torch.from_numpy(transform(bundle["preprocessing"], rows))
    with torch.no_grad():
        output = network(x).numpy().astype(float) if len(rows) else np.zeros((0, bundle["output_dim"]))
    if bundle["task"] == "classification":
        probabilities = torch.softmax(torch.from_numpy(output), dim=1).numpy().astype(float) if len(rows) else output
        classes = bundle["target"]["classes"]
        labels = [classes[i] for i in probabilities.argmax(axis=1)] if len(rows) else []
        return {"prediction": np.array(labels), "proba": probabilities, "classes": np.array(classes)}
    values = output.reshape(-1) * bundle["target"]["std"] + bundle["target"]["mean"] if len(rows) else np.zeros(0)
    return {"prediction": values, "proba": None, "classes": None}


# ---- artifacts: a state dict, validated metadata, and loading that refuses arbitrary objects -----------------------

def serialize(bundle, env):
    metadata = {"schema": ARTIFACT_SCHEMA, "backend": BACKEND, "task": bundle["task"], "architecture": bundle["architecture"],
                "inputDim": bundle["input_dim"], "outputDim": bundle["output_dim"], "target": bundle["target"],
                "preprocessing": bundle["preprocessing"], "environment": env}
    buffer = io.BytesIO()
    torch.save({"metadata": json.dumps(metadata, allow_nan=False), "state_dict": bundle["network"].state_dict()}, buffer)
    return base64.b64encode(buffer.getvalue()).decode()


def load_bundle(artifact, current_env):
    """Validate the artifact contract before any tensor is read. weights_only=True refuses arbitrary Python objects."""
    try:
        raw = base64.b64decode(artifact, validate=True)
        payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    except Exception:
        raise common.IncompatibleArtifact("unreadable artifact") from None
    if not isinstance(payload, dict) or set(payload) != {"metadata", "state_dict"} or not isinstance(payload["metadata"], str):
        raise common.IncompatibleArtifact("malformed artifact")
    metadata = json.loads(payload["metadata"])
    check_compatible(metadata.get("environment"), current_env)
    if metadata.get("task") not in TASK_LOSS or metadata.get("backend") != BACKEND:
        raise common.IncompatibleArtifact("unsupported artifact task")
    architecture = metadata["architecture"]
    hp = resolve_hyperparameters({"algorithm": "mlp_classifier" if metadata["task"] == "classification" else "mlp_regressor",
                                  "hyperparameters": {**architecture, "epochs": 1}})
    network = build_network({"hiddenLayers": hp["hiddenLayers"], "activation": hp["activation"], "dropout": float(hp["dropout"])},
                            metadata["inputDim"], metadata["outputDim"])
    network.load_state_dict(payload["state_dict"], strict=True)
    network.eval()
    return {"network": network, "preprocessing": metadata["preprocessing"], "task": metadata["task"], "target": metadata["target"],
            "architecture": architecture, "input_dim": metadata["inputDim"], "output_dim": metadata["outputDim"]}


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
        return {**common.split_partitions(spec, rows, y), "environment": env}
    if command == "predict":
        bundle = load_bundle(request["artifact"], env)
        return {"predictions": predict_bundle(bundle, rows)["prediction"].tolist()}
    if command == "train":
        prepared = request["prepared"]["indices"]
        started = time.perf_counter()
        run = fresh_run(spec, rows, {"indices": prepared})
        run_epochs(run, run.hp["epochs"])
        bundle = bundle_from_network(run.network, run.preprocessing, run.task, run.target_info, run.architecture,
                                     run.input_dim, run.output_dim, run.hp, run.last_train_loss, run.last_validation_loss)
        fit_seconds = time.perf_counter() - started
        artifact = serialize(bundle, env)
        performance = {"fitSeconds": fit_seconds, "epochsCompleted": float(run.completed), "finalTrainLoss": float(run.last_train_loss)}
        if run.last_validation_loss is not None:
            performance["finalValidationLoss"] = run.last_validation_loss
        return {"artifact": artifact, "format": ARTIFACT_FORMAT, "environment": env,
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
        env_run = {**env, **device_provenance(run.requested, run.device.type), "precision": run.precision}
        return {"checkpoint": save_run(run, spec, prepared, env_run), "checkpointFormat": CHECKPOINT_FORMAT,
                "checkpointSchema": CHECKPOINT_SCHEMA, "completedEpoch": run.completed, "epochs": summaries, "environment": env_run}
    if command == "export":
        # Builds the normal inference artifact from a checkpoint, from either the last or the best-epoch weights.
        payload, meta = load_checkpoint(request["checkpoint"], env)
        if meta["algorithm"] != spec["algorithm"] or meta["target"] is None:
            raise common.IncompatibleCheckpoint("Checkpoint does not match the snapshot")
        weights = payload["best_model"] if request.get("restoreBest") and payload["best_model"] is not None else payload["model"]
        network = build_network(meta["architecture"], meta["inputDim"], meta["outputDim"])
        network.load_state_dict(weights, strict=True)
        bundle = bundle_from_network(network, meta["preprocessing"], meta["task"], meta["target"], meta["architecture"],
                                     meta["inputDim"], meta["outputDim"], meta["hyperparameters"], meta["lastTrainLoss"], meta["lastValidationLoss"])
        performance = {"fitSeconds": float(meta["elapsedSeconds"]), "epochsCompleted": float(meta["completedEpoch"]),
                       "finalTrainLoss": float(meta["lastTrainLoss"])}
        if meta["lastValidationLoss"] is not None:
            performance["finalValidationLoss"] = meta["lastValidationLoss"]
        if meta["best"] is not None:
            performance["bestEpoch"] = float(meta["best"]["epoch"])
        provenance = meta.get("deviceProvenance", {})
        env_export = {**env, **provenance, "precision": meta.get("precision", "fp32")}
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
                values = common.measure_predictions(names, y[held_out], fold_output["prediction"], proba=fold_output["proba"],
                                                    scores=fold_output["proba"], classes=fold_output["classes"]) \
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
    sys.exit(common.serve(execute, task_of=MODELS.get))
