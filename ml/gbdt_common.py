"""What the XGBoost and LightGBM runners genuinely share: bounded hyperparameters, targets, the JSON artifact envelope,
early stopping on the validation partition, evaluation with cross-validation, and the command dispatch.

Each runner supplies only its framework: versions, parameter mapping, native fit, raw prediction and model text dump/load.
Preprocessing is neural_common's JSON-state NumPy implementation, fitted on training rows only; missing numbers pass
through as NaN, which both frameworks handle natively. No pickle anywhere: the artifact is JSON around the framework's own
text model format.
"""
import base64
import binascii
import json
import math
import time

import numpy as np

import ml_runtime_common as common
from neural_common import fit_preprocessing, transform

RunnerError = common.RunnerError
ARTIFACT_SCHEMA = 1
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
# Shared bounds (mirrored in src/ml/gbdt-models.ts and enforced again here, the last line of defence).
COMMON_PARAMETERS = {
    "n_estimators": ("int", 1, 2000, 100),
    "learning_rate": ("float_open", 0.0, 1.0, 0.1),
    "subsample": ("float_open", 0.0, 1.0, 1.0),
    "colsample_bytree": ("float_open", 0.0, 1.0, 1.0),
    "reg_alpha": ("float", 0.0, 1000.0, 0.0),
    "early_stopping_rounds": ("int", 0, 200, 0),
}


class Runner:
    """A framework adapter: the runner module's own constants and five functions."""

    def __init__(self, backend, artifact_format, protocol, runtime_keys, parameters, environment, fit, raw_predict, dump, load):
        self.backend, self.artifact_format, self.protocol = backend, artifact_format, protocol
        self.parameters = {**COMMON_PARAMETERS, **parameters}
        self.environment, self.fit, self.raw_predict, self.dump, self.load = environment, fit, raw_predict, dump, load
        self.contract = {"backend": backend, "artifact_format": artifact_format, "protocols": {protocol: ARTIFACT_SCHEMA},
                         "schemas": {ARTIFACT_SCHEMA}, "runtime_keys": runtime_keys}


MODELS = {"xgboost": {"xgboost_classifier": "classification", "xgboost_regressor": "regression"},
          "lightgbm": {"lightgbm_classifier": "classification", "lightgbm_regressor": "regression"}}


def resolve_hyperparameters(runner, spec):
    """Allowlisted, bounded configuration with the TypeScript catalog's defaults. Unknown names are refused, never forwarded."""
    supplied = spec.get("hyperparameters") or {}
    unknown = set(supplied) - set(runner.parameters)
    if unknown:
        raise ValueError("Unsupported gradient boosting hyperparameter")
    resolved = {}
    for name, (kind, low, high, default) in runner.parameters.items():
        value = supplied.get(name, default)
        if kind == "int_nullable" and value is None:
            resolved[name] = None
            continue
        if kind in ("int", "int_nullable"):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"Hyperparameter {name} out of range")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Hyperparameter {name} out of range")
            if not (low < value <= high if kind == "float_open" else low <= value <= high):
                raise ValueError(f"Hyperparameter {name} out of range")
            value = float(value)
        resolved[name] = value
    return resolved


def resolve_device(spec):
    """CPU only. A CUDA request is refused, never run on CPU instead."""
    training = spec.get("training") or {}
    if training.get("device", "cpu") == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported by this backend")
    if training.get("device", "cpu") != "cpu" or training.get("precision", "fp32") != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")


def target_info(task, raw):
    if task == "classification":
        classes = sorted(set(raw))
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes in the training partition")
        return {"classes": classes}
    return {}


def encode_targets(task, raw, info):
    if task == "regression":
        return np.array(raw, dtype=float)
    index = {c: i for i, c in enumerate(info["classes"])}
    if any(v not in index for v in raw):
        raise ValueError("Label not seen in the training partition")
    return np.array([index[v] for v in raw], dtype=np.int32)


def matrix(preprocessing, rows):
    """NaN passes through for missing numbers (an imputer, when declared, still fills them)."""
    return transform(preprocessing, rows, missing="nan").astype(np.float64)


def fit_model(runner, spec, rows, fit_idx, validation_idx, early_stopping=True):
    """Fits preprocessing and the booster on training rows only. With early stopping, the validation partition is the
    monitoring set and nothing else: it never fits preprocessing or trees."""
    task = MODELS[runner.backend][spec["algorithm"]]
    target = spec["dataset"]["schema"]["target"]
    hp = resolve_hyperparameters(runner, spec)
    if not fit_idx:
        raise ValueError("Training partition is empty")
    preprocessing = fit_preprocessing(spec, rows, fit_idx)
    raw = [rows[i][target] for i in fit_idx]
    info = target_info(task, raw)
    x, y = matrix(preprocessing, [rows[i] for i in fit_idx]), encode_targets(task, raw, info)
    monitor = None
    if early_stopping and hp["early_stopping_rounds"] > 0:
        if not validation_idx:
            raise ValueError("Early stopping needs a validation partition")
        monitor = (matrix(preprocessing, [rows[i] for i in validation_idx]), encode_targets(task, [rows[i][target] for i in validation_idx], info))
    classes = len(info.get("classes", [])) or None
    model, best_count = runner.fit(task, hp, x, y, classes, monitor, spec["seed"])
    return {"model": model, "best": int(best_count), "task": task, "preprocessing": preprocessing, "target": info,
            "features": int(x.shape[1]), "hyperparameters": hp}


def predict_bundle(runner, bundle, rows):
    """Normalised output shared with the other runners: predictions, class probabilities and training labels."""
    task, classes = bundle["task"], bundle["target"].get("classes")
    if not rows:
        empty = np.zeros((0, len(classes))) if task == "classification" else None
        return {"prediction": np.array([]) if task == "classification" else np.zeros(0), "proba": empty,
                "classes": np.array(classes) if classes else None}
    x = matrix(bundle["preprocessing"], rows)
    if x.shape[1] != bundle["features"]:
        raise common.IncompatibleArtifact("feature count differs from the trained model")
    raw = np.asarray(runner.raw_predict(bundle["model"], x, bundle["best"], task, len(classes) if classes else None), dtype=float)
    if task == "regression":
        return {"prediction": raw.reshape(-1), "proba": None, "classes": None}
    proba = np.column_stack([1.0 - raw, raw]) if raw.ndim == 1 else raw
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {"prediction": np.array([classes[i] for i in proba.argmax(axis=1)]), "proba": proba, "classes": np.array(classes)}


def serialize(runner, bundle, env):
    metadata = {"schema": ARTIFACT_SCHEMA, "backend": runner.backend, "task": bundle["task"], "preprocessing": bundle["preprocessing"],
                "target": bundle["target"], "bestIteration": bundle["best"], "featureCount": bundle["features"], "environment": env}
    envelope = json.dumps({"metadata": metadata, "model": runner.dump(bundle["model"])}, allow_nan=False)
    return base64.b64encode(envelope.encode("utf-8")).decode()


def load_bundle(runner, artifact, current_env):
    """Parses the JSON envelope and checks the environment and metadata before the framework parses any model text."""
    try:
        raw = base64.b64decode(artifact, validate=True)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError
        envelope = json.loads(raw.decode("utf-8"))
        if not isinstance(envelope, dict) or set(envelope) != {"metadata", "model"} or not isinstance(envelope["metadata"], dict) or not isinstance(envelope["model"], str):
            raise ValueError
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise common.IncompatibleArtifact("unreadable artifact") from None
    metadata = envelope["metadata"]
    common.check_compatible(metadata.get("environment"), current_env, runner.contract)
    best, features = metadata.get("bestIteration"), metadata.get("featureCount")
    if (metadata.get("backend") != runner.backend or metadata.get("schema") != ARTIFACT_SCHEMA or metadata.get("task") not in ("classification", "regression")
            or not isinstance(best, int) or isinstance(best, bool) or not 1 <= best <= 2000 or not isinstance(features, int) or features < 0
            or not isinstance(metadata.get("target"), dict) or not isinstance(metadata.get("preprocessing"), dict)):
        raise common.IncompatibleArtifact("malformed artifact metadata")
    try:
        model = runner.load(envelope["model"])
    except Exception:
        raise common.IncompatibleArtifact("unreadable model") from None
    return {"model": model, "best": best, "task": metadata["task"], "preprocessing": metadata["preprocessing"], "target": metadata["target"], "features": features}


def measure(names, y, output):
    if output["proba"] is not None:
        return common.measure_predictions(names, y, output["prediction"], proba=output["proba"], scores=output["proba"], classes=output["classes"])
    return common.measure_predictions(names, y, output["prediction"])


def make_execute(runner):
    def execute(request):
        env = runner.environment()
        spec = request["snapshot"]
        rows = request["rows"]
        if request.get("expectedEnvironment") is not None:
            common.check_compatible(request["expectedEnvironment"], env, runner.contract)
        command = request["command"]
        task = MODELS[runner.backend][spec["algorithm"]]
        target = spec["dataset"]["schema"]["target"]
        resolve_device(spec)
        cv = (spec.get("evaluation") or {}).get("cv")
        if cv and resolve_hyperparameters(runner, spec)["early_stopping_rounds"] > 0:
            # A fold has no separate monitoring partition, so early stopping inside cross-validation is refused, not disabled.
            raise ValueError("Early stopping is not supported with cross-validation")
        if command == "prepare":
            return {**common.split_partitions(spec, rows, np.array([row[target] for row in rows])), "environment": env}
        if command == "predict":
            return {"predictions": predict_bundle(runner, load_bundle(runner, request["artifact"], env), rows)["prediction"].tolist()}
        if command == "train":
            indices = request["prepared"]["indices"]
            started = time.perf_counter()
            bundle = fit_model(runner, spec, rows, indices["train"], indices["validation"])
            performance = {"fitSeconds": time.perf_counter() - started, "bestIteration": float(bundle["best"])}
            return {"artifact": serialize(runner, bundle, env), "format": runner.artifact_format, "environment": env,
                    "resolvedHyperparameters": {**bundle["hyperparameters"], "best_iteration": bundle["best"]}, "performance": performance}
        if command == "evaluate":
            bundle = load_bundle(runner, request["artifact"], env)
            names = common.requested_metrics(spec, task)
            y = np.array([row[target] for row in rows])
            prepared = request["prepared"]["indices"]
            metrics, performance = [], {}

            def emit(partition, values, **extra):
                metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n], **extra} for n, v in values.items())

            for partition in ("train", "validation", "test"):
                indices = prepared[partition]
                started = time.perf_counter()
                values = measure(names, y[indices], predict_bundle(runner, bundle, [rows[i] for i in indices]))
                if partition == "test":
                    performance["predictSeconds"] = time.perf_counter() - started
                emit(partition, values)
            if cv:
                pool = np.array(prepared["train"])
                per_fold, fit_total = [], 0.0
                for number, held_out in enumerate(request["prepared"]["folds"]):
                    started = time.perf_counter()
                    fold = fit_model(runner, spec, rows, [int(i) for i in pool[~np.isin(pool, held_out)]], [], early_stopping=False)
                    fit_total += time.perf_counter() - started
                    values = measure(names, y[held_out], predict_bundle(runner, fold, [rows[i] for i in held_out]))
                    per_fold.append(values)
                    emit("cv_fold", values, fold=number)
                for n in names:
                    series = np.array([f[n] for f in per_fold])
                    metrics.append({"name": n, "partition": "cv", "value": float(series.mean()), "direction": common.DIRECTIONS[n], "std": float(series.std())})
                performance["cvFitSeconds"] = fit_total
                performance["cvModelsBuilt"] = float(len(per_fold))
            return {"metrics": metrics, "performance": performance}
        raise ValueError("Unknown backend command")
    return execute
