"""NumPy/SciPy audio backend over the Meticule runner protocol: MFCC summary embeddings and a softmax classifier on them.

`audio_features` (task embedding, learning pretrained: a fixed transform, nothing is fitted) returns the MFCC summary vector
of each clip. `audio_mfcc_classifier` (task classification, supervised) standardises those vectors and fits a deterministic
softmax head (linear_heads) on training rows only; cross-validation fits a fresh head per fold. Decoding and features are in
audio_common. Artifacts are NumPy archives (`audio-npz-v1`) read with allow_pickle=False. CPU, fp32 (float64 internally).
"""
import base64
import io
import json
import sys
import time

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl

import audio_common
import linear_heads
import media_common
import ml_runtime_common as common

BACKEND = "audio"
ARTIFACT_FORMAT = "audio-npz-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-audio-v1"
RUNTIME_KEYS = ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": {PROTOCOL: ARTIFACT_SCHEMA}, "schemas": {ARTIFACT_SCHEMA},
            "runtime_keys": RUNTIME_KEYS}
MODELS = {"audio_features": "embedding", "audio_mfcc_classifier": "classification"}
HEAD = {"l2": (0.0, 100.0, 1.0), "maxIter": (10, 1000, 200)}
MAX_METADATA_BYTES = 1024 * 1024
RunnerError = common.RunnerError


def environment():
    return common.base_environment([__file__, common.__file__, media_common.__file__, audio_common.__file__, linear_heads.__file__],
                                   BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                    "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def resolve_hyperparameters(spec):
    supplied = dict(spec.get("hyperparameters") or {})
    head_names = set(HEAD) if MODELS[spec["algorithm"]] == "classification" else set()
    if set(supplied) - set(audio_common.LIMITS) - head_names:
        raise ValueError("Unsupported audio hyperparameter")
    resolved = audio_common.resolve_settings({k: v for k, v in supplied.items() if k in audio_common.LIMITS})
    for name in head_names:
        low, high, default = HEAD[name]
        value = supplied.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or (name == "maxIter" and not isinstance(value, int)):
            raise ValueError(f"Hyperparameter {name} out of range")
        resolved[name] = value
    return resolved


def resolve_device(spec):
    training = spec.get("training") or {}
    if training.get("device", "cpu") == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported by the audio backend")
    if training.get("device", "cpu") != "cpu" or training.get("precision", "fp32") != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")


def feature(spec):
    features = spec["pipeline"]["definition"]["features"]
    if len(features) != 1 or spec["pipeline"]["definition"]["steps"]:
        raise ValueError("Audio models read exactly one audio column and take no preprocessing steps")
    if {c["name"]: c["type"] for c in spec["dataset"]["schema"]["columns"]}.get(features[0]) != "audio":
        raise ValueError("The feature column must be an audio column")
    return features[0]


def serialize(metadata, head):
    arrays = {} if head is None else {f"head/{name}": np.asarray(value, dtype=np.float64) for name, value in head.items()}
    buffer = io.BytesIO()
    np.savez(buffer, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


def load_bundle(artifact, env):
    """Environment and metadata are checked before any array is read; array names and shapes must match the metadata."""
    try:
        archive = np.load(io.BytesIO(base64.b64decode(artifact, validate=True)), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile) or "metadata" not in archive.files or archive.zip.getinfo("metadata.npy").file_size > MAX_METADATA_BYTES:
            raise ValueError
        raw = archive["metadata"]
        if raw.dtype.kind != "U" or raw.shape != ():
            raise ValueError
        metadata = json.loads(str(raw))
        if not isinstance(metadata, dict):
            raise ValueError
    except Exception:
        raise common.IncompatibleArtifact("unreadable artifact") from None
    common.check_compatible(metadata.get("environment"), env, CONTRACT)
    algorithm = metadata.get("algorithm")
    if metadata.get("schema") != ARTIFACT_SCHEMA or metadata.get("backend") != BACKEND or algorithm not in MODELS or metadata.get("task") != MODELS[algorithm]:
        raise common.IncompatibleArtifact("malformed artifact")
    try:
        settings = audio_common.resolve_settings(metadata["settings"])
        width = 2 * settings["nMfcc"]
        head = None
        if MODELS[algorithm] == "classification":
            classes = list(metadata["classes"])
            if set(archive.files) != {"metadata", "head/mean", "head/std", "head/weights", "head/bias"} or not 2 <= len(classes) <= linear_heads.MAX_CLASSES:
                raise ValueError
            expected = {"mean": (width,), "std": (width,), "weights": (width, len(classes)), "bias": (len(classes),)}
            head = {}
            for name, shape in expected.items():
                value = archive[f"head/{name}"]
                if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
                    raise ValueError
                head[name] = value
        elif set(archive.files) != {"metadata"}:
            raise ValueError
    except Exception:
        raise common.IncompatibleArtifact("malformed artifact") from None
    return {"algorithm": algorithm, "settings": settings, "head": head, "classes": metadata.get("classes")}


def outputs(bundle, cells):
    vectors = audio_common.features(cells, bundle["settings"])
    if bundle["head"] is None:
        return {"vectors": vectors}
    proba = linear_heads.predict_proba(bundle["head"], vectors) if len(cells) else np.zeros((0, len(bundle["classes"])))
    classes = bundle["classes"]
    return {"prediction": np.array([classes[i] for i in proba.argmax(axis=1)]), "proba": proba, "classes": np.array(classes)}


def execute(request):
    env = environment()
    spec = request["snapshot"]
    rows = request["rows"]
    if request.get("expectedEnvironment") is not None:
        common.check_compatible(request["expectedEnvironment"], env, CONTRACT)
    command = request["command"]
    task = MODELS[spec["algorithm"]]
    target = spec["dataset"]["schema"].get("target")
    labelled = target is not None
    column = feature(spec)
    resolve_device(spec)
    hp = resolve_hyperparameters(spec)
    settings = {k: hp[k] for k in audio_common.LIMITS}
    if command == "prepare":
        y = np.array([row[target] for row in rows]) if labelled else None
        return {**common.split_partitions(spec, rows, y), "environment": env}
    if command == "train":
        started = time.perf_counter()
        indices = request["prepared"]["indices"]["train"]
        metadata = {"schema": ARTIFACT_SCHEMA, "backend": BACKEND, "task": task, "algorithm": spec["algorithm"], "settings": settings,
                    "embeddingDimension": 2 * settings["nMfcc"], "classes": None}
        head = None
        if task == "classification":
            train_rows = [rows[i] for i in indices]
            classes = sorted({row[target] for row in train_rows})
            if len(classes) < 2:
                raise ValueError("Classification needs at least two classes in the training partition")
            head = linear_heads.fit(audio_common.features([row[column] for row in train_rows], settings), [classes.index(row[target]) for row in train_rows],
                                    len(classes), float(hp["l2"]), hp["maxIter"])
            metadata["classes"] = classes
        trained_env = {**env, **({"embedding_dimension": str(2 * settings["nMfcc"])} if task == "embedding" else {})}
        metadata["environment"] = trained_env
        return {"artifact": serialize(metadata, head), "format": ARTIFACT_FORMAT, "environment": trained_env, "resolvedHyperparameters": hp,
                "performance": {"fitSeconds": time.perf_counter() - started}}
    bundle = load_bundle(request["artifact"], env)
    if command == "predict":
        result = outputs(bundle, [row[column] for row in rows])
        return {"predictions": result["vectors"].tolist() if task == "embedding" else result["prediction"].tolist()}
    if command == "evaluate":
        names = common.requested_metrics(spec, task, labelled)
        y = np.array([row[target] for row in rows]) if labelled else None
        prepared = request["prepared"]["indices"]
        metrics, performance = [], {}

        def emit(partition, values, **extra):
            metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n], **extra} for n, v in values.items())

        for partition in ("train", "validation", "test"):
            indices = prepared[partition]
            started = time.perf_counter()
            result = outputs(bundle, [rows[i][column] for i in indices])
            if task == "embedding":
                values = common.embedding_metrics(names, result["vectors"], y[indices] if labelled else None)
            else:
                values = common.measure_predictions(names, y[indices], result["prediction"], proba=result["proba"], scores=result["proba"], classes=result["classes"])
            if partition == "test":
                performance["predictSeconds"] = time.perf_counter() - started
            emit(partition, values)
        cv = (spec.get("evaluation") or {}).get("cv")
        if cv:
            if task != "classification":
                raise ValueError("Cross-validation is not supported for this task")
            pool = np.array(prepared["train"])
            vectors = audio_common.features([rows[i][column] for i in pool], settings)
            position = {int(i): k for k, i in enumerate(pool)}
            per_fold, fit_total = [], 0.0
            for number, held_out in enumerate(request["prepared"]["folds"]):
                fit_rows = [int(i) for i in pool[~np.isin(pool, held_out)]]
                classes = sorted({rows[i][target] for i in fit_rows})
                started = time.perf_counter()
                head = linear_heads.fit(vectors[[position[i] for i in fit_rows]], [classes.index(rows[i][target]) for i in fit_rows], len(classes), float(hp["l2"]), hp["maxIter"])
                fit_total += time.perf_counter() - started
                proba = linear_heads.predict_proba(head, vectors[[position[int(i)] for i in held_out]])
                values = common.measure_predictions(names, y[held_out], np.array([classes[i] for i in proba.argmax(axis=1)]), proba=proba, scores=proba, classes=np.array(classes))
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
