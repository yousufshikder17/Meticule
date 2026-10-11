"""Runtime pieces shared by every Meticule ML runner: artifact compatibility, partitions and metrics.

Contains no estimator or framework model code. Each runner keeps its own computation and artifact format.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Meticule ML Backend Contract v1. Must equal ML_BACKEND_CONTRACT in src/ml/contract.ts: requests are refused otherwise,
# and every successful response echoes it back so the TypeScript side can refuse a response from another contract.
ML_BACKEND_CONTRACT = "meticule-ml-backend/v1"

# Metric directions are declared once here and mirror src/ml/metrics.ts. Ranking never infers direction from names.
DIRECTIONS = {"accuracy": "higher", "precision": "higher", "recall": "higher", "f1": "higher", "roc_auc": "higher",
              "log_loss": "lower", "mae": "lower", "mse": "lower", "rmse": "lower", "r2": "higher"}
TASK_METRICS = {"classification": ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"),
                "regression": ("mae", "mse", "rmse", "r2")}
DEFAULT_METRICS = {"classification": ["accuracy"], "regression": ["mae", "mse"]}


class IncompatibleArtifact(ValueError):
    """A model artifact this runtime must not load: different backend, format, schema, protocol or library versions."""


class IncompatibleCheckpoint(IncompatibleArtifact):
    """A checkpoint from another configuration, backend, format or runtime. Resume is refused, never guessed."""


class InvalidCheckpoint(ValueError):
    """A checkpoint that is unreadable, malformed or tampered with. Distinct from a valid checkpoint that does not fit."""


class RunnerError(Exception):
    """A refusal whose code and message are safe to show to the caller. Everything else stays generic."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def serve(execute, stdin=None, stdout=None, stderr=None):
    """The one stdin/stdout entry point every runner uses.

    A request from another contract is refused before any work. A success is the command's JSON with the contract
    stamped on it. A refusal is one JSON line on stderr carrying a fixed code and a message written by the runner.
    Exception text from libraries, paths, dataset values and request bodies are never printed.
    """
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    try:
        request = json.load(stdin)
        if not isinstance(request, dict) or request.get("contract") != ML_BACKEND_CONTRACT:
            raise RunnerError("unsupported_contract", "Unsupported ML backend contract")
        response = execute(request)
        response["contract"] = ML_BACKEND_CONTRACT
        print(json.dumps(response, allow_nan=False), file=stdout)
        return 0
    except RunnerError as error:
        print(json.dumps({"error": "RunnerError", "code": error.code, "message": error.message}), file=stderr)
    except IncompatibleCheckpoint as error:
        print(json.dumps({"error": "IncompatibleCheckpoint", "code": "incompatible_checkpoint", "message": str(error)}), file=stderr)
    except InvalidCheckpoint as error:
        print(json.dumps({"error": "InvalidCheckpoint", "code": "invalid_checkpoint", "message": str(error)}), file=stderr)
    except IncompatibleArtifact as error:
        print(json.dumps({"error": "IncompatibleArtifact", "code": "incompatible_artifact", "message": str(error)}), file=stderr)
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=stderr)
    return 1


# Artifact compatibility is an explicit contract, separate from provenance (see docs/ml.md):
#   backend, artifact_format, artifact_schema, protocol   -> what the artifact IS; must be recognised.
#   python major.minor and the runner's runtime_keys      -> loader and numerics compatibility; must match.
#   adapter_sha256, platform, python patch version        -> provenance only; recorded, never compared.
def python_minor(version):
    return ".".join(str(version).split(".")[:2])


def runner_closure_sha256(*files):
    """Provenance of the runner: every source file that defines its behaviour, hashed in order."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def base_environment(adapter_files, backend, artifact_format, artifact_schema, protocol, versions):
    """Provenance and contract fields every runner records. `versions` holds library versions as strings.

    `adapter_files` is the runner's own file plus the shared modules it imports, so editing any of them
    changes the provenance checksum (never the compatibility decision).
    """
    return {"backend": backend, "artifact_format": artifact_format, "artifact_schema": str(artifact_schema), "protocol": protocol,
            "python": platform.python_version(), "platform": platform.platform(),
            "adapter_sha256": runner_closure_sha256(*adapter_files), **versions}


def check_compatible(recorded, current, contract):
    """Raise IncompatibleArtifact unless an artifact recorded under `recorded` can be used by this runtime.

    `contract` is a runner's declaration: backend, artifact_format, protocols ({protocol: implied schema}),
    schemas (supported artifact schema versions) and runtime_keys (library versions that must match exactly).
    """
    if not isinstance(recorded, dict):
        raise IncompatibleArtifact("missing environment")
    if recorded.get("backend", contract["backend"]) != contract["backend"]:
        raise IncompatibleArtifact("different backend")
    if recorded.get("artifact_format", contract["artifact_format"]) != contract["artifact_format"]:
        raise IncompatibleArtifact("unsupported artifact format")
    protocol = recorded.get("protocol")
    if protocol not in contract["protocols"]:
        raise IncompatibleArtifact("unsupported protocol")
    try:
        schema = int(recorded.get("artifact_schema", contract["protocols"][protocol]))
    except (TypeError, ValueError):
        raise IncompatibleArtifact("unreadable artifact schema") from None
    if schema not in contract["schemas"]:
        raise IncompatibleArtifact("unsupported artifact schema")
    if python_minor(recorded.get("python", "")) != python_minor(current["python"]):
        raise IncompatibleArtifact("python major.minor differs")
    for key in contract["runtime_keys"]:
        if recorded.get(key) != current[key]:
            raise IncompatibleArtifact(f"{key} version differs")


def partitions(spec, y):
    """Train/validation/test indices and optional CV folds. Identical for every backend with the same spec."""
    split = spec["split"]
    indices = np.arange(len(y))
    train, holdout = train_test_split(indices, train_size=split["train"], random_state=spec["seed"],
                                      stratify=y if split["stratify"] else None)
    validation, test = train_test_split(holdout, train_size=split["validation"] / (split["validation"] + split["test"]),
                                        random_state=spec["seed"], stratify=y[holdout] if split["stratify"] else None)
    result = {"indices": {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}}
    cv = (spec.get("evaluation") or {}).get("cv")
    if cv:
        stratified = cv["strategy"] == "stratified_kfold"
        splitter = (StratifiedKFold if stratified else KFold)(n_splits=cv["folds"], shuffle=True, random_state=spec["seed"])
        result["folds"] = [train[val].tolist() for _, val in splitter.split(np.zeros(len(train)), y[train] if stratified else None)]
    return result


def requested_metrics(spec, task):
    """The metrics a snapshot asks for, or the task defaults. A metric that does not apply to the task is refused."""
    names = (spec.get("evaluation") or {}).get("metrics") or DEFAULT_METRICS[task]
    for name in names:
        if name not in TASK_METRICS[task]:
            raise ValueError("Metric does not apply to task")
    return names


def measure_predictions(names, y, prediction, proba=None, scores=None, classes=None):
    """Compute only the requested metrics; an unavailable one is an error, never a silent omission.

    `proba` is a class-probability matrix (classifiers that expose one); `scores` is what roc_auc consumes
    (proba, or a decision function for binary SVC); `classes` are the training label values in sorted order.
    """
    values = {}
    for name in names:
        if name == "accuracy":
            values[name] = accuracy_score(y, prediction)
        elif name in ("precision", "recall", "f1"):
            fn = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[name]
            values[name] = fn(y, prediction, average="weighted", zero_division=0)
        elif name == "roc_auc":
            if scores is None or classes is None:
                raise ValueError("roc_auc needs scores")
            if len(classes) == 2:
                values[name] = roc_auc_score(y, scores[:, 1] if scores.ndim == 2 else scores)
            elif scores.ndim == 2 and proba is not None:
                values[name] = roc_auc_score(y, scores, multi_class="ovr", labels=classes)
            else:
                raise ValueError("Multiclass roc_auc needs probabilities")
        elif name == "log_loss":
            if proba is None or classes is None:
                raise ValueError("log_loss needs probabilities")
            values[name] = log_loss(y, proba, labels=classes)
        elif name == "mae":
            values[name] = mean_absolute_error(y, prediction)
        elif name == "mse":
            values[name] = mean_squared_error(y, prediction)
        elif name == "rmse":
            values[name] = float(np.sqrt(mean_squared_error(y, prediction)))
        elif name == "r2":
            values[name] = r2_score(y, prediction)
        else:
            raise ValueError("Unknown metric")
    return {k: float(v) for k, v in values.items()}
