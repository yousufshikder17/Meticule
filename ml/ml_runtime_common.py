"""Runtime pieces shared by every Meticule ML runner: the contract envelope, artifact compatibility, partitions,
timestamps and metrics.

Contains no estimator or framework model code. Each runner keeps its own computation and artifact format.
"""
import calendar
import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Meticule ML Backend Contracts. Must equal ML_BACKEND_CONTRACT and ML_BACKEND_CONTRACT_V2 in src/ml/contract.ts. v2 is v1
# plus the Phase 4 vocabulary (tasks beyond classification/regression, column types beyond number/string/boolean, the
# chronological split, datasets without a target); envelope, commands, failure categories and artifacts are unchanged.
# Requests from any other contract are refused, and every successful response echoes the request's contract so the
# TypeScript side can refuse a response from another one.
ML_BACKEND_CONTRACT = "meticule-ml-backend/v1"
ML_BACKEND_CONTRACT_V2 = "meticule-ml-backend/v2"
SUPPORTED_CONTRACTS = (ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2)
TABULAR_COLUMN_TYPES = ("number", "string", "boolean")

# The metric registry mirrors src/ml/metrics.ts (a unit test compares it with the dump printed by check_metrics.py).
# Ranking never infers direction from names. name: (tasks, direction, needs labels, diagnostic). A diagnostic describes
# a model's outputs (threshold transfer, anisotropy), not its quality, so it is never a ranking metric.
METRICS = {
    "accuracy": (("classification",), "higher", True, False),
    "precision": (("classification",), "higher", True, False),
    "recall": (("classification",), "higher", True, False),
    "f1": (("classification",), "higher", True, False),
    "roc_auc": (("classification", "anomaly_detection"), "higher", True, False),
    "log_loss": (("classification",), "lower", True, False),
    "mae": (("regression", "forecasting"), "lower", True, False),
    "mse": (("regression",), "lower", True, False),
    "rmse": (("regression", "forecasting"), "lower", True, False),
    "r2": (("regression",), "higher", True, False),
    "smape": (("forecasting",), "lower", True, False),
    "mase": (("forecasting",), "lower", True, False),
    "average_precision": (("anomaly_detection",), "higher", True, False),
    "anomaly_precision": (("anomaly_detection",), "higher", True, False),
    "anomaly_recall": (("anomaly_detection",), "higher", True, False),
    "anomaly_f1": (("anomaly_detection",), "higher", True, False),
    "threshold_calibration_error": (("anomaly_detection",), "lower", False, True),
    "retrieval_precision_at_1": (("embedding",), "higher", True, False),
    "retrieval_mrr": (("embedding",), "higher", True, False),
    "mean_pairwise_cosine": (("embedding",), "lower", False, True),
}
TASKS = ("classification", "regression", "forecasting", "anomaly_detection", "embedding")
DIRECTIONS = {name: metric[1] for name, metric in METRICS.items()}
TASK_METRICS = {task: tuple(name for name, metric in METRICS.items() if task in metric[0]) for task in TASKS}
LABEL_FREE = frozenset(name for name, metric in METRICS.items() if not metric[2])
DIAGNOSTIC = frozenset(name for name, metric in METRICS.items() if metric[3])
# Default metrics for (labelled, unlabelled) data; None where the task always has a target.
DEFAULTS = {"classification": (("accuracy",), None), "regression": (("mae", "mse"), None),
            "forecasting": (("mae", "rmse"), None),
            "anomaly_detection": (("average_precision",), ("threshold_calibration_error",)),
            "embedding": (("retrieval_precision_at_1",), ("mean_pairwise_cosine",))}


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


def serve(execute, stdin=None, stdout=None, stderr=None, task_of=None):
    """The one stdin/stdout entry point every runner uses.

    A request from an unsupported contract is refused before any work. `task_of(algorithm)` returns the runner's task
    for an algorithm (None when unknown); with it, every request carrying a snapshot passes require_contract before
    `execute`, which never reads the contract. A runner served without it (the frozen Phase 1 adapter) cannot check v2
    semantics, so it speaks v1 only and keeps its exact Phase 1 behaviour.

    A success is the command's JSON stamped with the request's contract. A refusal is one JSON line on stderr, the last
    one, carrying a fixed code and a message written by the runner. Exception text from libraries, paths, dataset values
    and request bodies are never printed. Nothing `execute` prints can reach the protocol channel: Python-level stdout
    goes to stderr while it runs and, on the real process stdout, file descriptor 1 is pointed at stderr for the rest of
    the process (native libraries write to it directly) while the response goes once to a duplicate of the original.
    One request per process. A custom `stdout` object (tests) gets the Python-level redirect only.
    """
    stdin, stderr = stdin or sys.stdin, stderr or sys.stderr
    try:
        request = json.load(stdin)
        contract = request.get("contract") if isinstance(request, dict) else None
        if contract not in SUPPORTED_CONTRACTS or (task_of is None and contract != ML_BACKEND_CONTRACT):
            raise RunnerError("unsupported_contract", "Unsupported ML backend contract")
        if task_of is not None and request.get("command") != "catalog" and isinstance(request.get("snapshot"), dict):
            require_contract(request, task_of(request["snapshot"].get("algorithm")))
        if stdout is None:
            sys.stdout.flush()
            stdout = open(os.dup(1), "w", encoding="utf-8")
            os.dup2(2, 1)
        with contextlib.redirect_stdout(stderr):
            response = execute(request)
        response["contract"] = contract
        text = json.dumps(response, allow_nan=False)
        _settle_output()
        print(text, file=stdout, flush=True)
        return 0
    except RunnerError as error:
        failure = {"error": "RunnerError", "code": error.code, "message": error.message}
    except IncompatibleCheckpoint as error:
        failure = {"error": "IncompatibleCheckpoint", "code": "incompatible_checkpoint", "message": str(error)}
    except InvalidCheckpoint as error:
        failure = {"error": "InvalidCheckpoint", "code": "invalid_checkpoint", "message": str(error)}
    except IncompatibleArtifact as error:
        failure = {"error": "IncompatibleArtifact", "code": "incompatible_artifact", "message": str(error)}
    except Exception as error:
        failure = {"error": type(error).__name__}
    _settle_output()
    print(json.dumps(failure), file=stderr, flush=True)
    return 1


def _settle_output():
    """Pushes out what `execute` left buffered (Python's stdout object, native libraries' C stdio, both now writing to
    stderr) so it lands before the final protocol line. The C flush is best effort: it reaches the C runtime Python
    itself uses, which is the one extension libraries normally share."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):  # no shared C runtime to flush: nothing can be buffered there
        (ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)).fflush(None)


def uses_extensions(spec, task):
    """True when a snapshot needs contract v2. Mirrors contractFor in src/ml/contract.ts exactly: v1 covers only a
    classification or regression task on a random split of a dataset that declares a target and whose columns are all
    number, string or boolean. `task` is the algorithm's catalog task (None when unknown)."""
    schema = spec["dataset"]["schema"]
    return (task not in ("classification", "regression") or spec["split"].get("strategy") != "random"
            or schema.get("target") is None or any(column["type"] not in TABULAR_COLUMN_TYPES for column in schema["columns"]))


def require_contract(request, task):
    """Refuses a request its contract does not cover, before any work. v1 keeps the exact Phase 1-3 semantics, so a v1
    request using Phase 4 vocabulary is refused; a v2 request must name the runner's own task for a known algorithm."""
    if request["contract"] == ML_BACKEND_CONTRACT:
        if uses_extensions(request["snapshot"], task):
            raise RunnerError("unsupported_contract", "Contract v1 does not cover this request")
    elif task is None or request.get("task") != task:
        raise RunnerError("unsupported_task", "The runner does not implement this task for the algorithm")


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


def split_partitions(spec, rows, y):
    """Every runner's prepare. Random: partitions() exactly as before; an unlabelled dataset (`y` None) splits as if every
    target were zero and cannot stratify. Chronological: rows stably sorted by the UTC instant of the time column (ties
    keep row order); the earliest floor(n*train) rows train, the next floor(n*validation) validate, the rest test. The
    sizes are computed exactly as the TypeScript validation computes them (IEEE double product, then floor)."""
    split = spec["split"]
    cv = (spec.get("evaluation") or {}).get("cv")
    if split["strategy"] == "random":
        if y is None:
            if split["stratify"] or (cv and cv["strategy"] == "stratified_kfold"):
                raise ValueError("Stratification needs a target")
            y = np.zeros(len(rows))
        return partitions(spec, y)
    if split["strategy"] != "chronological":
        raise ValueError("Unsupported split strategy")
    if cv:
        raise ValueError("Cross-validation is not supported with a chronological split")
    n = len(rows)
    n_train, n_validation = math.floor(n * split["train"]), math.floor(n * split["validation"])
    if min(n_train, n_validation, n - n_train - n_validation) < 1:
        raise RunnerError("invalid_time_series", "The chronological split leaves a partition empty")
    column = split["timeColumn"]
    order = sorted(range(n), key=lambda i: epoch_micros(rows[i].get(column)))
    cut = n_train + n_validation
    return {"indices": {"train": order[:n_train], "validation": order[n_train:cut], "test": order[cut:]}}


# ISO-8601 with an explicit zone (src/ml/media.ts uses the same pattern). ASCII digits only; fullmatch, so no trailing
# newline slips through as it would with `$`.
_TIMESTAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?(?:Z|([+-])(\d{2}):(\d{2}))", re.ASCII)
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def epoch_micros(value):
    """Exact UTC epoch microseconds of a timestamp cell, by the rules media.ts applies (C4): the pattern above in at most
    40 characters, a real proleptic Gregorian date, hour 0-23, minute and second 0-59, offset at most 23:59, year
    1900..2199. Naive timestamps are refused. Never datetime.fromisoformat, whose accepted syntax differs between Python
    versions. An integer, so instants a microsecond apart never tie (a float of seconds can, near 2199)."""
    match = _TIMESTAMP.fullmatch(value) if isinstance(value, str) and len(value) <= 40 else None
    if match:
        year, month, day, hour, minute, second, offset_hours, offset_minutes = (int(match[i] or 0) for i in (1, 2, 3, 4, 5, 6, 9, 10))
        if (1900 <= year <= 2199 and 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1] and hour <= 23
                and minute <= 59 and second <= 59 and offset_hours <= 23 and offset_minutes <= 59):
            offset = (offset_hours * 60 + offset_minutes) * 60 * (-1 if match[8] == "-" else 1)
            seconds = (date(year, month, day).toordinal() - _EPOCH_ORDINAL) * 86400 + hour * 3600 + minute * 60 + second - offset
            return seconds * 1_000_000 + int((match[7] or "").ljust(6, "0"))
    raise RunnerError("invalid_time_series", "Invalid timestamp: use ISO-8601 with an explicit zone")


def parse_timestamp(value):
    """UTC epoch seconds of a timestamp cell (see epoch_micros for the rules); refuses with invalid_time_series."""
    return epoch_micros(value) / 1e6


def default_metrics(task, labelled):
    """The metrics evaluated when a snapshot names none. Tasks that always have a target have no unlabelled default."""
    chosen = DEFAULTS[task][0 if labelled else 1]
    if chosen is None:
        raise ValueError("Task requires a target")
    return list(chosen)


def requested_metrics(spec, task, labelled=True):
    """The metrics a snapshot asks for, or the task defaults. A metric that does not apply to the task, or that needs
    labels an unlabelled dataset does not have, is refused."""
    names = (spec.get("evaluation") or {}).get("metrics") or default_metrics(task, labelled)
    for name in names:
        if name not in TASK_METRICS[task]:
            raise ValueError("Metric does not apply to task")
        if not labelled and name not in LABEL_FREE:
            raise ValueError("Metric needs a target")
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


# ---- Phase 4 metrics. Pure functions over one partition; an undefined value is refused, never reported as NaN. -------

def smape(actual, forecast):
    """Symmetric MAPE on the 0..200 percent scale; a term whose |actual| + |forecast| is 0 counts 0."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    scale = np.abs(actual) + np.abs(forecast)
    terms = np.divide(2 * np.abs(forecast - actual), scale, out=np.zeros_like(scale), where=scale > 0)
    return float(100 * terms.mean())


def mase(actual, forecast, history, period=None):
    """Mean absolute error over the mean absolute seasonal-naive error of `history` (the training block in time order)
    with the model's seasonal period, 1 when it has none. A zero (or empty) scale is refused."""
    period, history = int(period or 1), np.asarray(history, dtype=float)
    scale = float(np.mean(np.abs(history[period:] - history[:-period]))) if len(history) > period else 0.0
    if not scale > 0:
        raise RunnerError("undefined_metric", "MASE is undefined: the training block has no seasonal-naive error")
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))) / scale)


def _both_classes(labels):
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        raise RunnerError("undefined_metric", "Metric is undefined on a partition without both normal and anomalous rows; stratify the split")
    return labels


def anomaly_roc_auc(labels, scores):
    """roc_auc of the anomaly task: labels are booleans (anomaly = True), scores rise with abnormality."""
    return float(roc_auc_score(_both_classes(labels), np.asarray(scores, dtype=float)))


def average_precision(labels, scores):
    return float(average_precision_score(_both_classes(labels), np.asarray(scores, dtype=float)))


def anomaly_scores(labels, flags):
    """anomaly_precision, anomaly_recall and anomaly_f1 of the anomaly class (positive = True), zero_division 0."""
    labels, flags = _both_classes(labels), np.asarray(flags, dtype=bool)
    tp, fp, fn = int(np.sum(labels & flags)), int(np.sum(~labels & flags)), int(np.sum(labels & ~flags))
    return {"anomaly_precision": tp / (tp + fp) if tp + fp else 0.0, "anomaly_recall": tp / (tp + fn),
            "anomaly_f1": 2 * tp / (2 * tp + fp + fn)}


def threshold_calibration_error(flags, contamination):
    """|flagged fraction - contamination|: how the fitted threshold transfers to a partition. A diagnostic that needs no
    labels; it says nothing about detection quality."""
    return float(abs(np.mean(np.asarray(flags, dtype=bool)) - float(contamination)))


def _unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) < 2:
        raise RunnerError("undefined_metric", "Embedding metrics need at least two rows in the partition")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise RunnerError("undefined_metric", "Embedding metrics are undefined for a zero or non-finite vector")
    return vectors / norms[:, None]


def retrieval_scores(vectors, labels, block=256):
    """retrieval_precision_at_1 and retrieval_mrr. Every row queries the rest of its partition by cosine similarity
    (leave-one-out); relevant means the same label. Queries with no other same-label row are excluded, and the metric is
    refused when none remains. A non-relevant row tied with the best relevant one ranks ahead of it, so ties never
    flatter a model (a constant embedding scores 0). Computed in blocks of queries: O(block * n) memory."""
    units = _unit_rows(vectors)
    index = {}
    codes = np.array([index.setdefault(label, len(index)) for label in labels])
    ranks = []
    for start in range(0, len(units), block):
        rows = np.arange(start, min(start + block, len(units)))
        within = np.arange(len(rows))
        similarity = units[rows] @ units.T
        relevant = codes[rows, None] == codes[None, :]
        similarity[within, rows], relevant[within, rows] = -np.inf, False
        best = np.where(relevant, similarity, -np.inf).max(axis=1)
        rank = 1 + np.sum(~relevant & (similarity >= best[:, None]), axis=1)
        ranks.extend(rank[relevant.any(axis=1)].tolist())
    if not ranks:
        raise RunnerError("undefined_metric", "Retrieval metrics are undefined: no row shares its label with another row")
    ranks = np.array(ranks, dtype=float)
    return {"retrieval_precision_at_1": float(np.mean(ranks == 1)), "retrieval_mrr": float(np.mean(1 / ranks))}


def mean_pairwise_cosine(vectors):
    """Mean cosine similarity over distinct pairs (anisotropy; a diagnostic) via the O(n*d) identity over unit vectors:
    (|sum u|^2 - n) / (n(n-1))."""
    units = _unit_rows(vectors)
    total, n = units.sum(axis=0), len(units)
    return float((total @ total - n) / (n * (n - 1)))


def embedding_metrics(names, vectors, labels):
    """The requested embedding metrics for one partition (D1/D2): the retrieval pair needs labels, mean_pairwise_cosine
    does not. Every embedding backend's evaluate uses this, so the edge-case refusals are the same everywhere."""
    retrieval = retrieval_scores(vectors, labels) if {"retrieval_precision_at_1", "retrieval_mrr"} & set(names) else {}
    values = {}
    for name in names:
        if name == "mean_pairwise_cosine":
            values[name] = mean_pairwise_cosine(vectors)
        elif name in retrieval:
            values[name] = retrieval[name]
        else:
            raise ValueError("Metric does not apply to embeddings")
    return values
