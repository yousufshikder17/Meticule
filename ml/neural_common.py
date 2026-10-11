"""Framework-neutral pieces shared by the neural runners: the hyperparameter allowlist, NumPy preprocessing whose fitted
state is plain JSON, targets and training fingerprints. Imports no deep-learning framework, so each runner keeps its own
environment and its own network code.
"""
import hashlib
import json
import math
from collections import Counter

import numpy as np

# Mirror src/ml/pytorch-models.ts and src/ml/tensorflow-models.ts. Limits are enforced again here, because the runner
# process is the last line of defence.
MODELS = {"mlp_classifier": "classification", "mlp_regressor": "regression"}
LIMITS = {"maxLayers": 4, "maxWidth": 256, "maxTotalUnits": 512, "maxEpochs": 200, "maxBatchSize": 256}
ACTIVATION_NAMES = ("relu", "gelu", "tanh")
OPTIMIZER_NAMES = ("sgd", "adam", "adamw")
TASK_LOSS = {"classification": ("cross_entropy",), "regression": ("mse", "l1")}
DEFAULT_LOSS = {"classification": "cross_entropy", "regression": "mse"}
DEFAULTS = {"hiddenLayers": [64, 32], "activation": "relu", "dropout": 0.0, "epochs": 20, "batchSize": 32,
            "learningRate": 0.001, "optimizer": "adam"}
NUMERIC_OPS = ("impute_median", "impute_mean", "standard_scale", "min_max_scale")
CATEGORICAL_OPS = ("impute_most_frequent", "one_hot_encode", "ordinal_encode")
IMPUTERS = ("impute_median", "impute_mean", "impute_most_frequent")


def resolve_hyperparameters(spec):
    """Allowlisted, bounded configuration with the same defaults as the TypeScript catalog."""
    supplied = spec.get("hyperparameters") or {}
    task = MODELS[spec["algorithm"]]
    hp = {**DEFAULTS, "loss": DEFAULT_LOSS[task], **supplied}
    unknown = set(hp) - set(DEFAULTS) - {"loss"}
    if unknown:
        raise ValueError("Unsupported neural hyperparameter")
    if hp["loss"] not in TASK_LOSS[task] or hp["optimizer"] not in OPTIMIZER_NAMES or hp["activation"] not in ACTIVATION_NAMES:
        raise ValueError("Unsupported neural training option")
    layers = hp["hiddenLayers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= LIMITS["maxLayers"]:
        raise ValueError("Invalid hidden layer count")
    if any(not isinstance(w, int) or isinstance(w, bool) or not 1 <= w <= LIMITS["maxWidth"] for w in layers):
        raise ValueError("Invalid hidden layer width")
    if sum(layers) > LIMITS["maxTotalUnits"]:
        raise ValueError("Architecture exceeds the total unit limit")
    if not 0.0 <= float(hp["dropout"]) <= 0.8:
        raise ValueError("Dropout out of range")
    if not isinstance(hp["epochs"], int) or not 1 <= hp["epochs"] <= LIMITS["maxEpochs"]:
        raise ValueError("Epochs out of range")
    if not isinstance(hp["batchSize"], int) or not 1 <= hp["batchSize"] <= LIMITS["maxBatchSize"]:
        raise ValueError("Batch size out of range")
    if not 0.0 < float(hp["learningRate"]) <= 1.0:
        raise ValueError("Learning rate out of range")
    return hp


# ---- preprocessing: numpy implementation whose fitted state is plain JSON ----------------------------------------

def norm_cat(value):
    """Categorical values compare as strings; booleans use the same spelling as the sklearn runner."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def feature_types(spec):
    by_name = {c["name"]: c["type"] for c in spec["dataset"]["schema"]["columns"]}
    return [by_name[f] for f in spec["pipeline"]["definition"]["features"]]


def fit_numeric(operations, values):
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    states = []
    for op in operations:
        known = arr[~np.isnan(arr)]
        if op in ("impute_median", "impute_mean"):
            fill = 0.0 if known.size == 0 else float(np.median(known) if op == "impute_median" else np.mean(known))
            states.append({"op": op, "fill": fill})
            arr = np.where(np.isnan(arr), fill, arr)
        else:
            if op == "standard_scale":
                center = float(known.mean()) if known.size else 0.0
                spread = float(known.std()) if known.size else 1.0
            else:
                center = float(known.min()) if known.size else 0.0
                spread = float(known.max() - known.min()) if known.size else 1.0
            if not math.isfinite(spread) or spread == 0.0:
                spread = 1.0
            states.append({"op": op, "center": center, "spread": spread})
            arr = (arr - center) / spread
    return states


def apply_numeric(states, values):
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    for state in states:
        if state["op"] in IMPUTERS:
            arr = np.where(np.isnan(arr), state["fill"], arr)
        else:
            arr = (arr - state["center"]) / state["spread"]
    if np.isnan(arr).any():
        raise ValueError("Missing numeric value reached the neural model; add an imputer to the pipeline")
    return arr.reshape(-1, 1)


def fit_categorical(steps, feature, values):
    current = [norm_cat(v) for v in values]
    states = []
    for step in steps:
        op = step["operation"]
        if op == "impute_most_frequent":
            counts = Counter(v for v in current if v is not None)
            fill = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else None
            states.append({"op": op, "fill": fill})
            current = [fill if v is None else v for v in current]
        elif op == "one_hot_encode":
            states.append({"op": op, "categories": sorted({v for v in current if v is not None})})
        elif op == "ordinal_encode":
            states.append({"op": op, "categories": [norm_cat(c) for c in step["parameters"]["categories"][feature]]})
    if not states or states[-1]["op"] not in ("one_hot_encode", "ordinal_encode"):
        raise ValueError("Categorical feature has no encoder")
    return states


def apply_categorical(states, values):
    current = [norm_cat(v) for v in values]
    for state in states:
        op = state["op"]
        if op == "impute_most_frequent":
            current = [state["fill"] if v is None else v for v in current]
        elif op == "one_hot_encode":
            index = {c: i for i, c in enumerate(state["categories"])}
            matrix = np.zeros((len(current), len(index)), dtype=float)
            for row, v in enumerate(current):
                if v in index:
                    matrix[row, index[v]] = 1.0
            return matrix
        elif op == "ordinal_encode":
            index = {c: i for i, c in enumerate(state["categories"])}
            return np.array([float(index.get(v, -1)) if v is not None else -1.0 for v in current]).reshape(-1, 1)
    raise ValueError("Categorical feature has no encoder")


def fit_preprocessing(spec, rows, fit_indices):
    """Fitted only on the given training rows. Output columns follow the pipeline's feature order."""
    definition = spec["pipeline"]["definition"]
    steps = definition["steps"]
    numeric_ops = [s["operation"] for s in steps if s["operation"] in NUMERIC_OPS]
    categorical_steps = [s for s in steps if s["operation"] in CATEGORICAL_OPS]
    columns = []
    for feature, kind in zip(definition["features"], feature_types(spec)):
        values = [rows[i][feature] for i in fit_indices]
        if kind == "number":
            columns.append({"feature": feature, "kind": "number", "states": fit_numeric(numeric_ops, values)})
        else:
            columns.append({"feature": feature, "kind": "categorical", "states": fit_categorical(categorical_steps, feature, values)})
    return {"columns": columns}


def transform(preprocessing, rows):
    blocks = []
    for column in preprocessing["columns"]:
        values = [row[column["feature"]] for row in rows]
        blocks.append(apply_numeric(column["states"], values) if column["kind"] == "number" else apply_categorical(column["states"], values))
    if not blocks:
        return np.zeros((len(rows), 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32)


def target_vector(task, raw_values, target_info):
    """Classification targets are class indices; regression targets are standardised with the training statistics."""
    if task == "classification":
        index = {c: i for i, c in enumerate(target_info["classes"])}
        if any(v not in index for v in raw_values):
            raise ValueError("Label not seen in the training partition")
        return np.array([index[v] for v in raw_values], dtype=np.int64)
    values = np.array(raw_values, dtype=float)
    return ((values - target_info["mean"]) / target_info["std"]).astype(np.float32).reshape(-1, 1)


def fingerprint(spec, hp, prepared):
    """Identity of one training configuration. A checkpoint may only continue the configuration that wrote it."""
    precision = (spec.get("training") or {}).get("precision", "fp32")
    payload = {"algorithm": spec["algorithm"], "hyperparameters": hp, "seed": spec["seed"], "precision": precision,
               "features": spec["pipeline"]["definition"], "target": spec["dataset"]["schema"]["target"],
               "fit": prepared["indices"]["train"], "validation": prepared["indices"]["validation"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
