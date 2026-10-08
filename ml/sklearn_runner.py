"""Bounded scikit-learn computation protocol. No database, scheduler or client paths."""
import base64
import json
import hashlib
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
import joblib
import threadpoolctl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Model identifiers mirror src/ml/model-catalog.ts; hyperparameters were already allowlisted by the caller.
MODELS = {
    "logistic_regression": (LogisticRegression, "classification"),
    "random_forest_classifier": (RandomForestClassifier, "classification"),
    "gradient_boosting_classifier": (GradientBoostingClassifier, "classification"),
    "svc": (SVC, "classification"),
    "k_neighbors_classifier": (KNeighborsClassifier, "classification"),
    "decision_tree_classifier": (DecisionTreeClassifier, "classification"),
    "linear_regression": (LinearRegression, "regression"),
    "ridge": (Ridge, "regression"),
    "lasso": (Lasso, "regression"),
    "elastic_net": (ElasticNet, "regression"),
    "random_forest_regressor": (RandomForestRegressor, "regression"),
    "gradient_boosting_regressor": (GradientBoostingRegressor, "regression"),
    "svr": (SVR, "regression"),
    "k_neighbors_regressor": (KNeighborsRegressor, "regression"),
    "decision_tree_regressor": (DecisionTreeRegressor, "regression"),
}
SINGLE_THREAD = (RandomForestClassifier, RandomForestRegressor, KNeighborsClassifier, KNeighborsRegressor)
DIRECTIONS = {"accuracy": "higher", "precision": "higher", "recall": "higher", "f1": "higher", "roc_auc": "higher",
              "log_loss": "lower", "mae": "lower", "mse": "lower", "rmse": "lower", "r2": "higher"}
TASK_METRICS = {"classification": ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"),
                "regression": ("mae", "mse", "rmse", "r2")}
DEFAULT_METRICS = {"classification": ["accuracy"], "regression": ["mae", "mse"]}


# Artifact compatibility is an explicit contract, separate from provenance (see docs/ml.md):
#   backend, artifact_format, artifact_schema, protocol  -> what the artifact IS; must be recognised.
#   python major.minor and RUNTIME_KEYS                   -> pickle/numerics compatibility; must match.
#   adapter_sha256, platform, python patch version        -> provenance only; recorded, never compared.
# Bump ARTIFACT_SCHEMA only when a fitted artifact written by one adapter can no longer be used by another:
# the payload is {"pipeline": fitted sklearn Pipeline (last step = estimator, predicts from the matrix that
# to_matrix builds for the pipeline's features), "environment": dict}. Implementation-only edits do not bump it.
BACKEND = "sklearn"
ARTIFACT_FORMAT = "sklearn-pickle-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-sklearn-v2"
# protocol -> artifact schema it implies when the artifact predates explicit schema versions (Phase 1 wrote none).
SUPPORTED_PROTOCOLS = {"meticule-sklearn-v1": 1, PROTOCOL: 1}
SUPPORTED_SCHEMAS = {1}
RUNTIME_KEYS = ("sklearn", "numpy", "scipy", "joblib", "threadpoolctl")


class IncompatibleArtifact(ValueError):
    pass


def environment():
    return {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "artifact_schema": str(ARTIFACT_SCHEMA), "protocol": PROTOCOL,
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def check_compatible(recorded, current):
    """Raise IncompatibleArtifact unless an artifact recorded under `recorded` can be used by this runtime."""
    if not isinstance(recorded, dict):
        raise IncompatibleArtifact("missing environment")
    if recorded.get("backend", BACKEND) != BACKEND:
        raise IncompatibleArtifact("different backend")
    if recorded.get("artifact_format", ARTIFACT_FORMAT) != ARTIFACT_FORMAT:
        raise IncompatibleArtifact("unsupported artifact format")
    protocol = recorded.get("protocol")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise IncompatibleArtifact("unsupported protocol")
    try:
        schema = int(recorded.get("artifact_schema", SUPPORTED_PROTOCOLS[protocol]))
    except (TypeError, ValueError):
        raise IncompatibleArtifact("unreadable artifact schema") from None
    if schema not in SUPPORTED_SCHEMAS:
        raise IncompatibleArtifact("unsupported artifact schema")
    if ".".join(str(recorded.get("python", "")).split(".")[:2]) != ".".join(current["python"].split(".")[:2]):
        raise IncompatibleArtifact("python major.minor differs")
    for key in RUNTIME_KEYS:
        if recorded.get(key) != current[key]:
            raise IncompatibleArtifact(f"{key} version differs")


def feature_types(spec):
    by_name = {c["name"]: c["type"] for c in spec["dataset"]["schema"]["columns"]}
    return [by_name[f] for f in spec["pipeline"]["definition"]["features"]]


def to_matrix(rows, features, types):
    """Numeric-only data stays float (Phase 1 behaviour); otherwise numbers stay float and the rest become strings."""
    if all(t == "number" for t in types):
        return np.array([[np.nan if row[f] is None else row[f] for f in features] for row in rows], dtype=float)

    def cell(value, kind):
        if value is None:
            return np.nan
        if kind == "number":
            return float(value)
        return str(value).lower() if isinstance(value, bool) else str(value)

    return np.array([[cell(row[f], t) for f, t in zip(features, types)] for row in rows], dtype=object)


def build_pipeline(spec):
    """Unfitted preprocessing + estimator. Every call is independent, so each CV fold fits its own copy."""
    definition = spec["pipeline"]["definition"]
    features, types = definition["features"], feature_types(spec)
    numeric = [i for i, t in enumerate(types) if t == "number"]
    categorical = [i for i, t in enumerate(types) if t != "number"]
    num_steps, cat_steps, final = [], [], []
    for index, step in enumerate(definition["steps"]):
        op, params, name = step["operation"], step.get("parameters") or {}, f"step_{index}"
        if op == "impute_median":
            num_steps.append((name, SimpleImputer(strategy="median", keep_empty_features=True)))
        elif op == "impute_mean":
            num_steps.append((name, SimpleImputer(strategy="mean", keep_empty_features=True)))
        elif op == "standard_scale":
            num_steps.append((name, StandardScaler()))
        elif op == "min_max_scale":
            num_steps.append((name, MinMaxScaler()))
        elif op == "impute_most_frequent":
            cat_steps.append((name, SimpleImputer(strategy="most_frequent", missing_values=np.nan, keep_empty_features=True)))
        elif op == "one_hot_encode":
            cat_steps.append((name, OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif op == "ordinal_encode":
            order = [params["categories"][features[i]] for i in categorical]
            cat_steps.append((name, OrdinalEncoder(categories=order, handle_unknown="use_encoded_value", unknown_value=-1)))
        elif op == "variance_threshold":
            final.append((name, VarianceThreshold(threshold=params.get("threshold", 0.0))))
        else:
            raise ValueError("Unsupported preprocessing operation")
    branches = []
    if numeric:
        branches.append(("numeric", Pipeline(num_steps) if num_steps else "passthrough", numeric))
    if categorical:
        branches.append(("categorical", Pipeline(cat_steps), categorical))
    cls, _ = MODELS[spec["algorithm"]]
    params = dict(spec["hyperparameters"])
    accepted = cls().get_params()
    if "random_state" in accepted:
        params["random_state"] = spec["seed"]
    if issubclass(cls, SINGLE_THREAD):
        params["n_jobs"] = 1
    return Pipeline([("preprocess", ColumnTransformer(branches, sparse_threshold=0.0)), *final, ("estimator", cls(**params))])


def is_classification(spec):
    return MODELS[spec["algorithm"]][1] == "classification"


def requested_metrics(spec):
    task = MODELS[spec["algorithm"]][1]
    names = (spec.get("evaluation") or {}).get("metrics") or DEFAULT_METRICS[task]
    for name in names:
        if name not in TASK_METRICS[task]:
            raise ValueError("Metric does not apply to task")
    return names


def scores(pipeline, x):
    estimator = pipeline[-1]
    if hasattr(estimator, "predict_proba"):
        return pipeline.predict_proba(x)
    if hasattr(estimator, "decision_function"):
        return pipeline.decision_function(x)
    raise ValueError("Estimator exposes no scores")


def measure(names, spec, pipeline, x, y):
    """Compute only the requested metrics; an unavailable one is an error, never a silent omission."""
    prediction = pipeline.predict(x)
    values = {}
    for name in names:
        if name == "accuracy":
            values[name] = accuracy_score(y, prediction)
        elif name in ("precision", "recall", "f1"):
            fn = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[name]
            values[name] = fn(y, prediction, average="weighted", zero_division=0)
        elif name == "roc_auc":
            s, classes = scores(pipeline, x), pipeline[-1].classes_
            if len(classes) == 2:
                values[name] = roc_auc_score(y, s[:, 1] if s.ndim == 2 else s)
            elif s.ndim == 2 and hasattr(pipeline[-1], "predict_proba"):
                values[name] = roc_auc_score(y, s, multi_class="ovr", labels=classes)
            else:
                raise ValueError("Multiclass roc_auc needs probabilities")
        elif name == "log_loss":
            if not hasattr(pipeline[-1], "predict_proba"):
                raise ValueError("log_loss needs probabilities")
            values[name] = log_loss(y, pipeline.predict_proba(x), labels=pipeline[-1].classes_)
        elif name == "mae":
            values[name] = mean_absolute_error(y, prediction)
        elif name == "mse":
            values[name] = mean_squared_error(y, prediction)
        elif name == "rmse":
            values[name] = float(np.sqrt(mean_squared_error(y, prediction)))
        elif name == "r2":
            values[name] = r2_score(y, prediction)
    return {k: float(v) for k, v in values.items()}


def execute(request):
    env = environment()
    if request["command"] == "catalog":
        return {"models": sorted(MODELS), "metrics": DIRECTIONS}
    spec = request["snapshot"]
    rows = request["rows"]
    if request.get("expectedEnvironment") is not None:
        # Recorded in PostgreSQL and checked before any artifact bytes are unpickled.
        check_compatible(request["expectedEnvironment"], env)
    features = spec["pipeline"]["definition"]["features"]
    x = to_matrix(rows, features, feature_types(spec))
    command = request["command"]
    if command == "predict":
        # Only checksum-verified worker-produced artifacts reach this process. Never accept uploads.
        model = pickle.loads(base64.b64decode(request["artifact"], validate=True))
        check_compatible(model["environment"], env)
        return {"predictions": model["pipeline"].predict(x).tolist()}
    y = np.array([row[spec["dataset"]["schema"]["target"]] for row in rows])
    cv = (spec.get("evaluation") or {}).get("cv")
    if command == "prepare":
        split = spec["split"]
        indices = np.arange(len(rows))
        train, holdout = train_test_split(indices, train_size=split["train"], random_state=spec["seed"],
                                          stratify=y if split["stratify"] else None)
        validation, test = train_test_split(holdout, train_size=split["validation"] / (split["validation"] + split["test"]),
                                            random_state=spec["seed"], stratify=y[holdout] if split["stratify"] else None)
        result = {"indices": {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}, "environment": env}
        if cv:
            stratified = cv["strategy"] == "stratified_kfold"
            splitter = (StratifiedKFold if stratified else KFold)(n_splits=cv["folds"], shuffle=True, random_state=spec["seed"])
            result["folds"] = [train[val].tolist() for _, val in splitter.split(np.zeros(len(train)), y[train] if stratified else None)]
        return result
    if command == "train":
        pipeline = build_pipeline(spec)
        train = request["prepared"]["indices"]["train"]
        # Preprocessing and estimator are fitted together on training rows only; no other partition fits anything.
        started = time.perf_counter()
        pipeline.fit(x[train], y[train])
        fit_seconds = time.perf_counter() - started
        payload = pickle.dumps({"pipeline": pipeline, "environment": env}, protocol=5)
        return {"artifact": base64.b64encode(payload).decode(), "format": "sklearn-pickle-v1", "environment": env,
                "resolvedHyperparameters": pipeline[-1].get_params(deep=False), "performance": {"fitSeconds": fit_seconds}}
    if command == "evaluate":
        model = pickle.loads(base64.b64decode(request["artifact"], validate=True))
        check_compatible(model["environment"], env)
        names = requested_metrics(spec)
        metrics, performance = [], {}

        def emit(partition, values, **extra):
            metrics.extend({"name": n, "partition": partition, "value": v, "direction": DIRECTIONS[n], **extra} for n, v in values.items())

        prepared = request["prepared"]["indices"]
        for partition in ("train", "validation", "test"):
            indices = prepared[partition]
            started = time.perf_counter()
            values = measure(names, spec, model["pipeline"], x[indices], y[indices])
            if partition == "test":
                performance["predictSeconds"] = time.perf_counter() - started
            emit(partition, values)
        if cv:
            folds = request["prepared"]["folds"]
            pool = np.array(prepared["train"])
            per_fold, fit_total = [], 0.0
            for number, held_out in enumerate(folds):
                fit_rows = pool[~np.isin(pool, held_out)]
                fold_model = build_pipeline(spec)  # fresh, unfitted: preprocessing sees only this fold's training rows
                started = time.perf_counter()
                fold_model.fit(x[fit_rows], y[fit_rows])
                fit_total += time.perf_counter() - started
                values = measure(names, spec, fold_model, x[held_out], y[held_out])
                per_fold.append(values)
                emit("cv_fold", values, fold=number)
            for n in names:
                series = np.array([f[n] for f in per_fold])
                metrics.append({"name": n, "partition": "cv", "value": float(series.mean()), "direction": DIRECTIONS[n], "std": float(series.std())})
            performance["cvFitSeconds"] = fit_total
        return {"metrics": metrics, "performance": performance}
    raise ValueError("Unknown backend command")


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        # Do not echo dataset values, labels, paths or request bodies through API errors.
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
