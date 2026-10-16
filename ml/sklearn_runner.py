"""Bounded scikit-learn computation protocol. No database, scheduler or client paths."""
import base64
import json
import pickle
import sys
import time

import numpy as np
import scipy
import sklearn
import joblib
import threadpoolctl

import ml_runtime_common as common
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
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
    # Anomaly detection (Phase 4): unsupervised; labels, when present, are used only for evaluation.
    "isolation_forest": (IsolationForest, "anomaly_detection"),
    "local_outlier_factor": (LocalOutlierFactor, "anomaly_detection"),
    "robust_covariance": (EllipticEnvelope, "anomaly_detection"),
}
SINGLE_THREAD = (RandomForestClassifier, RandomForestRegressor, KNeighborsClassifier, KNeighborsRegressor, IsolationForest, LocalOutlierFactor)
DEFAULT_METRICS = {"classification": ["accuracy"], "regression": ["mae", "mse"]}


# Artifact compatibility is an explicit contract, separate from provenance (see docs/ml.md):
#   backend, artifact_format, artifact_schema, protocol  -> what the artifact IS; must be recognised.
#   python major.minor and RUNTIME_KEYS                   -> pickle/numerics compatibility; must match.
#   adapter_sha256, platform, python patch version        -> provenance only; recorded, never compared.
# Bump ARTIFACT_SCHEMA only when a fitted artifact written by one adapter can no longer be used by another:
# the payload is {"pipeline": fitted sklearn Pipeline (last step = estimator, predicts from the matrix that
# to_matrix builds for the pipeline's features), "environment": dict}. Implementation-only edits do not bump it.
# Anomaly-detection artifacts are schema 2: the same payload plus {"anomaly": {threshold, contamination, trainScores}}.
# Classification and regression keep writing schema 1, so their artifacts load in adapters that know only schema 1.
BACKEND = "sklearn"
ARTIFACT_FORMAT = "sklearn-pickle-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-sklearn-v2"
# protocol -> artifact schema it implies when the artifact predates explicit schema versions (Phase 1 wrote none).
SUPPORTED_PROTOCOLS = {"meticule-sklearn-v1": 1, PROTOCOL: 1}
SUPPORTED_SCHEMAS = {1, 2}
ANOMALY_SCHEMA = 2
RUNTIME_KEYS = ("sklearn", "numpy", "scipy", "joblib", "threadpoolctl")


IncompatibleArtifact = common.IncompatibleArtifact
TASK_METRICS = common.TASK_METRICS
# This runner's metrics only, so the catalog lists what it computes (the shared registry also covers Phase 4 tasks).
DIRECTIONS = {name: common.DIRECTIONS[name] for task in ("classification", "regression") for name in TASK_METRICS[task]}
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": SUPPORTED_PROTOCOLS,
            "schemas": SUPPORTED_SCHEMAS, "runtime_keys": RUNTIME_KEYS}


def environment():
    return common.base_environment([__file__, common.__file__], BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                    "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def check_compatible(recorded, current):
    """Raise IncompatibleArtifact unless an artifact recorded under `recorded` can be used by this runtime."""
    common.check_compatible(recorded, current, CONTRACT)


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
    if MODELS[spec["algorithm"]][1] == "anomaly_detection":
        params.setdefault("contamination", 0.1)  # the catalog's numeric default (scikit-learn's own is "auto" for some models)
    if cls is IsolationForest and params.get("max_samples") is None:
        params["max_samples"] = "auto"  # the catalog's null means scikit-learn's own default
    if cls is LocalOutlierFactor:
        params["novelty"] = True  # score_samples on new rows; training rows use negative_outlier_factor_
    accepted = cls().get_params()
    if "random_state" in accepted:
        params["random_state"] = spec["seed"]
    if issubclass(cls, SINGLE_THREAD):
        params["n_jobs"] = 1
    return Pipeline([("preprocess", ColumnTransformer(branches, sparse_threshold=0.0)), *final, ("estimator", cls(**params))])


def is_classification(spec):
    return MODELS[spec["algorithm"]][1] == "classification"


def requested_metrics(spec):
    return common.requested_metrics(spec, MODELS[spec["algorithm"]][1])


def measure(names, spec, pipeline, x, y):
    """Estimator-side extraction of predictions and scores, then the shared metric computation."""
    estimator = pipeline[-1]
    prediction = pipeline.predict(x)
    proba = pipeline.predict_proba(x) if hasattr(estimator, "predict_proba") else None
    scores_ = proba if proba is not None else (pipeline.decision_function(x) if hasattr(estimator, "decision_function") else None)
    classes = estimator.classes_ if hasattr(estimator, "classes_") else None
    return common.measure_predictions(names, y, prediction, proba=proba, scores=scores_, classes=classes)


def load_artifact(encoded, current_env):
    """Unpickles a worker-produced artifact only after its recorded environment is accepted. Unreadable bytes are refused
    as an incompatible artifact, never reported as a framework exception."""
    try:
        model = pickle.loads(base64.b64decode(encoded, validate=True))
    except Exception:
        raise IncompatibleArtifact("unreadable artifact") from None
    if not isinstance(model, dict) or "pipeline" not in model or "environment" not in model:
        raise IncompatibleArtifact("malformed artifact")
    check_compatible(model["environment"], current_env)
    return model


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
        model = load_artifact(request["artifact"], env)
        if MODELS[spec["algorithm"]][1] == "anomaly_detection":
            return {"predictions": anomaly_outputs(model, x)}
        return {"predictions": model["pipeline"].predict(x).tolist()}
    if MODELS[spec["algorithm"]][1] == "anomaly_detection":
        return execute_anomaly(request, spec, rows, x, env)
    y = np.array([row[spec["dataset"]["schema"]["target"]] for row in rows])
    cv = (spec.get("evaluation") or {}).get("cv")
    if command == "prepare":
        result = {**common.split_partitions(spec, rows, y), "environment": env}
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
        model = load_artifact(request["artifact"], env)
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


# ---- anomaly detection --------------------------------------------------------------------------------------------------
# Score = -score_samples, so a higher score is more anomalous. LocalOutlierFactor's score_samples must not be used on its
# own training rows, so training scores come from negative_outlier_factor_ (negated), and every model stores its training
# scores. Threshold = numpy quantile (linear interpolation) of the training scores at 1 - contamination; a row is an anomaly
# when its score is strictly above it.

def anomaly_scores(model, x):
    return (-model["pipeline"].score_samples(x)).astype(float)


def anomaly_outputs(model, x):
    if "anomaly" not in model:
        raise IncompatibleArtifact("artifact is not an anomaly-detection artifact")
    scores = anomaly_scores(model, x) if len(x) else np.zeros(0)
    return [{"score": float(s), "anomaly": bool(s > model["anomaly"]["threshold"])} for s in scores]


def execute_anomaly(request, spec, rows, x, env):
    """Unsupervised fit on training rows only; a boolean target, when present, is used only to evaluate."""
    target = spec["dataset"]["schema"].get("target")
    labels = np.array([bool(row[target]) for row in rows]) if target is not None else None
    command = request["command"]
    if (spec.get("evaluation") or {}).get("cv"):
        raise ValueError("Cross-validation is not supported for anomaly detection")
    if command == "prepare":
        return {**common.split_partitions(spec, rows, labels), "environment": env}
    anomaly_env = {**env, "artifact_schema": str(ANOMALY_SCHEMA)}
    if command == "train":
        pipeline = build_pipeline(spec)
        train = request["prepared"]["indices"]["train"]
        contamination = float(pipeline[-1].get_params()["contamination"])
        started = time.perf_counter()
        pipeline.fit(x[train])
        fit_seconds = time.perf_counter() - started
        estimator = pipeline[-1]
        train_scores = (-estimator.negative_outlier_factor_ if isinstance(estimator, LocalOutlierFactor) else -pipeline.score_samples(x[train])).astype(float)
        threshold = float(np.quantile(train_scores, 1.0 - contamination))
        payload = pickle.dumps({"pipeline": pipeline, "environment": anomaly_env,
                                "anomaly": {"threshold": threshold, "contamination": contamination, "trainScores": train_scores.tolist()}}, protocol=5)
        resolved = {k: v for k, v in estimator.get_params(deep=False).items() if not callable(v)}
        return {"artifact": base64.b64encode(payload).decode(), "format": ARTIFACT_FORMAT, "environment": anomaly_env,
                "resolvedHyperparameters": resolved, "performance": {"fitSeconds": fit_seconds}}
    if command == "evaluate":
        model = load_artifact(request["artifact"], env)
        if "anomaly" not in model:
            raise IncompatibleArtifact("artifact is not an anomaly-detection artifact")
        names = common.requested_metrics(spec, "anomaly_detection", labels is not None)
        prepared, metrics, performance = request["prepared"]["indices"], [], {}
        threshold, contamination = model["anomaly"]["threshold"], model["anomaly"]["contamination"]
        for partition in ("train", "validation", "test"):
            indices = prepared[partition]
            started = time.perf_counter()
            scores = np.array(model["anomaly"]["trainScores"]) if partition == "train" else anomaly_scores(model, x[indices])
            if len(scores) != len(indices):
                raise IncompatibleArtifact("training scores do not match the training partition")
            flags = scores > threshold
            values = {}
            for name in names:
                if name == "threshold_calibration_error":
                    values[name] = common.threshold_calibration_error(flags, contamination)
                elif name == "roc_auc":
                    values[name] = common.anomaly_roc_auc(labels[indices], scores)
                elif name == "average_precision":
                    values[name] = common.average_precision(labels[indices], scores)
                else:
                    values[name] = common.anomaly_scores(labels[indices], flags)[name]
            if partition == "test":
                performance["predictSeconds"] = time.perf_counter() - started
            metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n]} for n, v in values.items())
        return {"metrics": metrics, "performance": performance}
    raise ValueError("Unknown backend command")


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=lambda algorithm: MODELS[algorithm][1] if algorithm in MODELS else None))
