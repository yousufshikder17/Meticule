"""Bounded scikit-learn computation protocol. No database, scheduler or client paths."""
import base64
import json
import hashlib
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import scipy
import sklearn
import joblib
import threadpoolctl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def environment():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "protocol": "meticule-sklearn-v1"}


def execute(request):
    spec = request["snapshot"]
    rows = request["rows"]
    env = environment()
    if request.get("expectedEnvironment") is not None and request["expectedEnvironment"] != env:
        raise ValueError("Runtime environment differs from the training environment")
    features = spec["pipeline"]["definition"]["features"]
    x = np.array([[np.nan if row[f] is None else row[f] for f in features] for row in rows], dtype=float)
    command = request["command"]
    if command == "predict":
        # Only checksum-verified worker-produced artifacts reach this process. Never accept uploads.
        model = pickle.loads(base64.b64decode(request["artifact"], validate=True))
        if model["environment"] != env:
            raise ValueError("Artifact environment mismatch")
        return {"predictions": model["pipeline"].predict(x).tolist()}
    y = np.array([row[spec["dataset"]["schema"]["target"]] for row in rows])
    if command == "prepare":
        split = spec["split"]
        indices = np.arange(len(rows))
        train, holdout = train_test_split(indices, train_size=split["train"], random_state=spec["seed"],
                                          stratify=y if split["stratify"] else None)
        validation, test = train_test_split(holdout, train_size=split["validation"] / (split["validation"] + split["test"]),
                                            random_state=spec["seed"], stratify=y[holdout] if split["stratify"] else None)
        return {"indices": {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}, "environment": env}
    if command == "train":
        steps = []
        for index, step in enumerate(spec["pipeline"]["definition"]["steps"]):
            if step["operation"] == "impute_median":
                transform = SimpleImputer(strategy="median", keep_empty_features=True)
            elif step["operation"] == "standard_scale":
                transform = StandardScaler()
            else:
                raise ValueError("Unsupported preprocessing operation")
            steps.append((f"transform_{index}", transform))
        algorithms = {"logistic_regression": LogisticRegression, "ridge": Ridge,
                      "random_forest_classifier": RandomForestClassifier, "random_forest_regressor": RandomForestRegressor}
        params = {**spec["hyperparameters"], "random_state": spec["seed"]}
        if spec["algorithm"].startswith("random_forest"):
            params["n_jobs"] = 1
        estimator = algorithms[spec["algorithm"]](**params)
        model = Pipeline([*steps, ("estimator", estimator)])
        train = request["prepared"]["indices"]["train"]
        # Fit all preprocessing only on training rows; validation/test never fit transforms.
        model.fit(x[train], y[train])
        payload = pickle.dumps({"pipeline": model, "environment": env}, protocol=5)
        return {"artifact": base64.b64encode(payload).decode(), "format": "sklearn-pickle-v1", "environment": env,
                "resolvedHyperparameters": estimator.get_params(deep=False)}
    if command == "evaluate":
        model = pickle.loads(base64.b64decode(request["artifact"], validate=True))
        if model["environment"] != env:
            raise ValueError("Artifact environment mismatch")
        classification = spec["algorithm"] in ("logistic_regression", "random_forest_classifier")
        metrics = []
        for partition, indices in request["prepared"]["indices"].items():
            prediction = model["pipeline"].predict(x[indices])
            measures = [("accuracy", accuracy_score(y[indices], prediction), "higher")] if classification else [
                ("mae", mean_absolute_error(y[indices], prediction), "lower"),
                ("mse", mean_squared_error(y[indices], prediction), "lower")]
            metrics.extend({"name": name, "partition": partition, "value": float(value), "direction": direction}
                           for name, value, direction in measures)
        return {"metrics": metrics}
    raise ValueError("Unknown backend command")


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        # Do not echo dataset values, labels, paths or request bodies through API errors.
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
