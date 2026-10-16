"""LightGBM CPU backend over the Meticule runner protocol: native lgb.train on a Dataset, the booster's own text model format.

`lightgbm_forecaster` (task forecasting) fits a LightGBM regressor on forecasting_common's lag and rolling-mean features of the
training block and forecasts recursively; the series rules, leakage policy and evaluation are forecasting_common's.

Everything the two gradient-boosting runners share (bounds, targets, early stopping, the artifact envelope, evaluation) is
in gbdt_common. One fit per attempt; framework-native early stopping only, never Meticule's training policy.
pandas is a runtime key: lightgbm_forecaster builds its features with it.
"""
import base64
import binascii
import json
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas
import scipy
import sklearn
import threadpoolctl

import forecasting_common
import gbdt_common
import ml_runtime_common as common
import neural_common

BACKEND = "lightgbm"
ARTIFACT_FORMAT = "lightgbm-text-v1"
PROTOCOL = "meticule-lightgbm-v1"
RUNTIME_KEYS = ("lightgbm", "pandas", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
PARAMETERS = {"max_depth": ("int_nullable", 1, 32, None), "num_leaves": ("int", 2, 1024, 31), "min_child_samples": ("int", 1, 1000, 20),
              "reg_lambda": ("float", 0.0, 1000.0, 0.0)}


def environment():
    return common.base_environment([__file__, common.__file__, neural_common.__file__, gbdt_common.__file__, forecasting_common.__file__], BACKEND, ARTIFACT_FORMAT,
                                   gbdt_common.ARTIFACT_SCHEMA, PROTOCOL,
                                   {"lightgbm": lgb.__version__, "pandas": pandas.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                    "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def objective(task, classes):
    if task == "regression":
        return {"objective": "regression", "metric": "l2"}
    if classes == 2:
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "multiclass", "num_class": classes, "metric": "multi_logloss"}


def fit(task, hp, x, y, classes, monitor, seed):
    """Deterministic: fixed seeds, one thread, row-wise histograms. subsample < 1 bags every iteration (bagging_freq 1)."""
    params = {**objective(task, classes), "learning_rate": hp["learning_rate"], "max_depth": -1 if hp["max_depth"] is None else hp["max_depth"],
              "num_leaves": hp["num_leaves"], "min_data_in_leaf": hp["min_child_samples"], "bagging_fraction": hp["subsample"],
              "bagging_freq": 1 if hp["subsample"] < 1.0 else 0, "feature_fraction": hp["colsample_bytree"], "lambda_l2": hp["reg_lambda"],
              "lambda_l1": hp["reg_alpha"], "seed": int(seed), "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbosity": -1}
    train = lgb.Dataset(x, label=y, params={"verbosity": -1}, free_raw_data=True)
    if monitor is None:
        return lgb.train(params, train, num_boost_round=hp["n_estimators"]), hp["n_estimators"]
    validation = lgb.Dataset(monitor[0], label=monitor[1], reference=train, params={"verbosity": -1})
    booster = lgb.train(params, train, num_boost_round=hp["n_estimators"], valid_sets=[validation],
                        callbacks=[lgb.early_stopping(hp["early_stopping_rounds"], verbose=False)])
    return booster, booster.best_iteration or hp["n_estimators"]  # best_iteration already counts trees (1-based)


def raw_predict(booster, x, trees, task, classes):
    return booster.predict(x, num_iteration=trees)


def dump(booster):
    return booster.model_to_string()


def load(text):
    return lgb.Booster(model_str=text)


RUNNER = gbdt_common.Runner(BACKEND, ARTIFACT_FORMAT, PROTOCOL, RUNTIME_KEYS, PARAMETERS, environment, fit, raw_predict, dump, load)
FORECASTER = "lightgbm_forecaster"
MODELS = {**gbdt_common.MODELS[BACKEND], FORECASTER: "forecasting"}
TREE_PARAMETERS = tuple(name for name in RUNNER.parameters if name != "early_stopping_rounds")


class ForecastAdapter:
    """lightgbm_forecaster for forecasting_common: the same tree parameters (no native early stopping: the validation block
    belongs to evaluation), and the booster's text model in the same JSON envelope with forecasting metadata."""
    contract = RUNNER.contract
    environment = staticmethod(environment)

    @staticmethod
    def model_parameters(spec):
        return {}

    @staticmethod
    def settings(spec):
        supplied = spec.get("hyperparameters") or {}
        settings = forecasting_common.resolve_settings(supplied, "features", TREE_PARAMETERS)
        tree = gbdt_common.resolve_hyperparameters(RUNNER, {"hyperparameters": {k: v for k, v in supplied.items() if k in TREE_PARAMETERS}})
        tree.pop("early_stopping_rounds")
        settings["tree"] = tree
        return settings

    @staticmethod
    def fit(settings, matrix, values, seed):
        booster, _ = fit("regression", {**settings["tree"], "early_stopping_rounds": 0}, matrix, values, None, None, seed)
        return {"booster": booster, "predict": booster.predict}

    @staticmethod
    def serialize(metadata, model):
        envelope = {"metadata": {**metadata, "backend": BACKEND}, "model": model["booster"].model_to_string()}
        return base64.b64encode(json.dumps(envelope, allow_nan=False).encode("utf-8")).decode()

    @staticmethod
    def load(artifact, env):
        try:
            raw = base64.b64decode(artifact, validate=True)
            if len(raw) > gbdt_common.MAX_ARTIFACT_BYTES:
                raise ValueError
            envelope = json.loads(raw.decode("utf-8"))
            if not isinstance(envelope, dict) or set(envelope) != {"metadata", "model"} or not isinstance(envelope["metadata"], dict) or not isinstance(envelope["model"], str):
                raise ValueError
        except (ValueError, binascii.Error, UnicodeDecodeError):
            raise common.IncompatibleArtifact("unreadable artifact") from None
        metadata = envelope["metadata"]
        common.check_compatible(metadata.get("environment"), env, RUNNER.contract)
        if metadata.get("backend") != BACKEND or metadata.get("task") != "forecasting" or metadata.get("algorithm") != FORECASTER or not isinstance(metadata.get("settings"), dict):
            raise common.IncompatibleArtifact("malformed artifact metadata")
        try:
            booster = load(envelope["model"])
        except Exception:
            raise common.IncompatibleArtifact("unreadable model") from None
        return metadata, {"booster": booster, "predict": booster.predict}


_forecast = forecasting_common.make_execute(ForecastAdapter)
_boosting = gbdt_common.make_execute(RUNNER)


def execute(request):
    snapshot = request.get("snapshot") or {}
    return _forecast(request) if snapshot.get("algorithm") == FORECASTER else _boosting(request)


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
