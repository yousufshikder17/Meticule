"""LightGBM CPU backend over the Meticule runner protocol: native lgb.train on a Dataset, the booster's own text model format.

Everything the two gradient-boosting runners share (bounds, targets, early stopping, the artifact envelope, evaluation) is
in gbdt_common. One fit per attempt; framework-native early stopping only, never Meticule's training policy.
pandas is a runtime key from this milestone on: the forecasting model added later builds its features with it.
"""
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas
import scipy
import sklearn
import threadpoolctl

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
    return common.base_environment([__file__, common.__file__, neural_common.__file__, gbdt_common.__file__], BACKEND, ARTIFACT_FORMAT,
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
MODELS = gbdt_common.MODELS[BACKEND]
execute = gbdt_common.make_execute(RUNNER)

if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
