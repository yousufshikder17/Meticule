"""XGBoost CPU backend over the Meticule runner protocol: native xgb.train on a DMatrix, the booster's own JSON model format.

Everything the two gradient-boosting runners share (bounds, targets, early stopping, the artifact envelope, evaluation) is
in gbdt_common. One fit per attempt; framework-native early stopping only, never Meticule's training policy.
"""
import sys

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
import xgboost as xgb

import gbdt_common
import ml_runtime_common as common
import neural_common

BACKEND = "xgboost"
ARTIFACT_FORMAT = "xgboost-json-v1"
PROTOCOL = "meticule-xgboost-v1"
RUNTIME_KEYS = ("xgboost", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
PARAMETERS = {"max_depth": ("int", 1, 16, 6), "min_child_weight": ("float", 0.0, 1000.0, 1.0), "reg_lambda": ("float", 0.0, 1000.0, 1.0)}


def environment():
    return common.base_environment([__file__, common.__file__, neural_common.__file__, gbdt_common.__file__], BACKEND, ARTIFACT_FORMAT,
                                   gbdt_common.ARTIFACT_SCHEMA, PROTOCOL,
                                   {"xgboost": xgb.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                    "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def objective(task, classes):
    if task == "regression":
        return {"objective": "reg:squarederror", "eval_metric": "rmse"}
    if classes == 2:
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {"objective": "multi:softprob", "num_class": classes, "eval_metric": "mlogloss"}


def fit(task, hp, x, y, classes, monitor, seed):
    """Deterministic: fixed seed, one thread, histogram trees. Returns the booster and the number of trees to use."""
    params = {**objective(task, classes), "eta": hp["learning_rate"], "max_depth": hp["max_depth"], "min_child_weight": hp["min_child_weight"],
              "subsample": hp["subsample"], "colsample_bytree": hp["colsample_bytree"], "lambda": hp["reg_lambda"], "alpha": hp["reg_alpha"],
              "seed": int(seed), "nthread": 1, "tree_method": "hist", "verbosity": 0}
    train = xgb.DMatrix(x, label=y, missing=np.nan)
    if monitor is None:
        return xgb.train(params, train, num_boost_round=hp["n_estimators"], verbose_eval=False), hp["n_estimators"]
    validation = xgb.DMatrix(monitor[0], label=monitor[1], missing=np.nan)
    booster = xgb.train(params, train, num_boost_round=hp["n_estimators"], evals=[(validation, "validation")],
                        early_stopping_rounds=hp["early_stopping_rounds"], verbose_eval=False)
    return booster, booster.best_iteration + 1  # best_iteration is a 0-based index; record the number of trees used


def raw_predict(booster, x, trees, task, classes):
    return booster.predict(xgb.DMatrix(x, missing=np.nan), iteration_range=(0, trees))


def dump(booster):
    return booster.save_raw("json").decode("utf-8")


def load(text):
    booster = xgb.Booster()
    booster.load_model(bytearray(text.encode("utf-8")))
    return booster


RUNNER = gbdt_common.Runner(BACKEND, ARTIFACT_FORMAT, PROTOCOL, RUNTIME_KEYS, PARAMETERS, environment, fit, raw_predict, dump, load)
MODELS = gbdt_common.MODELS[BACKEND]
execute = gbdt_common.make_execute(RUNNER)

if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
