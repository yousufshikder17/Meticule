"""Explicit data-leakage checks against the real adapter. Prints one JSON object of named boolean results."""
import base64
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import emit, ml_backend, run, snapshot  # noqa: E402

N = 100
COLUMNS = [("x", "number"), ("y", "number")]
STEPS = [("impute_median", {}), ("standard_scale", {})]
results = {}


def rows_with(outlier_outside=None):
    """x equals the row index, so every fit call reveals exactly which rows it saw."""
    rows = [{"x": float(i), "y": 2.0 * i + 1.0} for i in range(N)]
    for i in outlier_outside or []:
        rows[i] = {"x": 1e9, "y": 1e9}
    return rows


def spec(cv=True, seed=5):
    return snapshot("ridge", COLUMNS, "y", ["x"], STEPS, hyperparameters={"alpha": 0.01}, seed=seed,
                    evaluation={"metrics": ["rmse"], **({"cv": {"strategy": "kfold", "folds": 4}} if cv else {})})


class Recorder:
    """Captures the first-column values handed to every preprocessing fit, without changing behaviour."""

    def __init__(self):
        self.fits = {"scaler": [], "imputer": []}
        self._scaler, self._imputer = StandardScaler.fit, SimpleImputer.fit

    def __enter__(self):
        def scaler_fit(this, X, y=None, **kw):
            self.fits["scaler"].append(np.array(X, dtype=float)[:, 0].copy())
            return self._scaler(this, X, y, **kw)

        def imputer_fit(this, X, y=None, **kw):
            self.fits["imputer"].append(np.array(X, dtype=float)[:, 0].copy())
            return self._imputer(this, X, y, **kw)

        StandardScaler.fit, SimpleImputer.fit = scaler_fit, imputer_fit
        return self

    def __exit__(self, *exc):
        StandardScaler.fit, SimpleImputer.fit = self._scaler, self._imputer

    def clear(self):
        self.fits = {"scaler": [], "imputer": []}


def as_set(values):
    return {int(v) for v in values if not np.isnan(v)}


def pipeline_of(model):
    return pickle.loads(base64.b64decode(model["artifact"]))["pipeline"]


def train(s, rows, prepared):
    return run("train", s, rows, prepared=prepared, expectedEnvironment=prepared["environment"])


rows = rows_with()
s = spec()
prepared = run("prepare", s, rows)
train_idx, validation_idx, test_idx = (set(prepared["indices"][k]) for k in ("train", "validation", "test"))
pool = np.array(prepared["indices"]["train"])
held_out = validation_idx | test_idx

# 1. The final fit sees exactly the training partition: no validation or test row reaches scaler or imputer.
with Recorder() as rec:
    model = train(s, rows, prepared)
results["final_fit_uses_training_rows_only"] = all(as_set(f) == train_idx for f in rec.fits["scaler"] + rec.fits["imputer"]) and len(rec.fits["scaler"]) == 1
results["no_held_out_row_in_final_fit"] = all(not (as_set(f) & held_out) for f in rec.fits["scaler"] + rec.fits["imputer"])
fitted = pipeline_of(model)[0].named_transformers_["numeric"]
results["scaler_mean_is_training_mean"] = bool(np.isclose(fitted[1].mean_[0], np.mean(sorted(train_idx))))
results["imputer_median_is_training_median"] = bool(np.isclose(fitted[0].statistics_[0], np.median(sorted(train_idx))))

# 2. Perturbing validation/test values cannot change anything that was fitted (preprocessing or estimator).
perturbed = rows_with(outlier_outside=sorted(held_out))
other = train(s, perturbed, prepared)
a, b = pipeline_of(model), pipeline_of(other)
results["held_out_values_do_not_change_fitted_state"] = bool(
    np.allclose(a[0].named_transformers_["numeric"][1].mean_, b[0].named_transformers_["numeric"][1].mean_)
    and np.allclose(a[0].named_transformers_["numeric"][1].scale_, b[0].named_transformers_["numeric"][1].scale_)
    and np.allclose(a[-1].coef_, b[-1].coef_) and np.isclose(a[-1].intercept_, b[-1].intercept_))

# 3. Cross-validation refits preprocessing inside every fold, on that fold's training rows only.
folds = prepared["folds"]
results["folds_partition_training_rows"] = sorted(sum(folds, [])) == sorted(train_idx) and not (set(sum(folds, [])) & held_out)
with Recorder() as rec:
    evaluation = run("evaluate", s, rows, prepared=prepared, artifact=model["artifact"], expectedEnvironment=model["environment"])
expected = [set(pool.tolist()) - set(f) for f in folds]
results["cv_fits_once_per_fold"] = len(rec.fits["scaler"]) == len(folds) and len(rec.fits["imputer"]) == len(folds)
results["cv_fold_fit_rows_exclude_own_validation_fold"] = all(as_set(f) == e for f, e in zip(rec.fits["scaler"], expected))
results["cv_fit_never_sees_validation_or_test_partition"] = all(not (as_set(f) & held_out) for f in rec.fits["scaler"] + rec.fits["imputer"])
fold_means = [float(np.mean(f)) for f in rec.fits["scaler"]]
results["cv_fold_statistics_are_independent"] = len(set(round(m, 9) for m in fold_means)) > 1 and all(np.isclose(m, np.mean(sorted(e))) for m, e in zip(fold_means, expected))

# 4. CV scores are blind to validation/test rows: corrupting them leaves every cv metric unchanged.
corrupt = run("evaluate", s, perturbed, prepared=prepared, artifact=other["artifact"], expectedEnvironment=other["environment"])
cv = lambda e: sorted((m["name"], m["partition"], m.get("fold", -1), round(m["value"], 12)) for m in e["metrics"] if m["partition"] in ("cv", "cv_fold"))
results["cv_scores_ignore_held_out_rows"] = cv(evaluation) == cv(corrupt) and len(cv(evaluation)) > 0
results["holdout_scores_do_see_held_out_rows"] = any(abs(x["value"] - y["value"]) > 1 for x, y in zip(evaluation["metrics"], corrupt["metrics"]) if x["partition"] == "test")

# 5. The target never becomes a feature: the design matrix has exactly the declared features.
mixed = snapshot("logistic_regression", [("a", "number"), ("kind", "string"), ("label", "string")], "label", ["a", "kind"], [("one_hot_encode", {})])
matrix = ml_backend.to_matrix([{"a": 1.0, "kind": "r", "label": "yes"}], ["a", "kind"], ["number", "string"])
results["design_matrix_excludes_target"] = matrix.shape == (1, 2) and "yes" not in matrix.tolist()[0]

# 6. Inference replays the persisted transformation: a null is imputed with the TRAINING median and scaled with
#    training statistics, regardless of what else is in the request batch.
median = float(np.median(sorted(train_idx)))
solo = run("predict", s, [{"x": None}], artifact=model["artifact"], expectedEnvironment=model["environment"])["predictions"][0]
batch = run("predict", s, [{"x": None}, {"x": 5000.0}, {"x": 7000.0}], artifact=model["artifact"], expectedEnvironment=model["environment"])["predictions"]
results["inference_imputes_with_training_median"] = abs(solo - (2 * median + 1)) < 0.05
results["inference_is_independent_of_batch_composition"] = abs(batch[0] - solo) < 1e-9

# 7. Same seed, same split and folds; a different seed changes them (reproducibility, not coincidence).
again = run("prepare", s, rows)
results["same_seed_same_split_and_folds"] = again["indices"] == prepared["indices"] and again["folds"] == prepared["folds"]
results["different_seed_different_split"] = run("prepare", spec(seed=6), rows)["indices"]["train"] != prepared["indices"]["train"]
emit(results)
