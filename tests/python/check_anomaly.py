"""Anomaly detection in the scikit-learn runner. Prints one JSON object of named results (booleans) plus diagnostics.

argv[1]: the TypeScript catalog defaults of the anomaly models, {id: {param: default}}, compared with scikit-learn's.
"""
import base64
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory

import numpy as np

from support import ml_backend as sklearn_runner, snapshot

common = sklearn_runner.common
results, details = {}, {}
MODELS = ("isolation_forest", "local_outlier_factor", "robust_covariance")
ts_defaults = json.loads(sys.argv[1])


def rows(n=240, outliers=16, seed=0, labelled=True):
    """Gaussian inliers plus planted far outliers; `outlier` (boolean) is the label, never a feature."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        planted = i % (n // outliers) == 0
        a, b = (rng.uniform(6, 9) * rng.choice([-1, 1]), rng.uniform(6, 9) * rng.choice([-1, 1])) if planted else (rng.randn(), rng.randn())
        row = {"a": float(a), "b": float(b)}
        if labelled:
            row["outlier"] = bool(planted)
        out.append(row)
    return out


def spec(algorithm, labelled=True, hyperparameters=None, metrics=None, cv=None):
    columns = [("a", "number"), ("b", "number")] + ([("outlier", "boolean")] if labelled else [])
    s = snapshot(algorithm, columns, "outlier" if labelled else None, ["a", "b"], [("standard_scale", {})], hyperparameters=hyperparameters or {},
                 evaluation={**({"metrics": metrics} if metrics else {}), **({"cv": cv} if cv else {})})
    if not labelled:
        del s["dataset"]["schema"]["target"]
    return s


def run(command, s, data, **extra):
    return sklearn_runner.execute({"command": command, "snapshot": s, "rows": data, **extra})


def lifecycle(s, data):
    prepared = run("prepare", s, data)
    trained = run("train", s, data, prepared=prepared)
    return prepared, trained, run("evaluate", s, data, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])


def value(evaluated, name, partition):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)


def refuses(fn, kind):
    try:
        fn()
    except kind:
        return True
    except Exception:
        return False
    return False


labelled = rows()
for model in MODELS:
    metrics = ["average_precision", "roc_auc", "anomaly_precision", "anomaly_recall", "anomaly_f1"]
    s = spec(model, metrics=metrics, hyperparameters={"contamination": 0.1})
    prepared, trained, evaluated = lifecycle(s, labelled)
    results[f"{model}_finds_planted_outliers"] = min(value(evaluated, "average_precision", p) for p in ("train", "test")) > 0.8
    details[f"{model}_test_average_precision"] = value(evaluated, "average_precision", "test")
    predictions = run("predict", s, labelled, artifact=trained["artifact"])["predictions"]
    scores = np.array([p["score"] for p in predictions])
    planted = np.array([r["outlier"] for r in labelled])
    results[f"{model}_scores_rise_with_abnormality"] = scores[planted].mean() > scores[~planted].mean() and all(set(p) == {"score", "anomaly"} and isinstance(p["anomaly"], bool) for p in predictions)
    payload = pickle.loads(base64.b64decode(trained["artifact"]))
    train_scores = np.array(payload["anomaly"]["trainScores"])
    flagged = float(np.mean(train_scores > payload["anomaly"]["threshold"]))
    results[f"{model}_training_flag_rate_matches_contamination"] = abs(flagged - 0.1) <= 1.0 / len(train_scores) + 1e-12
    details[f"{model}_training_flag_rate"] = flagged
    # Labels never fit anything: the same rows without labels, or with flipped labels, give the same scores.
    unlabelled_rows = [{"a": r["a"], "b": r["b"]} for r in labelled]
    flipped_rows = [{**r, "outlier": not r["outlier"]} for r in labelled]
    u = spec(model, labelled=False, hyperparameters={"contamination": 0.1})
    unlabelled_artifact = run("train", u, unlabelled_rows, prepared=run("prepare", u, unlabelled_rows))["artifact"]
    flipped_artifact = run("train", s, flipped_rows, prepared=prepared)["artifact"]
    probe = labelled[:40]
    same = lambda artifact: [p["score"] for p in run("predict", s, probe, artifact=artifact)["predictions"]]
    results[f"{model}_labels_never_change_the_fit"] = same(flipped_artifact) == same(trained["artifact"]) and np.allclose(same(unlabelled_artifact), same(trained["artifact"]))
    results[f"{model}_is_deterministic"] = same(run("train", s, labelled, prepared=prepared)["artifact"]) == same(trained["artifact"])

# Unlabelled data reports only the label-free diagnostic; labelled metrics are refused without labels.
u = spec("isolation_forest", labelled=False)
_, _, unlabelled_eval = lifecycle(u, [{"a": r["a"], "b": r["b"]} for r in labelled])
results["unlabelled_evaluation_reports_only_the_diagnostic"] = {m["name"] for m in unlabelled_eval["metrics"]} == {"threshold_calibration_error"}
results["labelled_metric_refused_without_labels"] = refuses(lambda: lifecycle(spec("isolation_forest", labelled=False, metrics=["average_precision"]), [{"a": r["a"], "b": r["b"]} for r in labelled]), ValueError)
results["single_class_partition_refused_as_undefined"] = refuses(lambda: lifecycle(spec("isolation_forest", metrics=["average_precision"]), [{**r, "outlier": False} for r in labelled]), common.RunnerError)
results["cross_validation_refused"] = refuses(lambda: run("prepare", spec("isolation_forest", cv={"strategy": "kfold", "folds": 3}), labelled), ValueError)

# Artifact schemas: anomaly artifacts are schema 2; a schema-1-only adapter refuses them; schema 1 still loads.
anomaly_env = lifecycle(spec("robust_covariance"), labelled)[1]["environment"]
results["anomaly_artifacts_are_schema_2"] = anomaly_env["artifact_schema"] == "2"
results["schema_1_only_adapter_refuses_schema_2"] = refuses(lambda: common.check_compatible(anomaly_env, sklearn_runner.environment(), {**sklearn_runner.CONTRACT, "schemas": {1}}), common.IncompatibleArtifact)
ridge = snapshot("ridge", [("a", "number"), ("y", "number")], "y", ["a"])
ridge_rows = [{"a": float(i), "y": 2.0 * i} for i in range(60)]
ridge_prepared = run("prepare", ridge, ridge_rows)
ridge_trained = run("train", ridge, ridge_rows, prepared=ridge_prepared)
results["regression_artifacts_stay_schema_1_and_load"] = ridge_trained["environment"]["artifact_schema"] == "1" and len(run("predict", ridge, ridge_rows[:3], artifact=ridge_trained["artifact"])["predictions"]) == 3
results["regression_artifact_refused_as_anomaly_model"] = refuses(lambda: run("predict", spec("isolation_forest"), labelled[:3], artifact=ridge_trained["artifact"]), common.IncompatibleArtifact)

# Catalog defaults equal scikit-learn's own, except the deliberate numeric contamination and max_samples null ("auto").
parity = True
for model, defaults in ts_defaults.items():
    cls = sklearn_runner.MODELS[model][0]
    native = cls().get_params()
    for name, default in defaults.items():
        if name == "contamination":
            parity = parity and default == 0.1
        elif name == "max_samples":
            parity = parity and default is None and native[name] == "auto"
        else:
            parity = parity and native[name] == default
results["catalog_defaults_match_scikit_learn_except_documented"] = parity and set(ts_defaults) == set(MODELS)

print(json.dumps({"results": {k: bool(v) for k, v in results.items()}, "details": details}))
