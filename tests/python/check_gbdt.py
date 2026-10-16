"""Real XGBoost and LightGBM backend checks. Prints one JSON object of named results (booleans) plus diagnostics.

Both runners run in-process with fixed seeds and tiny local fixtures, plus one real subprocess per runner that proves
early-stopping training through `serve` returns clean protocol JSON. Processor integration is in tests/integration/ml-gbdt.test.ts.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(PYTHON_DIR))
import lightgbm_runner  # noqa: E402
import ml_runtime_common as common  # noqa: E402
import neural_common  # noqa: E402
import xgboost_runner  # noqa: E402

results, details = {}, {}
UUID = "00000000-0000-4000-8000-000000000000"


def spec(backend, algorithm, columns, target, features, steps, hyperparameters, metrics, cv=None, seed=7, stratify=False):
    return {"algorithm": algorithm, "backend": backend, "seed": seed, "hyperparameters": hyperparameters,
            "evaluation": {"metrics": metrics, **({"cv": cv} if cv else {})},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": stratify},
            "dataset": {"schema": {"columns": [{"name": n, "type": t} for n, t in columns], "target": target}},
            "pipeline": {"definition": {"features": features, "steps": [{"operation": o, "parameters": {}} for o in steps]}}}


def refuses(fn, kind, code=None):
    try:
        fn()
    except kind as error:
        return code is None or getattr(error, "code", None) == code
    except Exception:
        return False
    return False


rng = np.random.RandomState(0)
binary_rows = [{"a": float(rng.randn() + (2 if i % 2 else -2)) if i % 9 else None, "b": float(rng.randn()), "kind": ["red", "blue", None][i % 3],
                "label": "yes" if i % 2 else "no"} for i in range(240)]
binary_columns = [("a", "number"), ("b", "number"), ("kind", "string"), ("label", "string")]
multi_rows = [{"x": float(rng.randn() + 3 * (i % 3)), "y": float(rng.randn()), "label": f"c{i % 3}"} for i in range(240)]
reg_rows = [{"u": float(u), "v": float(v), "t": float(3 * u - 2 * v + 0.1 * rng.randn())} for u, v in rng.rand(240, 2) * 4]
# A real signal with noisy labels: trees learn it, then overfit, so the validation loss has an interior minimum.
noisy_rows = []
for i in range(300):
    a_value, b_value = float(rng.randn()), float(rng.randn())
    noisy_rows.append({"a": a_value, "b": b_value, "label": "yes" if a_value + 0.8 * rng.randn() > 0 else "no"})

# Settings under which each framework learns the signal and then overfits, so the stopping point is interior.
STOPPING = {"xgb": {"learning_rate": 0.3, "early_stopping_rounds": 5}, "lgb": {"learning_rate": 0.1, "min_child_samples": 20, "early_stopping_rounds": 10}}

for prefix, runner, extra in (("xgb", xgboost_runner, {}), ("lgb", lightgbm_runner, {"min_child_samples": 2})):
    backend = runner.BACKEND
    env = runner.environment()
    clf, reg = f"{backend}_classifier", f"{backend}_regressor"

    def run(command, s, rows, **more):
        return runner.execute({"command": command, "snapshot": s, "rows": rows, **more})

    def lifecycle(s, rows):
        prepared = run("prepare", s, rows)
        trained = run("train", s, rows, prepared=prepared)
        evaluated = run("evaluate", s, rows, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])
        return prepared, trained, evaluated

    def metric(evaluated, name, partition="test"):
        return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)

    binary = spec(backend, clf, binary_columns, "label", ["a", "b", "kind"], ["impute_most_frequent", "one_hot_encode"],
                  {"n_estimators": 60, **extra}, ["accuracy", "roc_auc", "log_loss"], stratify=True)
    prepared, trained, evaluated = lifecycle(binary, binary_rows)
    results[f"{prefix}_binary_learns_with_missing_numbers_passed_through"] = metric(evaluated, "accuracy") > 0.9 and metric(evaluated, "roc_auc") > 0.9
    results[f"{prefix}_records_runtime_and_format"] = (trained["format"] == runner.ARTIFACT_FORMAT and trained["environment"][backend] == env[backend]
                                                     and (backend != "lightgbm" or "pandas" in trained["environment"]))
    envelope = json.loads(base64.b64decode(trained["artifact"]))
    results[f"{prefix}_artifact_is_a_json_envelope_not_a_pickle"] = set(envelope) == {"metadata", "model"} and isinstance(envelope["model"], str)
    bundle = runner.gbdt_common.load_bundle(runner.RUNNER, trained["artifact"], env)
    proba = runner.gbdt_common.predict_bundle(runner.RUNNER, bundle, binary_rows[:30])["proba"]
    results[f"{prefix}_binary_probabilities_are_distributions"] = proba.shape == (30, 2) and np.allclose(proba.sum(axis=1), 1.0)

    multi = spec(backend, clf, [("x", "number"), ("y", "number"), ("label", "string")], "label", ["x", "y"], [], {"n_estimators": 60, **extra}, ["accuracy", "log_loss"])
    multi_prepared, multi_trained, multi_eval = lifecycle(multi, multi_rows)
    multi_proba = runner.gbdt_common.predict_bundle(runner.RUNNER, runner.gbdt_common.load_bundle(runner.RUNNER, multi_trained["artifact"], env), multi_rows[:10])["proba"]
    results[f"{prefix}_multiclass_learns_with_three_column_probabilities"] = metric(multi_eval, "accuracy") > 0.85 and multi_proba.shape == (10, 3) and np.allclose(multi_proba.sum(axis=1), 1.0)

    regression = spec(backend, reg, [("u", "number"), ("v", "number"), ("t", "number")], "t", ["u", "v"], [], {"n_estimators": 200, **extra}, ["r2", "mae"])
    results[f"{prefix}_regression_fits"] = metric(lifecycle(regression, reg_rows)[2], "r2") > 0.9

    again = run("train", binary, binary_rows, prepared=prepared)
    results[f"{prefix}_same_seed_identical_model"] = json.loads(base64.b64decode(again["artifact"]))["model"] == envelope["model"]

    # Early stopping: the validation partition is the monitoring set only.
    stopping = spec(backend, clf, [("a", "number"), ("b", "number"), ("label", "string")], "label", ["a", "b"], ["standard_scale"],
                    {"n_estimators": 400, **STOPPING[prefix]}, ["accuracy"])
    stop_prepared = run("prepare", stopping, noisy_rows)
    stopped = run("train", stopping, noisy_rows, prepared=stop_prepared)
    best = stopped["resolvedHyperparameters"]["best_iteration"]
    stop_meta = json.loads(base64.b64decode(stopped["artifact"]))["metadata"]
    results[f"{prefix}_early_stopping_stops_and_persists_best_iteration"] = 1 <= best < 400 and stop_meta["bestIteration"] == best and stopped["performance"]["bestIteration"] == best
    details[f"{prefix}_best_iteration"] = best
    flipped = [dict(r) for r in noisy_rows]
    for i in stop_prepared["indices"]["validation"]:
        flipped[i]["label"] = "no" if flipped[i]["label"] == "yes" else "yes"
    results[f"{prefix}_validation_labels_steer_only_the_stopping_point"] = run("train", stopping, flipped, prepared=stop_prepared)["resolvedHyperparameters"]["best_iteration"] != best
    shifted = [dict(r) for r in noisy_rows]
    for i in stop_prepared["indices"]["validation"]:
        shifted[i]["a"] = 1000.0
    shifted_meta = json.loads(base64.b64decode(run("train", stopping, shifted, prepared=stop_prepared)["artifact"]))["metadata"]
    results[f"{prefix}_validation_rows_never_fit_preprocessing"] = shifted_meta["preprocessing"] == stop_meta["preprocessing"]
    no_stop = {**stopping, "hyperparameters": {**stopping["hyperparameters"], "early_stopping_rounds": 0, "n_estimators": 50}}
    results[f"{prefix}_without_early_stopping_validation_rows_are_unused"] = (
        json.loads(base64.b64decode(run("train", no_stop, flipped, prepared=stop_prepared)["artifact"]))["model"]
        == json.loads(base64.b64decode(run("train", no_stop, noisy_rows, prepared=stop_prepared)["artifact"]))["model"])
    stop_bundle = runner.gbdt_common.load_bundle(runner.RUNNER, stopped["artifact"], env)
    x = runner.gbdt_common.matrix(stop_bundle["preprocessing"], noisy_rows[:40])
    # XGBoost keeps every tree and prediction is cut at the recorded count; LightGBM saves only the trees up to its best
    # iteration. Either way the served model is exactly the best-iteration model.
    trees = stop_bundle["model"].num_boosted_rounds() if prefix == "xgb" else stop_bundle["model"].current_iteration()
    at_best = runner.raw_predict(stop_bundle["model"], x, best, "classification", 2)
    results[f"{prefix}_predict_uses_the_best_iteration"] = (
        np.allclose(runner.gbdt_common.predict_bundle(runner.RUNNER, stop_bundle, noisy_rows[:40])["proba"][:, 1], at_best)
        and (trees == best or (trees > best and not np.allclose(at_best, runner.raw_predict(stop_bundle["model"], x, trees, "classification", 2)))))
    details[f"{prefix}_stored_trees"] = trees

    # Cross-validation: a fresh model per fold; early stopping inside folds is refused, not silently disabled.
    cv = {**binary, "evaluation": {"metrics": ["accuracy"], "cv": {"strategy": "kfold", "folds": 3}}}
    cv_prepared = run("prepare", cv, binary_rows)
    cv_eval = run("evaluate", cv, binary_rows, prepared=cv_prepared, artifact=run("train", cv, binary_rows, prepared=cv_prepared)["artifact"])
    results[f"{prefix}_cv_fresh_model_per_fold"] = cv_eval["performance"]["cvModelsBuilt"] == 3.0 and sum(m["partition"] == "cv_fold" for m in cv_eval["metrics"]) == 3
    cv_stopping = {**cv, "hyperparameters": {**cv["hyperparameters"], "early_stopping_rounds": 5}}
    results[f"{prefix}_early_stopping_with_cv_refused"] = refuses(lambda: run("prepare", cv_stopping, binary_rows), ValueError)

    # Safe loading and refusals.
    results[f"{prefix}_corrupt_artifact_refused"] = refuses(lambda: run("predict", binary, binary_rows[:2], artifact=base64.b64encode(b"\x80garbage").decode()), common.IncompatibleArtifact)
    forged = base64.b64encode(json.dumps({**envelope, "extra": 1}).encode()).decode()
    results[f"{prefix}_unexpected_envelope_keys_refused"] = refuses(lambda: run("predict", binary, binary_rows[:2], artifact=forged), common.IncompatibleArtifact)
    broken = base64.b64encode(json.dumps({**envelope, "model": "not a model"}).encode()).decode()
    results[f"{prefix}_unparseable_model_text_refused"] = refuses(lambda: run("predict", binary, binary_rows[:2], artifact=broken), common.IncompatibleArtifact)
    results[f"{prefix}_foreign_runtime_refused"] = refuses(lambda: run("predict", binary, binary_rows[:2], artifact=trained["artifact"], expectedEnvironment={**env, backend: "0.0.1"}), common.IncompatibleArtifact)
    results[f"{prefix}_out_of_range_hyperparameter_refused"] = refuses(lambda: run("train", {**binary, "hyperparameters": {"n_estimators": 5000}}, binary_rows, prepared=prepared), ValueError)
    results[f"{prefix}_unknown_hyperparameter_refused"] = refuses(lambda: run("train", {**binary, "hyperparameters": {"booster": "dart"}}, binary_rows, prepared=prepared), ValueError)
    results[f"{prefix}_cuda_refused"] = refuses(lambda: run("prepare", {**binary, "training": {"device": "cuda", "precision": "fp32"}}, binary_rows), common.RunnerError, "cuda_unsupported")

    # The real runner process: early-stopping training through serve returns clean protocol JSON on stdout.
    request = {"contract": common.ML_BACKEND_CONTRACT, "command": "train", "snapshot": stopping, "rows": noisy_rows, "prepared": stop_prepared}
    # Launched exactly as src/ml/runner-process.ts launches it: `python <script>`, so the script directory is importable.
    process = subprocess.run([sys.executable, str(PYTHON_DIR / f"{backend}_runner.py")], input=json.dumps(request), capture_output=True, text=True, timeout=300)
    try:
        response = json.loads(process.stdout)
        results[f"{prefix}_served_training_returns_clean_json"] = process.returncode == 0 and response["contract"] == common.ML_BACKEND_CONTRACT and response["resolvedHyperparameters"]["best_iteration"] == best
    except Exception:
        results[f"{prefix}_served_training_returns_clean_json"] = False

# B8: the GBDT NaN pass-through is opt-in; the neural default still refuses a missing number without an imputer.
state = neural_common.fit_preprocessing(spec("x", "mlp_classifier", [("a", "number"), ("label", "string")], "label", ["a"], [], {}, []), [{"a": 1.0}, {"a": 2.0}], [0, 1])
results["neural_default_still_refuses_missing_numbers"] = refuses(lambda: neural_common.transform(state, [{"a": None}]), ValueError)
results["nan_pass_through_is_opt_in"] = bool(np.isnan(neural_common.transform(state, [{"a": None}], missing="nan")[0, 0]))

print(json.dumps({"results": results, "details": details}))
