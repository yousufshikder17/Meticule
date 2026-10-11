"""Real PyTorch backend checks. Prints one JSON object of named results (booleans plus numbers for diagnostics).

Runs the runner in-process with fixed seeds and tiny local fixtures. Subprocess-level determinism and the
TypeScript integration are covered separately by tests/integration/ml-pytorch.test.ts.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import pytorch_runner as pt  # noqa: E402
import ml_runtime_common as common  # noqa: E402

results, details = {}, {}
env = pt.environment()


def spec(algorithm, columns, target, features, steps, hyperparameters, metrics, cv=None, stratify=False, seed=7):
    evaluation = {"metrics": metrics, **({"cv": cv} if cv else {})}
    return {"algorithm": algorithm, "backend": "pytorch", "seed": seed, "hyperparameters": hyperparameters, "evaluation": evaluation,
            "split": {"id": "00000000-0000-4000-8000-000000000000", "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": stratify},
            "dataset": {"schema": {"columns": [{"name": n, "type": t} for n, t in columns], "target": target}},
            "pipeline": {"definition": {"features": features, "steps": [{"operation": o, "parameters": p} for o, p in steps]}}}


def run_all(s, rows):
    prepared = pt.execute({"command": "prepare", "snapshot": s, "rows": rows})
    trained = pt.execute({"command": "train", "snapshot": s, "rows": rows, "prepared": prepared, "expectedEnvironment": prepared["environment"]})
    evaluated = pt.execute({"command": "evaluate", "snapshot": s, "rows": rows, "prepared": prepared, "artifact": trained["artifact"], "expectedEnvironment": trained["environment"]})
    return prepared, trained, evaluated


def metric(evaluated, name, partition):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)


rng = np.random.RandomState(0)
# Binary: two separable Gaussian clusters, with a categorical feature and some missing values.
binary_rows = []
for i in range(160):
    lab = "yes" if i % 2 else "no"
    binary_rows.append({"a": float(rng.randn() + (2.5 if lab == "yes" else -2.5)) if i % 11 else None, "b": float(rng.randn()),
                        "kind": ["red", "blue", None][i % 3], "label": lab})
binary = spec("mlp_classifier", [("a", "number"), ("b", "number"), ("kind", "string"), ("label", "string")], "label", ["a", "b", "kind"],
              [("impute_median", {}), ("standard_scale", {}), ("impute_most_frequent", {}), ("one_hot_encode", {})],
              {"hiddenLayers": [16], "epochs": 40, "batchSize": 16, "learningRate": 0.01}, ["accuracy", "f1", "roc_auc", "log_loss"], stratify=True)
prepared, trained, evaluated = run_all(binary, binary_rows)
results["binary_learns_train_and_test"] = metric(evaluated, "accuracy", "train") > 0.9 and metric(evaluated, "accuracy", "test") > 0.85
results["binary_roc_auc_and_log_loss_defined"] = 0.9 < metric(evaluated, "roc_auc", "test") <= 1.0 and metric(evaluated, "log_loss", "test") < 0.5
details["binary_test_accuracy"] = metric(evaluated, "accuracy", "test")
results["binary_metric_directions"] = {m["name"]: m["direction"] for m in evaluated["metrics"] if m["partition"] == "test"} == {"accuracy": "higher", "f1": "higher", "roc_auc": "higher", "log_loss": "lower"}
results["training_records_final_losses_and_epochs"] = trained["performance"]["epochsCompleted"] == 40 and "finalTrainLoss" in trained["performance"] and "finalValidationLoss" in trained["performance"] and trained["performance"]["fitSeconds"] > 0

# Multiclass: three clusters.
multi_rows = []
for i in range(180):
    k = i % 3
    multi_rows.append({"x": float(rng.randn() * 0.4 + 3 * k), "y": float(rng.randn() * 0.4 - 2 * k), "label": f"class_{k}"})
multi = spec("mlp_classifier", [("x", "number"), ("y", "number"), ("label", "string")], "label", ["x", "y"],
             [("impute_mean", {}), ("min_max_scale", {})], {"hiddenLayers": [24, 12], "epochs": 60, "batchSize": 24, "learningRate": 0.01, "optimizer": "adamw"},
             ["accuracy", "roc_auc", "log_loss"], stratify=True)
_, _, multi_eval = run_all(multi, multi_rows)
results["multiclass_learns"] = metric(multi_eval, "accuracy", "test") > 0.9
results["multiclass_roc_auc_ovr_defined"] = metric(multi_eval, "roc_auc", "test") > 0.95
details["multiclass_test_accuracy"] = metric(multi_eval, "accuracy", "test")

# Regression: a linear signal with noise; target standardisation must be undone for metrics.
reg_rows = [{"a": float(v), "b": float(rng.randn()), "y": float(3 * v + 1 + rng.randn() * 0.1)} for v in rng.rand(160) * 10]
reg = spec("mlp_regressor", [("a", "number"), ("b", "number"), ("y", "number")], "y", ["a", "b"],
           [("impute_median", {}), ("standard_scale", {})], {"hiddenLayers": [32], "epochs": 80, "batchSize": 16, "learningRate": 0.01, "optimizer": "adam", "loss": "mse"},
           ["mae", "mse", "rmse", "r2"])
_, reg_trained, reg_eval = run_all(reg, reg_rows)
results["regression_learns_and_metrics_are_original_scale"] = metric(reg_eval, "r2", "test") > 0.9 and metric(reg_eval, "mae", "test") < 2.0
results["regression_rmse_is_sqrt_mse"] = abs(metric(reg_eval, "rmse", "test") - metric(reg_eval, "mse", "test") ** 0.5) < 1e-9
details["regression_test_r2"] = metric(reg_eval, "r2", "test")

# Preprocessing is fitted on training rows only: an extreme validation value must not move the imputer.
leak_rows = [{"a": float(i), "b": 0.0, "label": "lo" if i < 20 else "hi"} for i in range(40)]
leak_rows[30]["a"] = 1e6  # validation/test row outside the training partition
leak_rows[32]["a"] = None
leak_spec = spec("mlp_classifier", [("a", "number"), ("b", "number"), ("label", "string")], "label", ["a"],
                 [("impute_median", {}), ("standard_scale", {})], {"hiddenLayers": [4], "epochs": 1}, ["accuracy"])
train_idx = list(range(0, 24))
pp = pt.fit_preprocessing(leak_spec, leak_rows, train_idx)
train_known = [leak_rows[i]["a"] for i in train_idx if leak_rows[i]["a"] is not None]
imputer_fill = pp["columns"][0]["states"][0]["fill"]
results["imputer_fitted_on_training_rows_only"] = abs(imputer_fill - float(np.median(train_known))) < 1e-9 and imputer_fill < 100
scaler = pp["columns"][0]["states"][1]
results["scaler_fitted_on_training_rows_only"] = abs(scaler["center"] - float(np.mean(train_known))) < 1e-6 and scaler["spread"] < 100

# Folds: every fold trains a fresh network and fresh preprocessing; no weights are reused between folds.
cv_spec = spec("mlp_classifier", [("a", "number"), ("b", "number"), ("kind", "string"), ("label", "string")], "label", ["a", "b", "kind"],
               [("impute_median", {}), ("standard_scale", {}), ("impute_most_frequent", {}), ("one_hot_encode", {})],
               {"hiddenLayers": [8], "epochs": 5, "batchSize": 16}, ["accuracy"], cv={"strategy": "stratified_kfold", "folds": 3}, stratify=True)
cv_prepared, cv_trained, cv_eval = run_all(cv_spec, binary_rows)
results["cv_builds_one_fresh_model_per_fold"] = cv_eval["performance"]["cvModelsBuilt"] == 3 and len(cv_prepared["folds"]) == 3
results["cv_folds_partition_training_rows"] = sorted(sum(cv_prepared["folds"], [])) == sorted(cv_prepared["indices"]["train"])
results["cv_reports_per_fold_and_aggregate"] = len([m for m in cv_eval["metrics"] if m["partition"] == "cv_fold"]) == 3 and any(m["partition"] == "cv" and m["std"] >= 0 for m in cv_eval["metrics"])
fold_a = pt.train_network(cv_spec, binary_rows, cv_prepared["indices"]["train"][:40])
fold_b = pt.train_network(cv_spec, binary_rows, cv_prepared["indices"]["train"][40:80])
results["fresh_networks_do_not_share_weights"] = not torch.equal(fold_a["network"][0].weight, fold_b["network"][0].weight)

# Reproducibility in one process: identical seed and data give identical predictions and metrics.
_, trained_again, evaluated_again = run_all(binary, binary_rows)
pred_a = pt.execute({"command": "predict", "snapshot": binary, "rows": binary_rows[:20], "artifact": trained["artifact"], "expectedEnvironment": trained["environment"]})["predictions"]
pred_b = pt.execute({"command": "predict", "snapshot": binary, "rows": binary_rows[:20], "artifact": trained_again["artifact"], "expectedEnvironment": trained_again["environment"]})["predictions"]
results["same_seed_gives_identical_predictions"] = pred_a == pred_b
results["same_seed_gives_identical_metrics"] = all(abs(a["value"] - b["value"]) < 1e-9 for a, b in zip(evaluated["metrics"], evaluated_again["metrics"]))

# Artifact: reloaded inference equals the in-memory network, and the artifact holds no executable objects.
bundle = pt.load_bundle(trained["artifact"], env)
results["artifact_predicts_requested_row_count"] = len(pred_a) == 20
results["artifact_reload_matches_network_outputs"] = pt.predict_bundle(bundle, binary_rows[:10])["prediction"].tolist() == pred_a[:10]
raw_payload = torch.load(io.BytesIO(base64.b64decode(trained["artifact"])), weights_only=True, map_location="cpu")
results["artifact_payload_is_metadata_plus_state_dict"] = set(raw_payload) == {"metadata", "state_dict"} and all(isinstance(v, torch.Tensor) for v in raw_payload["state_dict"].values())
results["artifact_environment_matches_training"] = json.loads(raw_payload["metadata"])["environment"] == trained["environment"]

# Safe loading: an artifact carrying an arbitrary object is refused by weights_only, and the refusal is generic.
class Evil:
    def __reduce__(self):
        return (print, ("EXECUTED",))


buffer = io.BytesIO()
torch.save({"metadata": "{}", "state_dict": {"x": Evil()}}, buffer)
try:
    pt.load_bundle(base64.b64encode(buffer.getvalue()).decode(), env)
    results["arbitrary_objects_refused_by_weights_only"] = False
except common.IncompatibleArtifact as error:
    results["arbitrary_objects_refused_by_weights_only"] = str(error) == "unreadable artifact"

# Compatibility contract: provenance may drift, contract fields may not, and the recorded environment is checked first.
drift = {**trained["environment"], "adapter_sha256": "0" * 64, "platform": "Other-1.0", "python": ".".join(env["python"].split(".")[:2]) + ".99"}
results["adapter_and_platform_drift_is_permitted"] = pt.execute({"command": "predict", "snapshot": binary, "rows": binary_rows[:2], "artifact": trained["artifact"], "expectedEnvironment": drift})["predictions"] == pred_a[:2]
violations = {"different_backend": {"backend": "sklearn"}, "unsupported_format": {"artifact_format": "sklearn-pickle-v1"},
              "unsupported_protocol": {"protocol": "meticule-pytorch-v9"}, "unsupported_schema": {"artifact_schema": "2"},
              "torch_version_differs": {"torch": "0.0.1"}, "sklearn_version_differs": {"sklearn": "0.0.1"},
              "python_minor_differs": {"python": "2.7.18"}}
for name, change in violations.items():
    try:
        pt.execute({"command": "predict", "snapshot": binary, "rows": binary_rows[:2], "artifact": "AAAA", "expectedEnvironment": {**trained["environment"], **change}})
        results[f"rejects_{name}"] = False
    except common.IncompatibleArtifact as error:
        results[f"rejects_{name}"] = "unreadable" not in str(error)
    except Exception:
        results[f"rejects_{name}"] = False
tampered = pt.serialize(pt.train_network(binary, binary_rows, prepared["indices"]["train"]), {**env, "torch": "0.0.1"})
try:
    pt.load_bundle(tampered, env)
    results["rejects_torch_version_differs_embedded"] = False
except common.IncompatibleArtifact as error:
    results["rejects_torch_version_differs_embedded"] = str(error) == "torch version differs"

results["rejects_malformed_payload"] = False
bad_payload = io.BytesIO()
torch.save({"state_dict": {}}, bad_payload)
try:
    pt.load_bundle(base64.b64encode(bad_payload.getvalue()).decode(), env)
except common.IncompatibleArtifact:
    results["rejects_malformed_payload"] = True

# Architecture and training configuration are bounded and allowlisted; each bad input must be refused.
bad_configs = {
    "too_many_layers": {"hiddenLayers": [8, 8, 8, 8, 8]}, "too_wide": {"hiddenLayers": [300]}, "too_many_units": {"hiddenLayers": [256, 256, 256]},
    "empty_layers": {"hiddenLayers": []}, "dropout_too_high": {"dropout": 0.95}, "zero_epochs": {"epochs": 0}, "absurd_epochs": {"epochs": 10000},
    "unknown_optimizer": {"optimizer": "lbfgs"}, "unknown_activation": {"activation": "swish"}, "unknown_parameter": {"n_jobs": 4},
    "cross_entropy_for_regression": {"loss": "cross_entropy"}, "zero_learning_rate": {"learningRate": 0}, "bool_width": {"hiddenLayers": [True]},
}
refused = []
for name, hp in bad_configs.items():
    s = {**binary, "algorithm": "mlp_classifier" if name != "cross_entropy_for_regression" else "mlp_regressor", "hyperparameters": hp}
    try:
        pt.resolve_hyperparameters(s)
    except ValueError:
        refused.append(name)
results["invalid_configurations_refused"] = len(refused) == len(bad_configs)
details["refused"] = sorted(set(bad_configs) - set(refused))
results["defaults_match_catalog"] = pt.DEFAULTS == {"hiddenLayers": [64, 32], "activation": "relu", "dropout": 0.0, "epochs": 20, "batchSize": 32, "learningRate": 0.001, "optimizer": "adam"}

# Unsupported preprocessing in the NumPy path fails explicitly; a missing value without an imputer is never passed through.
no_imputer = spec("mlp_classifier", [("a", "number"), ("b", "number"), ("label", "string")], "label", ["a"], [("standard_scale", {})], {"hiddenLayers": [4], "epochs": 1}, ["accuracy"])
try:
    pt.execute({"command": "prepare", "snapshot": no_imputer, "rows": leak_rows})
    pt.execute({"command": "train", "snapshot": no_imputer, "rows": leak_rows, "prepared": {"indices": {"train": list(range(40)), "validation": [], "test": []}}})
    results["missing_value_without_imputer_fails_explicitly"] = False
except ValueError as error:
    results["missing_value_without_imputer_fails_explicitly"] = "imputer" in str(error)

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=float))
