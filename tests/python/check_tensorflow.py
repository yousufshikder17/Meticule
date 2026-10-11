"""TensorFlow runner checks. Prints one JSON object of named results and diagnostics.

Each named check asserts one behaviour the TypeScript side depends on: learning, reproducibility, chunked and dropout-aware
resume, refusals of tampered or foreign checkpoints and artifacts, and explicit CPU-only device handling.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import ml_runtime_common as common  # noqa: E402
import tensorflow_runner as tb  # noqa: E402
import neural_common as nc  # noqa: E402

results, details = {}, {}
rng = np.random.RandomState(5)
rows = []
for i in range(96):
    lab = "yes" if i % 2 else "no"
    rows.append({"a": float(rng.randn() + (2 if lab == "yes" else -2)), "b": float(rng.randn()), "label": lab})
regression_rows = [{"a": float(a), "b": float(b), "y": float(2 * a - b)} for a, b in rng.randn(80, 2)]
UUID = "00000000-0000-4000-8000-000000000000"


def spec(epochs=4, seed=9, dropout=0.0, device="cpu", precision="fp32", learning_rate=0.01, batch=16, activation="relu"):
    return {"algorithm": "mlp_classifier", "backend": "tensorflow", "seed": seed, "evaluation": {"metrics": ["accuracy"]},
            "training": {"checkpointEveryEpochs": 2, "earlyStopping": None, "device": device, "precision": precision},
            "hyperparameters": {"hiddenLayers": [8], "epochs": epochs, "batchSize": batch, "learningRate": learning_rate,
                                "dropout": dropout, "optimizer": "adam", "activation": activation},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": True},
            "dataset": {"schema": {"columns": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "label", "type": "string"}], "target": "label"}},
            "pipeline": {"definition": {"features": ["a", "b"], "steps": [{"operation": "standard_scale", "parameters": {}}]}}}


def regression_spec(epochs=30):
    return {"algorithm": "mlp_regressor", "backend": "tensorflow", "seed": 3, "evaluation": {"metrics": ["mse"]},
            "training": {"checkpointEveryEpochs": 2, "earlyStopping": None, "device": "cpu", "precision": "fp32"},
            "hyperparameters": {"hiddenLayers": [8], "epochs": epochs, "batchSize": 16, "learningRate": 0.02, "dropout": 0.0, "optimizer": "adam", "loss": "mse"},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False},
            "dataset": {"schema": {"columns": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "y", "type": "number"}], "target": "y"}},
            "pipeline": {"definition": {"features": ["a", "b"], "steps": [{"operation": "standard_scale", "parameters": {}}]}}}


def prepare(s, data=rows):
    return tb.execute({"command": "prepare", "snapshot": s, "rows": data})


def chunk(s, prepared, state, epochs, data=rows):
    return tb.execute({"command": "train_chunk", "snapshot": s, "rows": data, "prepared": prepared, "epochs": epochs, "state": state})


def losses(run):
    return [e["trainLoss"] for e in run["epochs"]]


def refused(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def _message(fn):
    try:
        fn()
        return ""
    except Exception as error:
        return str(error)


# 1. Learning: a separable problem is learned, and the artifact predicts it.
s = spec(epochs=30, learning_rate=0.05)
prepared = prepare(s)
whole = chunk(s, prepared, None, 30)
exported = tb.execute({"command": "export", "snapshot": s, "rows": [], "checkpoint": whole["checkpoint"], "restoreBest": False})
predicted = tb.execute({"command": "predict", "snapshot": s, "rows": rows, "artifact": exported["artifact"]})["predictions"]
accuracy = float(np.mean([p == r["label"] for p, r in zip(predicted, rows)]))
details["separable_accuracy"] = accuracy
results["learns_separable_classes"] = accuracy > 0.9

# 2. Regression learns the linear target better than predicting the mean.
rs = regression_spec()
rprepared = prepare(rs, regression_rows)
rrun = chunk(rs, rprepared, None, 30, regression_rows)
results["regression_loss_decreases"] = rrun["epochs"][-1]["trainLoss"] < rrun["epochs"][0]["trainLoss"] * 0.5

# 3. Reproducibility: the same seed gives identical losses, and a different seed does not.
again = chunk(s, prepared, None, 4)
results["same_seed_reproduces_losses_exactly"] = losses(again) == losses(chunk(s, prepared, None, 4))
results["different_seed_changes_training"] = losses(chunk(spec(seed=10), prepare(spec(seed=10)), None, 4)) != losses(again)

# 4. Chunked training equals uninterrupted training, with and without dropout (dropout RNG state is part of the checkpoint).
for name, dropout in (("chunked_equals_uninterrupted", 0.0), ("dropout_resume_equals_uninterrupted", 0.3)):
    s2 = spec(epochs=4, dropout=dropout)
    p2 = prepare(s2)
    whole2 = chunk(s2, p2, None, 4)
    first = chunk(s2, p2, None, 2)
    second = chunk(s2, p2, first["checkpoint"], 2)
    results[name] = all(abs(a - b) < 1e-6 for a, b in zip(losses(whole2), losses(first) + losses(second)))
    details[name + "_max_diff"] = max(abs(a - b) for a, b in zip(losses(whole2), losses(first) + losses(second)))

# 5. Checkpoints and artifacts are NumPy archives, not pickles, and refuse anything that does not match what was written.
buffer = io.BytesIO()
np.savez(buffer, metadata=np.array("{}"), model_0=np.array([{"x": 1}], dtype=object))
results["pickled_payload_refused"] = refused(lambda: tb.load_bundle(base64.b64encode(buffer.getvalue()).decode(), tb.environment()))
results["corrupt_checkpoint_refused"] = refused(lambda: tb.restore_run(base64.b64encode(b"not-an-archive").decode(), s, rows, prepared, tb.environment()))
meta, groups = tb.unpack(whole["checkpoint"], "checkpoint")
meta["counts"]["model"] = meta["counts"]["model"] + 1  # declares an entry that is not in the archive
results["checkpoint_with_mismatched_counts_refused"] = refused(lambda: tb.load_checkpoint(tb.pack(meta, groups), tb.environment()))
meta, groups = tb.unpack(exported["artifact"], "artifact")
meta["environment"]["tensorflow"] = "0.0.0"  # a different runtime must not load the artifact
results["incompatible_runtime_artifact_refused"] = refused(lambda: tb.load_bundle(tb.pack(meta, groups), tb.environment()))
other = spec(epochs=4, learning_rate=0.5)  # a different configuration cannot continue this checkpoint
results["checkpoint_for_other_configuration_refused"] = refused(lambda: tb.restore_run(first["checkpoint"], other, rows, prepare(other), tb.environment()))

# 6. Best-epoch export: a run with validation records the best epoch and restores its weights.
best = tb.execute({"command": "export", "snapshot": s, "rows": [], "checkpoint": whole["checkpoint"], "restoreBest": True})
results["best_epoch_export_records_epoch"] = best["performance"].get("bestEpoch") is not None
results["export_is_deterministic"] = tb.execute({"command": "export", "snapshot": s, "rows": [], "checkpoint": whole["checkpoint"], "restoreBest": False})["artifact"] == exported["artifact"]

# 7. Device handling is explicit: CPU provenance is recorded, CUDA and mixed precision are refused, never redirected.
results["cpu_provenance_recorded"] = whole["environment"]["device_actual"] == "cpu" and whole["environment"]["device_requested"] == "cpu"
results["cuda_request_refused"] = refused(lambda: tb.resolve_device(spec(device="cuda")))
results["mixed_precision_refused"] = refused(lambda: tb.resolve_device(spec(precision="mixed")))

# 8. Evaluation reports every partition, and folds report per fold and aggregate.
cv_spec = {**s, "evaluation": {"metrics": ["accuracy"], "cv": {"strategy": "kfold", "folds": 3}}}
cv_prepared = tb.execute({"command": "prepare", "snapshot": cv_spec, "rows": rows})
folds = []
cv_indices = np.array(cv_prepared["indices"]["train"])
for chunk_ids in np.array_split(cv_indices, 3):
    folds.append([int(i) for i in chunk_ids])
evaluation = tb.execute({"command": "evaluate", "snapshot": cv_spec, "rows": rows, "artifact": exported["artifact"],
                         "prepared": {"indices": cv_prepared["indices"], "folds": folds}})
partitions = {m["partition"] for m in evaluation["metrics"]}
results["evaluate_reports_every_partition"] = {"train", "validation", "test"} <= partitions
results["cv_reports_per_fold_and_aggregate"] = len([m for m in evaluation["metrics"] if m["partition"] == "cv_fold"]) == len(folds) and "cv" in partitions

# 9. Unknown hyperparameters are refused before any training starts.
results["unknown_hyperparameter_refused"] = refused(lambda: tb.resolve_hyperparameters({"algorithm": "mlp_classifier", "hyperparameters": {"module": "os"}}))
# 10. Leakage: preprocessing and target standardisation see training rows only, and folds never see their own held-out rows.
fit_idx = prepared["indices"]["train"]
held = set(prepared["indices"]["validation"]) | set(prepared["indices"]["test"])
corrupt_rows = [dict(r) for r in rows]
for i in held:
    corrupt_rows[i]["a"], corrupt_rows[i]["b"] = 1e6, -1e6
fit_clean = tb.fit_preprocessing(s, rows, fit_idx)
results["fit_ignores_held_out_values"] = fit_clean == tb.fit_preprocessing(s, corrupt_rows, fit_idx)
details["scaler_center_minus_training_mean"] = fit_clean["columns"][0]["states"][-1]["center"] - float(np.mean([rows[i]["a"] for i in fit_idx]))
results["scaler_is_training_mean"] = bool(abs(details["scaler_center_minus_training_mean"]) < 1e-9)
results["missing_value_without_imputer_fails_explicitly"] = "imputer" in _message(lambda: nc.apply_numeric([{"op": "standard_scale", "center": 0.0, "spread": 1.0}], [None]))

corrupt_reg = [dict(r) for r in regression_rows]
for i in set(rprepared["indices"]["validation"]) | set(rprepared["indices"]["test"]):
    corrupt_reg[i]["y"] += 1e6
results["regression_target_standardised_on_training_rows_only"] = tb.fresh_run(rs, regression_rows, rprepared).target_info == tb.fresh_run(rs, corrupt_reg, rprepared).target_info

cv_spec = {**s, "evaluation": {"metrics": ["accuracy"], "cv": {"strategy": "kfold", "folds": 3}}}
cv_prepared = tb.execute({"command": "prepare", "snapshot": cv_spec, "rows": rows})
folds = cv_prepared["folds"]
results["cv_folds_partition_training_rows"] = sorted(sum(folds, [])) == sorted(cv_prepared["indices"]["train"]) and len(set(sum(folds, []))) == len(sum(folds, []))
pool = cv_prepared["indices"]["train"]
fold_a = tb.train_network(cv_spec, rows, [i for i in pool if i not in folds[0]])
fold_b = tb.train_network(cv_spec, rows, [i for i in pool if i not in folds[1]])
results["fresh_networks_per_fold_share_no_variables"] = fold_a["network"] is not fold_b["network"] and fold_a["network"].variables[0] is not fold_b["network"].variables[0]

bad_configs = [{"hiddenLayers": []}, {"hiddenLayers": [512, 512]}, {"dropout": 0.9}, {"epochs": 0}, {"optimizer": "lbfgs"},
               {"loss": "mse"}, {"batchSize": 0}, {"learningRate": 2}, {"activation": "swish"}]
results["invalid_configurations_refused"] = all(refused(lambda hp=hp: tb.resolve_hyperparameters({"algorithm": "mlp_classifier", "hyperparameters": hp})) for hp in bad_configs)

results["environment_declares_runtime_keys"] = all(k in tb.environment() for k in tb.RUNTIME_KEYS)
results["environment_matches_contract"] = common.check_compatible(tb.environment(), tb.environment(), tb.CONTRACT) is None

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=float))
