"""Real JAX backend checks. Prints one JSON object of named results (booleans) plus diagnostics.

Runs the runner in-process with fixed seeds and tiny local fixtures. The TypeScript integration (durable chunks through the
processor, fencing, registry and endpoint) is covered by tests/integration/ml-jax.test.ts.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import jax_runner as jb  # noqa: E402  (sets the JAX platform flags before jax is imported)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_runtime_common as common  # noqa: E402

results, details = {}, {}
env = jb.environment()


def spec(algorithm, columns, target, features, steps, hyperparameters, metrics, cv=None, stratify=False, seed=7, training=None):
    evaluation = {"metrics": metrics, **({"cv": cv} if cv else {})}
    return {"algorithm": algorithm, "backend": "jax", "seed": seed, "hyperparameters": hyperparameters, "evaluation": evaluation,
            **({"training": training} if training else {}),
            "split": {"id": "00000000-0000-4000-8000-000000000000", "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": stratify},
            "dataset": {"schema": {"columns": [{"name": n, "type": t} for n, t in columns], "target": target}},
            "pipeline": {"definition": {"features": features, "steps": [{"operation": o, "parameters": p} for o, p in steps]}}}


def run(command, s, rows, **extra):
    return jb.execute({"command": command, "snapshot": s, "rows": rows, **extra})


def run_all(s, rows):
    prepared = run("prepare", s, rows)
    trained = run("train", s, rows, prepared=prepared, expectedEnvironment=prepared["environment"])
    evaluated = run("evaluate", s, rows, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])
    return prepared, trained, evaluated


def metric(evaluated, name, partition):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)


def refuses(fn, kind, code=None):
    try:
        fn()
    except kind as error:
        return code is None or getattr(error, "code", None) == code
    except Exception:
        return False
    return False


def archive(encoded):
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def rewrite(encoded, change):
    """Re-packs an archive after `change(arrays)` edits its arrays (used to forge tampered artifacts and checkpoints)."""
    with archive(encoded) as source:
        arrays = {name: source[name] for name in source.files}
    change(arrays)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


rng = np.random.RandomState(0)
binary_rows = [{"a": float(rng.randn() + (2.5 if i % 2 else -2.5)) if i % 11 else None, "b": float(rng.randn()),
                "kind": ["red", "blue", None][i % 3], "label": "yes" if i % 2 else "no"} for i in range(160)]
binary_columns = [("a", "number"), ("b", "number"), ("kind", "string"), ("label", "string")]
binary_steps = [("impute_median", {}), ("standard_scale", {}), ("impute_most_frequent", {}), ("one_hot_encode", {})]
binary = spec("mlp_classifier", binary_columns, "label", ["a", "b", "kind"], binary_steps,
              {"hiddenLayers": [16], "epochs": 40, "batchSize": 16, "learningRate": 0.01}, ["accuracy", "roc_auc", "log_loss"], stratify=True)
prepared, trained, evaluated = run_all(binary, binary_rows)
results["binary_learns_train_and_test"] = metric(evaluated, "accuracy", "train") > 0.9 and metric(evaluated, "accuracy", "test") > 0.85
results["binary_probability_metrics_defined"] = 0.9 < metric(evaluated, "roc_auc", "test") <= 1.0 and metric(evaluated, "log_loss", "test") < 0.5
details["binary_test_accuracy"] = metric(evaluated, "accuracy", "test")
results["training_records_losses_and_epochs"] = trained["performance"]["epochsCompleted"] == 40 and "finalValidationLoss" in trained["performance"]
results["environment_records_real_jax_cpu"] = (trained["environment"]["jax_platform"] == "cpu" and trained["environment"]["device_actual"] == "cpu"
                                              and trained["environment"]["jax"] == jax.__version__ and trained["format"] == "jax-params-v1")
bundle = jb.load_bundle(trained["artifact"], env)
probabilities = jb.predict_bundle(bundle, binary_rows[:20])["proba"]
results["probabilities_are_distributions"] = probabilities.shape == (20, 2) and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

multi_rows = [{"x": float(rng.randn() + 3 * (i % 3)), "y": float(rng.randn() + 3 * ((i % 3) == 1)), "label": f"c{i % 3}"} for i in range(150)]
multi = spec("mlp_classifier", [("x", "number"), ("y", "number"), ("label", "string")], "label", ["x", "y"], [("standard_scale", {})],
             {"hiddenLayers": [16, 8], "epochs": 60, "batchSize": 16, "learningRate": 0.02, "activation": "gelu"}, ["accuracy", "log_loss"])
results["multiclass_learns"] = metric(run_all(multi, multi_rows)[2], "accuracy", "test") > 0.8

reg_rows = [{"u": float(u), "v": float(v), "t": float(3 * u - 2 * v + 0.05 * rng.randn())} for u, v in rng.rand(200, 2) * 4]
regression = spec("mlp_regressor", [("u", "number"), ("v", "number"), ("t", "number")], "t", ["u", "v"], [("standard_scale", {})],
                  {"hiddenLayers": [32], "epochs": 80, "batchSize": 16, "learningRate": 0.01, "activation": "tanh"}, ["r2", "mae"])
reg_eval = run_all(regression, reg_rows)[2]
results["regression_fits"] = metric(reg_eval, "r2", "test") > 0.95
details["regression_test_r2"] = metric(reg_eval, "r2", "test")
for optimizer in ("sgd", "adamw"):
    s = spec("mlp_regressor", [("u", "number"), ("v", "number"), ("t", "number")], "t", ["u", "v"], [("standard_scale", {})],
             {"hiddenLayers": [16], "epochs": 30, "batchSize": 16, "learningRate": 0.05 if optimizer == "sgd" else 0.01, "optimizer": optimizer}, ["r2"])
    results[f"{optimizer}_optimizer_learns"] = metric(run_all(s, reg_rows)[2], "r2", "test") > 0.9

# Real autodiff: JAX gradients of the training objective equal central finite differences of an independent float64 NumPy
# implementation of the same network and loss.
architecture = {"hiddenLayers": [5, 4], "activation": "tanh", "dropout": 0.0}
params = jb.init_params(jax.random.key(3), architecture, 3, 1)
x = rng.randn(6, 3).astype(np.float32)
y = rng.randn(6, 1).astype(np.float32)


def numpy_loss(layers, inputs, targets):
    h = inputs.astype(np.float64)
    for layer in layers[:-1]:
        h = np.tanh(h @ layer["w"] + layer["b"])
    return float(np.mean((h @ layers[-1]["w"] + layers[-1]["b"] - targets) ** 2))


grads = jax.grad(lambda p: jb.loss_value("mse", jb.forward(p, jnp.asarray(x), "tanh", 0.0), jnp.asarray(y)))(params)
reference = [{k: np.asarray(v, dtype=np.float64) for k, v in layer.items()} for layer in params]
worst = 0.0
for i, layer in enumerate(reference):
    for part in ("w", "b"):
        for index in np.ndindex(layer[part].shape):
            plus = [{k: v.copy() for k, v in l.items()} for l in reference]
            minus = [{k: v.copy() for k, v in l.items()} for l in reference]
            plus[i][part][index] += 1e-6
            minus[i][part][index] -= 1e-6
            numeric = (numpy_loss(plus, x, y) - numpy_loss(minus, x, y)) / 2e-6
            worst = max(worst, abs(numeric - float(np.asarray(grads[i][part])[index])))
results["gradients_equal_finite_differences"] = worst < 1e-4
details["gradient_max_abs_error"] = worst

# Determinism: same seed and split -> identical artifact parameters and predictions; another seed -> different initial params.
_, again, _ = run_all(binary, binary_rows)
first, second = archive(trained["artifact"]), archive(again["artifact"])
results["same_seed_identical_parameters"] = all(np.array_equal(first[n], second[n]) for n in first.files if n != "metadata")
results["same_seed_identical_predictions"] = jb.predict_bundle(bundle, binary_rows)["prediction"].tolist() == \
    jb.predict_bundle(jb.load_bundle(again["artifact"], env), binary_rows)["prediction"].tolist()
init_a = jb.init_params(jb.derive_keys(7)[1], architecture, 3, 1)
init_b = jb.init_params(jb.derive_keys(8)[1], architecture, 3, 1)
results["different_seed_different_initial_parameters"] = not np.array_equal(np.asarray(init_a[0]["w"]), np.asarray(init_b[0]["w"]))

# Chunked training (fresh process state per chunk, as Meticule runs it) equals uninterrupted training, with dropout on.
chunky = spec("mlp_classifier", binary_columns, "label", ["a", "b", "kind"], binary_steps,
              {"hiddenLayers": [16, 8], "epochs": 6, "batchSize": 16, "learningRate": 0.01, "dropout": 0.3}, ["accuracy"])
chunk_prepared = run("prepare", chunky, binary_rows)
whole = run("train_chunk", chunky, binary_rows, prepared=chunk_prepared, state=None, epochs=6)
part_one = run("train_chunk", chunky, binary_rows, prepared=chunk_prepared, state=None, epochs=2)
part_two = run("train_chunk", chunky, binary_rows, prepared=chunk_prepared, state=part_one["checkpoint"], epochs=3)
part_three = run("train_chunk", chunky, binary_rows, prepared=chunk_prepared, state=part_two["checkpoint"], epochs=1)
whole_arrays, chunk_arrays = archive(whole["checkpoint"]), archive(part_three["checkpoint"])
results["chunked_equals_uninterrupted"] = (part_three["completedEpoch"] == 6 and set(whole_arrays.files) == set(chunk_arrays.files)
                                           and all(np.array_equal(whole_arrays[n], chunk_arrays[n]) for n in whole_arrays.files if n != "metadata")
                                           and [e["trainLoss"] for e in whole["epochs"]] == [e["trainLoss"] for e in part_one["epochs"] + part_two["epochs"] + part_three["epochs"]])
results["dropout_changes_training"] = not np.array_equal(
    whole_arrays["params/0/w"],
    archive(run("train_chunk", {**chunky, "hyperparameters": {**chunky["hyperparameters"], "dropout": 0.0}}, binary_rows,
                prepared=chunk_prepared, state=None, epochs=6)["checkpoint"])["params/0/w"])

# A checkpoint restores the adam moments and step exactly.
layers, key_data, meta = jb.load_checkpoint(part_two["checkpoint"], env)
restored = jb.restore_run(part_two["checkpoint"], chunky, binary_rows, {"indices": chunk_prepared["indices"]}, env)
results["checkpoint_restores_optimizer_state_exactly"] = (
    int(restored.opt_state["step"]) == meta["optimizerStep"] == 5 * int(np.ceil(len(chunk_prepared["indices"]["train"]) / 16))
    and all(np.array_equal(np.asarray(restored.opt_state[m][i][p]), layers[f"opt_{m}"][i][p]) for m in ("m", "v") for i in range(3) for p in ("w", "b"))
    and restored.completed == 5)

# Best-epoch export uses the best parameters; the default export uses the last.
best_spec = spec("mlp_classifier", binary_columns, "label", ["a", "b", "kind"], binary_steps,
                 {"hiddenLayers": [32], "epochs": 25, "batchSize": 8, "learningRate": 0.3, "optimizer": "sgd"}, ["accuracy"])
best_prepared = run("prepare", best_spec, binary_rows)
best_chunk = run("train_chunk", best_spec, binary_rows, prepared=best_prepared, state=None, epochs=25)
best_layers, _, best_meta = jb.load_checkpoint(best_chunk["checkpoint"], env)
exported_best = archive(run("export", best_spec, [], checkpoint=best_chunk["checkpoint"], restoreBest=True)["artifact"])
exported_last = archive(run("export", best_spec, [], checkpoint=best_chunk["checkpoint"], restoreBest=False)["artifact"])
results["best_epoch_export_uses_best_parameters"] = (best_meta["best"]["epoch"] < 25
                                                     and np.array_equal(exported_best["params/0/w"], best_layers["best"][0]["w"])
                                                     and np.array_equal(exported_last["params/0/w"], best_layers["params"][0]["w"]))
details["best_epoch"] = best_meta["best"]["epoch"]

# Safe loading: corrupt, foreign-runtime, pickled, reshaped or padded archives are refused before use.
results["corrupt_artifact_refused"] = refuses(lambda: jb.load_bundle(base64.b64encode(b"not an archive").decode(), env), common.IncompatibleArtifact)
foreign = {**env, "jax": "0.0.1"}
results["foreign_runtime_artifact_refused"] = refuses(lambda: jb.load_bundle(trained["artifact"], foreign), common.IncompatibleArtifact)
results["foreign_runtime_checkpoint_refused"] = refuses(lambda: jb.load_checkpoint(part_one["checkpoint"], foreign), common.IncompatibleCheckpoint)
buffer = io.BytesIO()
np.savez(buffer, metadata=np.array(json.dumps({"schema": 1})), evil=np.array([{"x": 1}], dtype=object))
pickled = base64.b64encode(buffer.getvalue()).decode()
results["object_array_artifact_refused"] = refuses(lambda: jb.load_bundle(pickled, env), common.IncompatibleArtifact)
results["object_array_checkpoint_refused"] = refuses(
    lambda: jb.load_checkpoint(rewrite(part_one["checkpoint"], lambda a: a.update({"opt_m/0/w": np.array([object()], dtype=object)})), env), common.InvalidCheckpoint)
results["object_array_in_real_artifact_refused"] = refuses(
    lambda: jb.load_bundle(rewrite(trained["artifact"], lambda a: a.update({"params/0/w": np.array([object()], dtype=object)})), env), common.IncompatibleArtifact)
results["reshaped_parameters_refused"] = refuses(
    lambda: jb.load_bundle(rewrite(trained["artifact"], lambda a: a.update({"params/0/w": a["params/0/w"][:-1]})), env), common.IncompatibleArtifact)
results["extra_array_refused"] = refuses(
    lambda: jb.load_bundle(rewrite(trained["artifact"], lambda a: a.update({"extra": np.zeros(2, dtype=np.float32)})), env), common.IncompatibleArtifact)
results["non_finite_parameters_refused"] = refuses(
    lambda: jb.load_bundle(rewrite(trained["artifact"], lambda a: a.update({"params/0/b": np.full_like(a["params/0/b"], np.nan)})), env), common.IncompatibleArtifact)
results["corrupt_checkpoint_invalid"] = refuses(lambda: jb.load_checkpoint(base64.b64encode(b"\x00" * 40).decode(), env), common.InvalidCheckpoint)
other_config = {**chunky, "hyperparameters": {**chunky["hyperparameters"], "learningRate": 0.02}}
results["other_configuration_checkpoint_refused"] = refuses(
    lambda: run("train_chunk", other_config, binary_rows, prepared=chunk_prepared, state=part_one["checkpoint"], epochs=1), common.IncompatibleCheckpoint)
results["forged_key_data_refused"] = refuses(
    lambda: jb.restore_run(rewrite(part_one["checkpoint"], lambda a: a.update({"key_data": np.array([0, 99], dtype=np.uint32)})),
                           chunky, binary_rows, {"indices": chunk_prepared["indices"]}, env), common.IncompatibleCheckpoint)

# Devices: CUDA and mixed precision are refused with the TensorFlow runner's codes, never run on CPU instead.
results["cuda_refused"] = refuses(lambda: run("prepare", {**binary, "training": {"device": "cuda", "precision": "fp32"}}, binary_rows), common.RunnerError, "cuda_unsupported")
results["mixed_precision_refused"] = refuses(lambda: run("prepare", {**binary, "training": {"device": "cpu", "precision": "mixed"}}, binary_rows),
                                             common.RunnerError, "unsupported_device_or_precision")

# Fitting scope: validation and test rows never change the fitted preprocessing or parameters.
changed = [dict(r) for r in binary_rows]
for i in prepared["indices"]["validation"] + prepared["indices"]["test"]:
    changed[i]["a"], changed[i]["kind"] = 1000.0, "violet"
refit = run("train", binary, changed, prepared=prepared)
refit_arrays = archive(refit["artifact"])
results["held_out_rows_do_not_change_the_fit"] = (json.loads(str(refit_arrays["metadata"]))["preprocessing"] == json.loads(str(first["metadata"]))["preprocessing"]
                                                  and all(np.array_equal(first[n], refit_arrays[n]) for n in first.files if n != "metadata"))

# Cross-validation fits a fresh model per fold.
cv_spec = spec("mlp_classifier", binary_columns, "label", ["a", "b", "kind"], binary_steps,
               {"hiddenLayers": [8], "epochs": 10, "batchSize": 16, "learningRate": 0.02}, ["accuracy"], cv={"strategy": "kfold", "folds": 3})
cv_prepared = run("prepare", cv_spec, binary_rows)
cv_trained = run("train", cv_spec, binary_rows, prepared=cv_prepared)
cv_eval = run("evaluate", cv_spec, binary_rows, prepared=cv_prepared, artifact=cv_trained["artifact"])
results["cv_fresh_model_per_fold"] = (cv_eval["performance"]["cvModelsBuilt"] == 3.0
                                      and sorted(m["fold"] for m in cv_eval["metrics"] if m["partition"] == "cv_fold") == [0, 1, 2]
                                      and any(m["partition"] == "cv" for m in cv_eval["metrics"]))

print(json.dumps({"results": results, "details": details}))
