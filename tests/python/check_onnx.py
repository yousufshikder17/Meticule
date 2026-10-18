"""ONNX export of every scikit-learn classification and regression model in the catalog, under both numeric preprocessing
chains, executed with ONNX Runtime and compared with the native pipeline under the platform's tolerance policy
(src/ml/onnx.ts PARITY_TOLERANCE). Also checks the refusals. Prints one JSON object of named results (booleans)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory

import numpy as np

from support import full_run, ml_backend, snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import onnx_runner  # noqa: E402
from ml_runtime_common import RunnerError  # noqa: E402

results, details = {}, {}
CHAINS = {"scaled": [("impute_median", {}), ("standard_scale", {})], "minmax": [("impute_mean", {}), ("min_max_scale", {}), ("variance_threshold", {})]}


def rows(task, n=120, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        a, b, c = rng.randn(), (None if i % 11 == 0 else float(rng.randn())), rng.randn()
        score = 2 * a - (b or 0) + 0.5 * c
        out.append({"a": float(a), "b": b, "c": float(c), "y": float(score) if task == "regression" else ("hi" if score > 0 else "lo")})
    return out


def within(task, native, onnx):
    if len(native["predictions"]) != len(onnx["predictions"]):
        return False, None
    if task == "regression":
        errors = [abs(o - n) - (1e-4 + 1e-3 * abs(n)) for n, o in zip(native["predictions"], onnx["predictions"])]
        return max(errors) <= 0, max(abs(o - n) for n, o in zip(native["predictions"], onnx["predictions"]))
    labels = all(n == o for n, o in zip(native["predictions"], onnx["predictions"]))
    proba = 0.0
    if native["probabilities"] is not None:
        proba = float(np.max(np.abs(np.array(native["probabilities"]) - np.array(onnx["probabilities"]))))
    return labels and proba <= 1e-4, proba


def export_and_run(spec, data, artifact, environment):
    test = [{k: r[k] for k in ("a", "b", "c")} for r in data[-24:]]
    exported = onnx_runner.export({"backend": "sklearn", "snapshot": spec, "rows": test, "artifact": artifact, "expectedEnvironment": environment})
    task = ml_backend.MODELS[spec["algorithm"]][1]
    predicted = onnx_runner.predict({"onnx": exported["onnx"], "features": ["a", "b", "c"], "rows": test, "task": task})
    return exported, predicted, task


for algorithm, (_, task) in sorted(ml_backend.MODELS.items()):
    if task == "anomaly_detection":
        continue
    for chain, steps in CHAINS.items():
        data = rows(task)
        spec = snapshot(algorithm, [("a", "number"), ("b", "number"), ("c", "number"), ("y", "number" if task == "regression" else "string")], "y", ["a", "b", "c"], steps=steps)
        _, model, _ = full_run(spec, data)
        exported, predicted, task = export_and_run(spec, data, model["artifact"], model["environment"])
        ok, error = within(task, exported["native"], predicted)
        results[f"{algorithm}:{chain}"] = ok
        details[f"{algorithm}:{chain}"] = {"error": error, "operators": exported["operators"], "opsets": exported["opsets"]}


def refused(code, build):
    try:
        build()
    except RunnerError as e:
        return e.code == code
    return False


def mixed():
    data = [{**r, "kind": "x" if i % 2 else "y"} for i, r in enumerate(rows("classification"))]
    spec = snapshot("logistic_regression", [("a", "number"), ("b", "number"), ("c", "number"), ("kind", "string"), ("y", "string")], "y", ["a", "b", "c", "kind"], steps=[("impute_median", {}), ("one_hot_encode", {}), ("impute_most_frequent", {})])
    _, model, _ = full_run(spec, data)
    onnx_runner.export({"backend": "sklearn", "snapshot": spec, "rows": data[:3], "artifact": model["artifact"], "expectedEnvironment": model["environment"]})


def boolean_labels():
    data = [{**r, "y": r["y"] == "hi"} for r in rows("classification")]
    spec = snapshot("logistic_regression", [("a", "number"), ("b", "number"), ("c", "number"), ("y", "boolean")], "y", ["a", "b", "c"], steps=[("impute_median", {})])
    _, model, _ = full_run(spec, data)
    onnx_runner.export({"backend": "sklearn", "snapshot": spec, "rows": data[:3], "artifact": model["artifact"], "expectedEnvironment": model["environment"]})


results["refuses_string_features"] = refused("unsupported_conversion", mixed)
results["refuses_boolean_labels"] = refused("unsupported_conversion", boolean_labels)
results["refuses_anomaly_models"] = refused("unsupported_conversion", lambda: onnx_runner.export({"backend": "sklearn", "snapshot": snapshot("isolation_forest", [("a", "number")], None, ["a"]), "rows": [], "artifact": "", "expectedEnvironment": {}}))
results["refuses_other_backends"] = refused("unsupported_conversion", lambda: onnx_runner.export({"backend": "pytorch", "snapshot": {}, "rows": [], "artifact": "", "expectedEnvironment": {}}))
print(json.dumps({"results": results, "details": details}))
