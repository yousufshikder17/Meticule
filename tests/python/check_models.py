"""Every catalog model trains, evaluates every metric it supports, and reproduces under the same seed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory

import numpy as np
from sklearn.base import clone

from support import classification_rows, emit, full_run, ml_backend, regression_rows, snapshot

CLASS_COLUMNS = [("a", "number"), ("b", "number"), ("c", "number"), ("kind", "string"), ("flag", "boolean"), ("label", "string")]
STEPS = [("impute_median", {}), ("standard_scale", {}), ("impute_most_frequent", {}), ("one_hot_encode", {})]
failures, summary = [], {}
metadata = json.loads(sys.argv[1])  # TypeScript catalog: {id: {"taskType", "scores"}}

for model_id, info in metadata.items():
    classification = info["taskType"] == "classification"
    names = ["accuracy", "precision", "recall", "f1"] + (["roc_auc"] if info["scores"] != "none" else []) + (["log_loss"] if info["scores"] == "probability" else []) if classification else ["mae", "mse", "rmse", "r2"]
    rows = classification_rows() if classification else regression_rows()
    spec = (snapshot(model_id, CLASS_COLUMNS, "label", ["a", "b", "c", "kind", "flag"], STEPS, evaluation={"metrics": names, "cv": {"strategy": "stratified_kfold", "folds": 3}}, stratify=True)
            if classification else snapshot(model_id, [("a", "number"), ("b", "number"), ("y", "number")], "y", ["a", "b"], [("impute_mean", {}), ("min_max_scale", {})], evaluation={"metrics": names, "cv": {"strategy": "kfold", "folds": 3}}))
    try:
        prepared, model, evaluation = full_run(spec, rows)
        again = full_run(spec, rows)[2]
        key = lambda m: (m["name"], m["partition"], m.get("fold", -1))
        first = {key(m): m["value"] for m in evaluation["metrics"]}
        second = {key(m): m["value"] for m in again["metrics"]}
        got = {m["name"] for m in evaluation["metrics"] if m["partition"] == "cv"}
        ok = got == set(names) and all(np.isfinite(v) for v in first.values()) and first.keys() == second.keys() and all(abs(first[k] - second[k]) < 1e-9 for k in first)
        ok = ok and len({m["fold"] for m in evaluation["metrics"] if m["partition"] == "cv_fold"}) == 3 and all(m["std"] >= 0 for m in evaluation["metrics"] if m["partition"] == "cv")
        summary[model_id] = {"ok": bool(ok), "fitSeconds": model["performance"]["fitSeconds"], "predictSeconds": evaluation["performance"]["predictSeconds"]}
        if not ok:
            failures.append(model_id)
    except Exception as error:  # report which model, never swallow
        failures.append(f"{model_id}: {type(error).__name__}")
emit({"summary": summary, "failures": failures, "catalog": sorted(ml_backend.MODELS),
      "defaults": {mid: {k: v for k, v in clone(cls()).get_params().items()} for mid, (cls, _) in ml_backend.MODELS.items()}})
