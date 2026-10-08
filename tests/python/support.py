"""Shared helpers for backend checks. Run through vitest with ML_TEST_PYTHON; never imported by production code."""
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import sklearn_runner as ml_backend  # noqa: E402

UUID = "00000000-0000-4000-8000-000000000000"


def snapshot(algorithm, columns, target, features, steps=(), hyperparameters=None, evaluation=None, stratify=False, seed=7):
    return {"algorithm": algorithm, "backend": "sklearn", "hyperparameters": hyperparameters or {}, "seed": seed,
            "evaluation": evaluation or {},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": stratify},
            "dataset": {"schema": {"id": UUID, "columns": [{"name": n, "type": t, "nullable": True} for n, t in columns], "target": target}},
            "pipeline": {"definition": {"features": list(features), "steps": [{"operation": o, "parameters": p} for o, p in steps]}}}


def run(command, spec, rows, **extra):
    return ml_backend.execute({"command": command, "snapshot": spec, "rows": rows, **extra})


def full_run(spec, rows):
    """prepare -> train -> evaluate exactly as the TypeScript processor drives the protocol."""
    prepared = run("prepare", spec, rows)
    model = run("train", spec, rows, prepared=prepared, expectedEnvironment=prepared["environment"])
    evaluation = run("evaluate", spec, rows, prepared=prepared, artifact=model["artifact"], expectedEnvironment=model["environment"])
    return prepared, model, evaluation


def classification_rows(n=150, classes=2, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, classes, n)
    return [{"a": float(labels[i] * 2 + rng.randn()), "b": float(rng.randn()), "c": float(labels[i] - rng.randn() * 0.5),
             "kind": "red" if rng.rand() > 0.5 else "blue", "flag": bool(rng.rand() > 0.5), "label": f"class_{labels[i]}"} for i in range(n)]


def regression_rows(n=150, seed=0):
    rng = np.random.RandomState(seed)
    return [{"a": float(v), "b": float(rng.randn()), "y": float(3 * v + 1 + rng.randn() * 0.1)} for v in rng.rand(n) * 10]


def emit(payload):
    print(json.dumps(payload))
