"""Legacy guard for the neural runners (spec amendment B8). Test-only.

neural_common.py and pytorch_runner.py in this folder are exact copies of ml/ at public commit 1b4118f (before Phase 4).
`guard.py frozen OUT` trains with them and writes, per model, the artifact, a checkpoint after 2 of 4 epochs and the frozen
runner's own predictions (from the artifact, and after resuming that checkpoint) to OUT. `guard.py current OUT`, in a
fresh process, predicts with the frozen artifact and resumes the frozen checkpoint with the current ml/ runner; both
must succeed with predictions identical to the frozen runner's. ml_runtime_common is the current one in both processes.
Prints one JSON line: {"results": {name: bool}}.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parents[3] / "ml"
MODE, OUT = sys.argv[1], Path(sys.argv[2])
sys.path[:0] = [str(HERE), str(PYTHON)] if MODE == "frozen" else [str(PYTHON)]

import neural_common  # noqa: E402
import pytorch_runner as pt  # noqa: E402

# Git blob ids of the two files at public 1b4118f (line endings normalised as git stores them).
FROZEN_BLOBS = {"neural_common.py": "c0d3755fee71b5d08b3fb7732f0c3f390c4c2ceb", "pytorch_runner.py": "6e75ee2acd9508498dc20170b8b78f4cda9dfaf3"}
UUID = "00000000-0000-4000-8000-000000000000"
COLUMNS = [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "kind", "type": "string"},
           {"name": "flag", "type": "boolean"}, {"name": "label", "type": "string"}, {"name": "y", "type": "number"}]
STEPS = [{"operation": "impute_median", "parameters": {}}, {"operation": "standard_scale", "parameters": {}},
         {"operation": "impute_most_frequent", "parameters": {}}, {"operation": "one_hot_encode", "parameters": {}}]


def spec(algorithm, target):
    return {"algorithm": algorithm, "backend": "pytorch", "seed": 11, "evaluation": {},
            "hyperparameters": {"hiddenLayers": [8], "epochs": 4, "batchSize": 16, "learningRate": 0.01, "dropout": 0.1, "optimizer": "adam"},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False},
            "dataset": {"schema": {"id": UUID, "columns": COLUMNS, "target": target}},
            "pipeline": {"definition": {"features": ["a", "b", "kind", "flag"], "steps": STEPS}}}


SPECS = {"classifier": spec("mlp_classifier", "label"), "regressor": spec("mlp_regressor", "y")}
ROWS = [{"a": None if i % 7 == 0 else float(i % 13) - 6.0, "b": float((i * 37) % 11) / 3, "kind": ["red", "blue", None][i % 3],
         "flag": i % 4 == 0, "label": "yes" if (i % 13) > 6 else "no", "y": float(i % 13) * 1.5 - (i % 5)} for i in range(90)]
QUERY = ROWS[:12] + [{"a": None, "b": 2.0, "kind": "green", "flag": True}]  # an unseen category and a missing number


def run(command, s, **extra):
    return pt.execute({"command": command, "snapshot": s, "rows": extra.pop("rows", ROWS), **extra})


def resumed(s, prepared, checkpoint):
    """Two more epochs from `checkpoint`, exported and used for prediction."""
    rest = run("train_chunk", s, prepared=prepared, epochs=2, state=checkpoint)
    model = run("export", s, checkpoint=rest["checkpoint"])
    return run("predict", s, rows=QUERY, artifact=model["artifact"], expectedEnvironment=model["environment"])["predictions"]


def git_blob(path):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def attempt(check):
    """A refusal by the current runner (incompatible artifact or checkpoint, changed fingerprint) is a failed check."""
    try:
        return bool(check())
    except Exception:
        return False


if MODE == "frozen":
    assert Path(pt.__file__).parent == HERE and Path(neural_common.__file__).parent == HERE
    record = {}
    for name, s in SPECS.items():
        prepared = run("prepare", s)
        model = run("train", s, prepared=prepared, expectedEnvironment=prepared["environment"])
        half = run("train_chunk", s, prepared=prepared, epochs=2, state=None)
        record[name] = {"prepared": prepared, "artifact": model["artifact"], "environment": model["environment"], "checkpoint": half["checkpoint"],
                        "predictions": run("predict", s, rows=QUERY, artifact=model["artifact"], expectedEnvironment=model["environment"])["predictions"],
                        "resumed": resumed(s, prepared, half["checkpoint"])}
    OUT.write_text(json.dumps(record), encoding="utf-8")
    print(json.dumps({"written": sorted(record)}))
else:
    assert Path(pt.__file__).parent == PYTHON and Path(neural_common.__file__).parent == PYTHON
    record = json.loads(OUT.read_text(encoding="utf-8"))
    results = {"frozen_copies_are_phase3": all(git_blob(HERE / file) == blob for file, blob in FROZEN_BLOBS.items())}
    for name, s in SPECS.items():
        frozen = record[name]
        without_env = lambda prepared: {k: v for k, v in prepared.items() if k != "environment"}  # noqa: E731
        results[f"{name}_prepare_matches_frozen"] = attempt(lambda: without_env(run("prepare", s)) == without_env(frozen["prepared"]))
        results[f"{name}_frozen_artifact_predicts_identically"] = attempt(lambda: run(
            "predict", s, rows=QUERY, artifact=frozen["artifact"], expectedEnvironment=frozen["environment"])["predictions"] == frozen["predictions"])
        results[f"{name}_frozen_checkpoint_resumes_identically"] = attempt(lambda: resumed(s, frozen["prepared"], frozen["checkpoint"]) == frozen["resumed"])
    print(json.dumps({"results": results}))
