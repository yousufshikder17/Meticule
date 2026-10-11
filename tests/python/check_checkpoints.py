"""Checkpoint and resume checks for the PyTorch runner. Prints one JSON object of named booleans and diagnostics.

Chunked training must reproduce uninterrupted training. Checkpoints must be validated before they are used, and
restore-best must export exactly the best-epoch weights.
"""
import base64
import copy
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
rng = np.random.RandomState(3)
rows = []
for i in range(120):
    lab = "yes" if i % 2 else "no"
    rows.append({"a": float(rng.randn() + (2 if lab == "yes" else -2)) if i % 9 else None, "b": float(rng.randn()), "label": lab})
UUID = "00000000-0000-4000-8000-000000000000"


def spec(seed=7, hidden=(12,), dropout=0.2, epochs=6):
    return {"algorithm": "mlp_classifier", "backend": "pytorch", "seed": seed, "evaluation": {"metrics": ["accuracy"]},
            "hyperparameters": {"hiddenLayers": list(hidden), "epochs": epochs, "batchSize": 16, "learningRate": 0.01, "dropout": dropout, "optimizer": "adam"},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": True},
            "dataset": {"schema": {"columns": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "label", "type": "string"}], "target": "label"}},
            "pipeline": {"definition": {"features": ["a", "b"], "steps": [{"operation": "impute_median", "parameters": {}}, {"operation": "standard_scale", "parameters": {}}]}}}


def prepared_for(s):
    return pt.execute({"command": "prepare", "snapshot": s, "rows": rows})


def chunk(s, prepared, state, epochs):
    request = {"command": "train_chunk", "snapshot": s, "rows": rows, "prepared": prepared, "epochs": epochs,
               "state": state}
    return pt.execute(request)


def weights_of(checkpoint):
    payload = torch.load(io.BytesIO(base64.b64decode(checkpoint)), weights_only=True, map_location="cpu")
    return payload["model"]


s = spec()
prepared = prepared_for(s)

# 1. One six-epoch chunk versus two three-epoch chunks through a checkpoint: identical result.
whole = chunk(s, prepared, None, 6)
first = chunk(s, prepared, None, 3)
second = chunk(s, prepared, first["checkpoint"], 3)
same_weights = all(torch.equal(weights_of(whole["checkpoint"])[k], weights_of(second["checkpoint"])[k]) for k in weights_of(whole["checkpoint"]))
results["chunked_training_matches_uninterrupted_weights"] = same_weights
whole_losses = [e["trainLoss"] for e in whole["epochs"]]
chunked_losses = [e["trainLoss"] for e in first["epochs"] + second["epochs"]]
results["chunked_training_matches_uninterrupted_losses"] = all(abs(a - b) < 1e-12 for a, b in zip(whole_losses, chunked_losses)) and len(chunked_losses) == 6
results["epoch_numbering_is_contiguous"] = [e["epoch"] for e in first["epochs"] + second["epochs"]] == [1, 2, 3, 4, 5, 6] and second["completedEpoch"] == 6
details["max_loss_difference"] = max(abs(a - b) for a, b in zip(whole_losses, chunked_losses))

# 2. Restoring a checkpoint in a fresh run state continues exactly as the uninterrupted run would. Every chunk runs in a
# new process, so the determinism flags a fresh run sets are cleared first: a restore must set them itself.
torch.use_deterministic_algorithms(False)
torch.set_num_threads(2)
restored = pt.restore_run(first["checkpoint"], s, rows, prepared, env)
results["restored_run_enforces_determinism"] = torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 1
results["restored_run_reports_completed_epochs"] = restored.completed == 3
continued = pt.run_epochs(restored, 3)
results["restored_continuation_matches_losses"] = all(abs(a["trainLoss"] - b["trainLoss"]) < 1e-12 for a, b in zip(continued, whole["epochs"][3:]))

# 3. A checkpoint belongs to one configuration: any change to the configuration is refused.
other_seed = copy.deepcopy(s); other_seed["seed"] = 8
try:
    pt.restore_run(first["checkpoint"], other_seed, rows, prepared, env)
    results["fingerprint_mismatch_refused"] = False
except ValueError as error:
    results["fingerprint_mismatch_refused"] = "different training configuration" in str(error)
other_arch = spec(hidden=(10,))
try:
    pt.restore_run(first["checkpoint"], other_arch, rows, prepared, env)
    results["architecture_change_refused"] = False
except ValueError:
    results["architecture_change_refused"] = True

# 4. Corrupt, foreign or tampered checkpoints are refused before any state is used.
def refused(checkpoint, current=env):
    try:
        pt.restore_run(checkpoint, s, rows, prepared, current)
        return False
    except (common.IncompatibleArtifact, ValueError, RuntimeError):
        return True


results["corrupt_checkpoint_refused"] = refused(base64.b64encode(b"not a checkpoint").decode())
buffer = io.BytesIO(); torch.save({"metadata": "{}", "model": {"x": 1}}, buffer)
results["malformed_checkpoint_refused"] = refused(base64.b64encode(buffer.getvalue()).decode())
payload = torch.load(io.BytesIO(base64.b64decode(first["checkpoint"])), weights_only=True, map_location="cpu")
meta = json.loads(payload["metadata"])
for name, change in {"torch_version": {"torch": "0.0.1"}, "format": {"format": "pytorch-checkpoint-v9"}}.items():
    if name == "format":
        bad_meta = {**meta, **change}
    else:
        bad_meta = {**meta, "environment": {**meta["environment"], **change}}
    tampered = {**payload, "metadata": json.dumps(bad_meta)}
    buffer = io.BytesIO(); torch.save(tampered, buffer)
    results[f"tampered_checkpoint_{name}_refused"] = refused(base64.b64encode(buffer.getvalue()).decode())
class Evil:
    def __reduce__(self):
        return (print, ("EXECUTED",))
buffer = io.BytesIO(); torch.save({"metadata": "{}", "model": Evil(), "optimizer": {}, "generator": torch.zeros(1, dtype=torch.uint8), "rng": torch.zeros(1, dtype=torch.uint8), "best_model": None}, buffer)
results["arbitrary_objects_in_checkpoint_refused"] = refused(base64.b64encode(buffer.getvalue()).decode())

# 5. Export builds the normal artifact; restoring the best epoch returns exactly the best-epoch weights.
long = spec(epochs=12)
long["hyperparameters"]["learningRate"] = 0.5  # high rate: validation loss rises after its minimum
long["hyperparameters"]["dropout"] = 0.0
long_prepared = prepared_for(long)
trained = chunk(long, long_prepared, None, 12)
validations = [e["validationLoss"] for e in trained["epochs"]]
best_index = int(np.argmin(validations)) + 1
best_export = pt.execute({"command": "export", "snapshot": long, "rows": [], "checkpoint": trained["checkpoint"], "restoreBest": True})
last_export = pt.execute({"command": "export", "snapshot": long, "rows": [], "checkpoint": trained["checkpoint"], "restoreBest": False})
results["export_reports_best_epoch"] = int(best_export["performance"]["bestEpoch"]) == best_index
details["best_is_last_epoch"] = best_index == 12
results["export_best_and_last_artifacts_are_distinct_states"] = best_index == 8 or best_export["artifact"] != last_export["artifact"]
bundle_best = pt.load_bundle(best_export["artifact"], env)
bundle_last = pt.load_bundle(last_export["artifact"], env)
state_best = bundle_best["network"].state_dict()
stored_best = torch.load(io.BytesIO(base64.b64decode(trained["checkpoint"])), weights_only=True, map_location="cpu")["best_model"]
results["exported_weights_are_the_best_epoch_weights"] = all(torch.equal(state_best[k], stored_best[k]) for k in stored_best)
results["exported_artifact_loads_through_inference_path"] = len(pt.predict_bundle(bundle_best, rows[:5])["prediction"]) == 5
results["export_refuses_different_algorithm"] = False
try:
    pt.execute({"command": "export", "snapshot": {**long, "algorithm": "mlp_regressor"}, "rows": [], "checkpoint": trained["checkpoint"], "restoreBest": True})
except ValueError:
    results["export_refuses_different_algorithm"] = True

# 6. Epoch summaries are bounded: one entry per epoch in the chunk, no per-batch data.
results["epoch_summaries_have_only_bounded_fields"] = all(set(e) == {"epoch", "trainLoss", "validationLoss", "elapsedSeconds"} for e in trained["epochs"])
details["best_epoch"] = best_index

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=float))
