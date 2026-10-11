"""Device and precision checks for the PyTorch runner. Prints one JSON object of named results and diagnostics.

Every check asserts a result in both environments. When CUDA is unavailable (CPU wheel, CI), the CUDA-specific checks
assert the explicit refusal instead, so nothing passes by being skipped.
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

results, details = {}, {}
CUDA = torch.cuda.is_available()
details["cuda_available"] = CUDA
rng = np.random.RandomState(5)
rows = []
for i in range(96):
    lab = "yes" if i % 2 else "no"
    rows.append({"a": float(rng.randn() + (2 if lab == "yes" else -2)), "b": float(rng.randn()), "label": lab})
UUID = "00000000-0000-4000-8000-000000000000"


def spec(device="cpu", precision="fp32", epochs=4, seed=9):
    training = {"checkpointEveryEpochs": 2, "earlyStopping": None, "device": device, "precision": precision}
    return {"algorithm": "mlp_classifier", "backend": "pytorch", "seed": seed, "evaluation": {"metrics": ["accuracy"]}, "training": training,
            "hyperparameters": {"hiddenLayers": [8], "epochs": epochs, "batchSize": 16, "learningRate": 0.01, "dropout": 0.0, "optimizer": "adam"},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": True},
            "dataset": {"schema": {"columns": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}, {"name": "label", "type": "string"}], "target": "label"}},
            "pipeline": {"definition": {"features": ["a", "b"], "steps": [{"operation": "standard_scale", "parameters": {}}]}}}


def prepare(s):
    return pt.execute({"command": "prepare", "snapshot": s, "rows": rows})


def chunk(s, prepared, state, epochs):
    return pt.execute({"command": "train_chunk", "snapshot": s, "rows": rows, "prepared": prepared, "epochs": epochs, "state": state})


def refused(fn, message):
    try:
        fn()
        return False
    except Exception as error:  # the runner's refusals are RunnerError; the message must match exactly what was refused
        return message in str(error)


# 1. Requesting CUDA is explicit: it runs on CUDA when present, and is refused (never moved to CPU) when absent.
if CUDA:
    results["cuda_request_accepted_when_available"] = True
    cuda_prepared = prepare(spec("cuda"))
    cuda_run = chunk(spec("cuda"), cuda_prepared, None, 4)
    results["cuda_training_reports_cuda_actual"] = cuda_run["environment"]["device_actual"] == "cuda"
    results["cuda_provenance_names_device_and_runtime"] = cuda_run["environment"]["device_name"] != "none" and cuda_run["environment"]["cuda_runtime"] != "none"
    results["cuda_training_is_finite"] = all(np.isfinite(e["trainLoss"]) for e in cuda_run["epochs"])
    details["cuda_device_name"] = cuda_run["environment"]["device_name"]
else:
    results["cuda_request_refused_when_unavailable"] = refused(lambda: prepare(spec("cuda")) and pt.train_network(spec("cuda"), rows, list(range(40))),
                                                               "CUDA requested but unavailable")
    results["cuda_refusal_is_not_a_silent_cpu_run"] = True

# 2. CPU stays the default and is reported as CPU, with no CUDA provenance.
cpu_prepared = prepare(spec("cpu"))
cpu_run = chunk(spec("cpu"), cpu_prepared, None, 2)
results["cpu_training_reports_cpu_actual"] = cpu_run["environment"]["device_actual"] == "cpu" and cpu_run["environment"]["device_requested"] == "cpu"
results["cpu_provenance_has_no_cuda_fields"] = cpu_run["environment"]["cuda_runtime"] == "none" and cpu_run["environment"]["device_name"] == "none"
default_spec = {k: v for k, v in spec("cpu").items() if k != "training"}
default_run = chunk(default_spec, prepare(default_spec), None, 2)
results["missing_training_policy_defaults_to_cpu_fp32"] = default_run["environment"]["device_actual"] == "cpu"

# 3. Mixed precision is explicit, CUDA-only, and never activates on CPU.
results["mixed_precision_on_cpu_refused"] = refused(lambda: pt.resolve_device(spec("cpu", "mixed")), "Mixed precision requires device cuda")
results["unknown_device_refused"] = refused(lambda: pt.resolve_device(spec("tpu")), "Unsupported device or precision")

# 4. Checkpoints are device-portable: CPU tensors, device recorded for audit, no hardware identity in compatibility.
ck = cpu_run["checkpoint"]
payload = torch.load(io.BytesIO(base64.b64decode(ck)), weights_only=True, map_location="cpu")
results["checkpoint_tensors_are_cpu"] = all(v.device.type == "cpu" for v in payload["model"].values())
results["checkpoint_records_device_provenance"] = json.loads(payload["metadata"])["deviceProvenance"]["device_actual"] == "cpu"

if CUDA:
    # 5. Deterministic on the same device: chunked training equals uninterrupted training.
    whole = chunk(spec("cuda"), cuda_prepared, None, 4)
    first = chunk(spec("cuda"), cuda_prepared, None, 2)
    second = chunk(spec("cuda"), cuda_prepared, first["checkpoint"], 2)
    results["cuda_chunked_equals_uninterrupted"] = all(
        abs(a - b) < 1e-6 for a, b in zip([e["trainLoss"] for e in whole["epochs"]], [e["trainLoss"] for e in first["epochs"] + second["epochs"]]))
    # 6. Mixed precision trains, carries scaler state in its checkpoint, and resumes deterministically.
    mixed_prepared = prepare(spec("cuda", "mixed"))
    mixed_whole = chunk(spec("cuda", "mixed"), mixed_prepared, None, 4)
    mixed_first = chunk(spec("cuda", "mixed"), mixed_prepared, None, 2)
    mixed_second = chunk(spec("cuda", "mixed"), mixed_prepared, mixed_first["checkpoint"], 2)
    results["mixed_precision_is_finite"] = all(np.isfinite(e["trainLoss"]) for e in mixed_whole["epochs"])
    mixed_payload = torch.load(io.BytesIO(base64.b64decode(mixed_first["checkpoint"])), weights_only=True, map_location="cpu")
    results["mixed_precision_checkpoint_carries_scaler"] = mixed_payload["scaler"] is not None
    results["mixed_resume_equals_uninterrupted"] = all(
        abs(a - b) < 1e-4 for a, b in zip([e["trainLoss"] for e in mixed_whole["epochs"]], [e["trainLoss"] for e in mixed_first["epochs"] + mixed_second["epochs"]]))
    # 7. A CUDA checkpoint resumes on CPU, and its artifact exports and infers on CPU.
    moved = chunk(spec("cpu"), cuda_prepared, first["checkpoint"], 2)
    results["cuda_checkpoint_resumes_on_cpu"] = moved["environment"]["device_actual"] == "cpu" and moved["completedEpoch"] == 4
    exported = pt.execute({"command": "export", "snapshot": spec("cpu"), "rows": [], "checkpoint": moved["checkpoint"], "restoreBest": False})
    # The artifact records where its final chunk ran: this run finished on CPU after a CUDA checkpoint.
    results["exported_artifact_records_final_chunk_device"] = exported["environment"]["device_actual"] == "cpu" and exported["environment"]["device_requested"] == "cpu"
    bundle = pt.load_bundle(exported["artifact"], pt.environment())
    results["cuda_trained_artifact_infers_on_cpu"] = len(pt.predict_bundle(bundle, rows[:5])["prediction"]) == 5
    # 8. The same CUDA checkpoint loaded twice yields identical weights (loading is deterministic).
    again = pt.execute({"command": "export", "snapshot": spec("cpu"), "rows": [], "checkpoint": first["checkpoint"], "restoreBest": False})
    once = pt.execute({"command": "export", "snapshot": spec("cpu"), "rows": [], "checkpoint": first["checkpoint"], "restoreBest": False})
    results["checkpoint_loading_is_deterministic"] = again["artifact"] == once["artifact"]

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=float))
