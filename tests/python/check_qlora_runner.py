"""Real GPU runner lifecycle, including fresh-process complete-state continuation and artifact inference."""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import lora_backend as lb
import safetensors.torch
import base64
import torch

rows = [{"seq": "0 1 2 3 4 5 6 7"} for _ in range(12)]
timings = {}
spec = {"algorithm": "lora_causal_lm", "backend": "peft", "seed": 7,
        "hyperparameters": {"epochs": 2, "batchSize": 2, "dropout": 0.1, "quantization": "nf4",
                            "doubleQuantization": True, "computeDtype": "bf16", "gradientAccumulationSteps": 2},
        "training": {"device": "cuda", "precision": "fp32"}, "evaluation": {"metrics": ["token_cross_entropy"]},
        "split": {"strategy": "random", "train": 0.5, "validation": 0.25, "test": 0.25, "stratify": False},
        "dataset": {"schema": {"columns": [{"name": "seq", "type": "tokens"}]}},
        "pipeline": {"definition": {"features": ["seq"], "steps": []}}}


def call(command, **extra):
    request = {"command": command, "contract": "meticule-ml-backend/v2", "task": "language_modeling",
               "snapshot": spec, "rows": rows, **extra}
    started = time.perf_counter()
    child = subprocess.run([sys.executable, str(Path(lb.__file__))], input=json.dumps(request), text=True,
                           capture_output=True, timeout=180)
    timings.setdefault(command, []).append(time.perf_counter() - started)
    if child.returncode:
        raise RuntimeError(child.stderr)
    print("completed " + command, file=sys.stderr, flush=True)
    return json.loads(child.stdout)


prepared = call("prepare")
reference = call("train_chunk", prepared=prepared, epochs=2, state=None)
first = call("train_chunk", prepared=prepared, epochs=1, state=None)
resumed = call("train_chunk", prepared=prepared, epochs=1, state=first["checkpoint"])
ref = safetensors.torch.load(base64.b64decode(reference["checkpoint"]))
actual = safetensors.torch.load(base64.b64decode(resumed["checkpoint"]))
results = {"fresh_process_state_identical": set(ref) == set(actual) and all(torch.equal(ref[k], actual[k]) for k in ref),
           "cuda_rng_durable": "rng.cuda" in actual,
           "real_qlora_metadata": resumed["environment"]["quantization"] == "nf4",
           "progress_continues": resumed["completedEpoch"] == 2}
artifact = call("export", checkpoint=resumed["checkpoint"], restoreBest=False)
try:
    lb.lf.load_adapter(base64.b64decode(artifact["artifact"]), lb.lf.build_base())
    results["legacy_dense_base_refuses_quantized_adapter"] = False
except lb.common.IncompatibleArtifact:
    results["legacy_dense_base_refuses_quantized_adapter"] = True
prediction = call("predict", artifact=artifact["artifact"], expectedEnvironment=artifact["environment"])
results["real_artifact_inference"] = len(prediction["predictions"]) == len(rows)
metadata, tensors = lb.read_checkpoint(first["checkpoint"])
metadata["environment"]["bitsandbytes"] = "0.0.0"
tampered = base64.b64encode(safetensors.torch.save(tensors, metadata={"meticule": json.dumps(metadata)})).decode()
try:
    lb.restore_run(tampered, spec, rows, prepared, lb.environment())
    results["incompatible_quantization_runtime_rejected"] = False
except lb.common.IncompatibleCheckpoint:
    results["incompatible_quantization_runtime_rejected"] = True
try:
    lb.restore_run(base64.b64encode(b"corrupt checkpoint").decode(), spec, rows, prepared, lb.environment())
    results["corrupted_quantized_checkpoint_rejected"] = False
except lb.common.InvalidCheckpoint:
    results["corrupted_quantized_checkpoint_rejected"] = True
print(json.dumps({"results": results, "environment": artifact["environment"], "performance": artifact["performance"],
                  "checkpointBytes": len(base64.b64decode(resumed["checkpoint"])), "commandWallSeconds": timings}))
if not all(results.values()):
    sys.exit(1)
