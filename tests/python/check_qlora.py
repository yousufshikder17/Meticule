"""Real local NF4 training smoke plus explicitly labelled mocked preflight refusals. No downloads."""
import json
from pathlib import Path
import sys
import itertools
import time
import psutil
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import lora_finetune as lf
import lora_quantization as q
import lora_backend as lb
import ml_runtime_common as common
import torch


def refuses(call, error=ValueError):
    try:
        call()
    except error:
        return True
    return False


results = {}
results["defaults"] = q.settings({}) == q.DEFAULTS
for value in ({"quantization": "int8"}, {"computeDtype": "fp16"}, {"doubleQuantization": 1},
              {"quantization": "none", "doubleQuantization": True}, {"unknown": True}):
    results["invalid_" + str(value)] = refuses(lambda: q.settings(value))
cfg = q.settings({"quantization": "nf4", "doubleQuantization": True})
results["cpu_refused"] = refuses(lambda: q.preflight(cfg, "cpu", 1), common.RunnerError)
results["cpu_lora"] = q.preflight(q.DEFAULTS, "cpu", 1)["device"] == "cpu"
with patch.object(torch.cuda, "is_available", return_value=False):
    results["mock_missing_gpu"] = refuses(lambda: q.preflight(cfg, "cuda", 1), common.RunnerError)
with patch.object(q.importlib.metadata, "version", return_value="0.0.0"):
    results["mock_incompatible_dependency"] = refuses(q.dependencies, common.RunnerError)
results["invalid_device"] = refuses(lambda: q.preflight(cfg, "cuda:99", 1))
results["invalid_memory_estimate"] = refuses(lambda: q.preflight(cfg, "cpu", -1))
with patch.object(torch.cuda, "is_available", return_value=True), patch.object(q, "dependencies", return_value={}), \
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
        patch.object(torch.version, "cuda", "12.8"), patch.object(torch.cuda, "mem_get_info", return_value=(1024, 2048)):
    results["mock_insufficient_memory"] = refuses(lambda: q.preflight(cfg, "cuda", 1000), common.RunnerError)
    with patch.object(torch.cuda, "get_device_capability", return_value=(6, 0)):
        results["mock_unsupported_capability"] = refuses(lambda: q.preflight(cfg, "cuda", 1), common.RunnerError)
    with patch.object(torch.cuda, "is_bf16_supported", return_value=False):
        results["mock_bf16_unavailable"] = refuses(lambda: q.preflight({**cfg, "computeDtype": "bf16"}, "cuda", 1), common.RunnerError)
with patch.object(lb, "_execute", side_effect=torch.cuda.OutOfMemoryError("sensitive details")):
    try:
        lb.execute({"snapshot": {"hyperparameters": {}}})
    except common.RunnerError as error:
        results["mock_oom_classified_and_sanitized"] = error.code == "cuda_out_of_memory" and "sensitive" not in str(error)


def gpu_smoke(cfg):
    import bitsandbytes as bnb
    checked = {}
    started = time.perf_counter()
    evidence = q.preflight(cfg, "cuda", 256 * 1024 ** 2)
    torch.cuda.reset_peak_memory_stats()
    base = q.load_quantized(lf.build_base(), cfg)
    frozen = [(p, p.detach().cpu().clone()) for p in base.parameters()]
    model = lf.attach_lora(base, lf.resolve_config())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    checked["real_adapter_only_trainable"] = bool(trainable) and all("lora_" in n for n, _ in trainable)
    before = {n: p.detach().clone() for n, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.02, weight_decay=0.0)
    ids = lf.synthetic_sequences(count=4).cuda()
    model.train()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    checked["real_adapter_changed"] = any(".lora_B." in n and not torch.equal(before[n], p) for n, p in trainable)
    checked["real_frozen_base"] = all(not p.requires_grad and torch.equal(snapshot, p.detach().cpu()) for p, snapshot in frozen)
    layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    checked["real_quantized_layers"] = bool(layers) and all(m.weight.quant_state.quant_type == cfg["quantization"]
        and m.weight.quant_state.nested == cfg["doubleQuantization"]
        and m.compute_dtype == (torch.bfloat16 if cfg["computeDtype"] == "bf16" else torch.float32) for m in layers)
    model.eval()
    with torch.no_grad():
        checked["real_inference"] = bool(torch.isfinite(model(input_ids=ids).logits).all())
    memory = psutil.Process().memory_info()
    return checked, {"configuration": cfg, "preflight": evidence, "loss": float(loss.detach()), "seconds": time.perf_counter() - started,
                     "peakAllocatedBytes": torch.cuda.max_memory_allocated(), "peakReservedBytes": torch.cuda.max_memory_reserved(),
                     "peakProcessWorkingSetBytes": getattr(memory, "peak_wset", None)}


gpu = {"status": "NOT REQUESTED — CPU configuration checks only"}
if "--gpu" in sys.argv:
    combinations = list(itertools.product(("nf4", "fp4"), ("fp32", "bf16"), (False, True))) if "--matrix" in sys.argv else [("nf4", "bf16", True)]
    observations = []
    for kind, dtype, double in combinations:
        cfg = q.settings({"quantization": kind, "computeDtype": dtype, "doubleQuantization": double})
        checked, observed = gpu_smoke(cfg)
        results.update({f"{kind}_{dtype}_{double}_{name}": passed for name, passed in checked.items()})
        observations.append(observed)
        torch.cuda.empty_cache()
    gpu = {"status": "real CUDA quantized training and inference", "cases": observations}
print(json.dumps({"results": results, "gpu": gpu}))
if not all(results.values()):
    sys.exit(1)
