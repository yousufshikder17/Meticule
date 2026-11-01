"""Real CPU gradient equivalence and checkpoint RNG validation; faults are explicitly injected."""
import base64
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import lora_backend as lb
import safetensors.torch
import torch

torch.set_num_threads(1)
results = {}
seqs = [[1, 2], [2, 3, 4, 5, 6], [3, 4, 5], [4, 5, 6, 7], [5, 6]]


def gradients(batch, accumulation):
    base = lb.lf.build_base()
    model = lb.lf.attach_lora(base, {"rank": 4, "alpha": 8, "dropout": 0, "targetModules": ["q_proj", "v_proj"], "seed": 7})
    captured = []
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0)
    original = optimizer.step

    def step():
        captured.append(torch.cat([p.grad.flatten().clone() for p in model.parameters() if p.requires_grad]))
        return original()

    optimizer.step = step
    run = SimpleNamespace(train=seqs, validation=[], hp={"batchSize": batch, "gradientAccumulationSteps": accumulation},
                          sampler=torch.Generator().manual_seed(4), model=model, device="cpu", optimizer=optimizer,
                          steps=0, completed=0, elapsed=0, best=None, distillation=None)
    lb.run_epochs(run, 1)
    return captured


for batch, accumulation, effective in [(1, 5, 5), (2, 3, 6), (1, 3, 3), (2, 2, 4)]:
    combined, accumulated = gradients(effective, 1), gradients(batch, accumulation)
    print(f"gradient {batch}/{accumulation}: max absolute differences {[float((a-b).abs().max()) for a,b in zip(combined, accumulated)]}", file=sys.stderr)
    results[f"gradient_batch_{batch}_accumulation_{accumulation}"] = len(combined) == len(accumulated) and all(
        torch.allclose(a, b, atol=5e-7, rtol=2e-5) for a, b in zip(combined, accumulated))

s = {"algorithm": "lora_causal_lm", "backend": "peft", "seed": 7, "hyperparameters": {"epochs": 2, "batchSize": 2},
     "training": {"device": "cpu", "precision": "fp32"},
     "dataset": {"schema": {"columns": [{"name": "seq", "type": "tokens"}]}},
     "pipeline": {"definition": {"features": ["seq"], "steps": []}}}
rows = [{"seq": " ".join(map(str, seq))} for seq in seqs]
prepared = {"indices": {"train": list(range(5)), "validation": []}}
run = lb.fresh_run(s, rows, prepared)
lb.run_epochs(run, 1)
env = lb.environment()
checkpoint = lb.save_run(run, env)
meta, tensors = lb.read_checkpoint(checkpoint)


def encoded(state):
    changed = {**meta, "rng": {**meta["rng"], "python": state}}
    return base64.b64encode(safetensors.torch.save(tensors, metadata={"meticule": json.dumps(changed)})).decode()


for index in [-1, 625, 2**32 - 1, True, 1.5]:
    state = [3, [*meta["rng"]["python"][1][:-1], index], None]
    try:
        lb.validated_checkpoint(encoded(state), s, env, run)
        results[f"reject_index_{index}"] = False
    except lb.common.InvalidCheckpoint:
        results[f"reject_index_{index}"] = True
for i, state in enumerate([None, [], [3, [0], None], [3, [True] * 624 + [0], None], [3, [0] * 625, "bad"], [True, [0] * 625, None]]):
    try:
        lb.validated_checkpoint(encoded(state), s, env, run)
        results[f"reject_structure_{i}"] = False
    except lb.common.InvalidCheckpoint:
        results[f"reject_structure_{i}"] = True
for index in [0, 624]:
    lb.validated_checkpoint(encoded([3, [*meta["rng"]["python"][1][:-1], index], None]), s, env, run)
    results[f"accept_index_{index}"] = True

before = {k: v.clone() for k, v in run.adapter_state().items()}
global_rng = random.getstate()
with patch.object(random.Random, "setstate", side_effect=ValueError("injected restore failure")):
    try:
        lb.validated_checkpoint(checkpoint, s, env, run)
        results["restoration_failure_classified"] = False
    except lb.common.InvalidCheckpoint:
        results["restoration_failure_classified"] = True
results["failure_does_not_apply_state"] = global_rng == random.getstate() and all(torch.equal(v, run.adapter_state()[k]) for k, v in before.items())
with patch.object(random, "setstate", side_effect=ValueError("injected global restore failure")), patch.object(lb, "load_adapter_tensors") as assign:
    try:
        lb.restore_run(checkpoint, s, rows, prepared, env)
        results["global_restoration_failure_before_assignment"] = False
    except lb.common.InvalidCheckpoint:
        results["global_restoration_failure_before_assignment"] = not assign.called
print(json.dumps({"results": results}))
assert all(results.values()), results
