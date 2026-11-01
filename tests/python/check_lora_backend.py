"""PEFT LoRA runner checks (Phase 6B.2): real PEFT training through the runner commands, checkpoint and resume equivalence
(including across real runner processes), optimizer and RNG restoration with negative controls, and every refusal.

Prints one JSON object of named results and diagnostics. Nothing is mocked; the one fault injection (a non-finite adapter
weight, to reach the divergence guard) is labelled where it happens.
"""
import base64
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(PYTHON_DIR))
import lora_backend as lb  # noqa: E402
import lora_finetune as lf  # noqa: E402
import ml_runtime_common as common  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402

results, details = {}, {}
UUID = "00000000-0000-4000-8000-000000000000"
N, K = 6, 2
# Counting sequences through 8 tokens, lengths 8-16, so batches really pad and the mask really matters.
rows = [{"seq": " ".join(str((start + j) % 8) for j in range(8 + (i % 9)))} for i, start in enumerate([(i * 5) % 8 for i in range(120)])]


def spec(seed=3, device="cpu", **hp):
    hyper = {"rank": 4, "alpha": 8, "dropout": 0.1, "targetModules": ["q_proj", "v_proj"], "learningRate": 0.02, "batchSize": 16, "epochs": N, **hp}
    return {"algorithm": "lora_causal_lm", "backend": "peft", "seed": seed, "evaluation": {"metrics": ["token_cross_entropy", "perplexity", "next_token_accuracy"]},
            "training": {"checkpointEveryEpochs": K, "earlyStopping": None, "device": device, "precision": "fp32"}, "hyperparameters": hyper,
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False},
            "dataset": {"schema": {"columns": [{"name": "seq", "type": "tokens"}]}},
            "pipeline": {"definition": {"features": ["seq"], "steps": []}}}


def run(command, s, data=rows, **extra):
    return lb.execute({"command": command, "snapshot": s, "rows": data, **extra})


def chunk(s, epochs, state=None, data=rows, prep=None):
    return run("train_chunk", s, data, prepared=prep or prepared, epochs=epochs, state=state)


def subprocess_chunk(s, epochs, state):
    """One train_chunk in a fresh `python lora_backend.py` process, through the real stdin/stdout protocol."""
    envelope = {"command": "train_chunk", "contract": common.ML_BACKEND_CONTRACT_V2, "task": "language_modeling", "snapshot": s, "rows": rows,
                "prepared": prepared, "epochs": epochs, "state": state}
    # Spawned as src/ml/runner-process.ts spawns runners: `python <script>`, so the runner directory is importable.
    out = subprocess.run([sys.executable, str(PYTHON_DIR / "lora_backend.py")], input=json.dumps(envelope), capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError("runner failed: " + out.stderr.strip().splitlines()[-1])
    return json.loads(out.stdout)


def refusal(fn):
    try:
        fn()
        return None
    except common.RunnerError as error:
        return f"RunnerError({error.code})"
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def tensors_of(checkpoint):
    return safetensors.torch.load(base64.b64decode(checkpoint))


def meta_of(checkpoint):
    return lb.read_checkpoint(checkpoint)[0]


def diff(a, b, prefix):
    """Max absolute difference over every tensor whose name starts with `prefix` (both sides must have the same names)."""
    ta, tb = tensors_of(a), tensors_of(b)
    names = sorted(n for n in ta if n.startswith(prefix))
    if names != sorted(n for n in tb if n.startswith(prefix)):
        return float("inf")
    return max(float((ta[n].double() - tb[n].double()).abs().max()) for n in names)


def rewrite(checkpoint, meta_changes=None, tensor_changes=None, drop=()):
    """A modified checkpoint written the same way the runner writes one. Used only for the refusal checks."""
    meta, tensors = lb.read_checkpoint(checkpoint)
    tensors = {k: v for k, v in tensors.items() if k not in drop}
    tensors.update(tensor_changes or {})
    blob = safetensors.torch.save(tensors, metadata={"meticule": json.dumps({**meta, **(meta_changes or {})}, sort_keys=True)})
    return base64.b64encode(blob).decode()


s = spec()
prepared = run("prepare", s)
per_epoch = math.ceil(len(prepared["indices"]["train"]) / 16)
results["prepare_validates_and_splits"] = sorted(len(v) for v in prepared["indices"].values()) == [24, 24, 72] and prepared["environment"]["backend"] == "peft"

# 1. Uninterrupted N epochs in this process; K epochs, then N-K in two separate runner processes from the checkpoint bytes.
whole = chunk(s, N)
first = subprocess_chunk(s, K, None)
rest = subprocess_chunk(s, N - K, first["checkpoint"])
details["resume"] = {"adapterMaxDiff": diff(whole["checkpoint"], rest["checkpoint"], "adapter."),
                     "optimizerMaxDiff": diff(whole["checkpoint"], rest["checkpoint"], "optimizer."),
                     "rngTorchEqual": torch.equal(tensors_of(whole["checkpoint"])["rng.torch"], tensors_of(rest["checkpoint"])["rng.torch"]),
                     "finalTrainLoss": [meta_of(whole["checkpoint"])["lastTrainLoss"], meta_of(rest["checkpoint"])["lastTrainLoss"]],
                     "optimizerSteps": [meta_of(whole["checkpoint"])["optimizerSteps"], meta_of(rest["checkpoint"])["optimizerSteps"]]}
results["new_process_resume_adapter_bit_identical"] = details["resume"]["adapterMaxDiff"] == 0.0
results["new_process_resume_optimizer_state_bit_identical"] = details["resume"]["optimizerMaxDiff"] == 0.0
results["new_process_resume_rng_states_identical"] = all(torch.equal(tensors_of(whole["checkpoint"])[k], tensors_of(rest["checkpoint"])[k]) for k in ("rng.torch", "rng.sampler", "rng.numpy")) \
    and meta_of(whole["checkpoint"])["rng"] == meta_of(rest["checkpoint"])["rng"]
results["new_process_resume_progress_identical"] = (
    rest["completedEpoch"] == whole["completedEpoch"] == N and [e["epoch"] for e in rest["epochs"]] == list(range(K + 1, N + 1))
    and details["resume"]["optimizerSteps"] == [N * per_epoch] * 2 and len(set(details["resume"]["finalTrainLoss"])) == 1
    and [(e["trainLoss"], e["validationLoss"]) for e in first["epochs"] + rest["epochs"]] == [(e["trainLoss"], e["validationLoss"]) for e in whole["epochs"]])
exports = [run("export", s, [], checkpoint=c["checkpoint"], restoreBest=False) for c in (whole, rest)]
probe = rows[:16]
predictions = [run("predict", s, probe, artifact=e["artifact"])["predictions"] for e in exports]
results["resumed_and_uninterrupted_predict_identically"] = predictions[0] == predictions[1] and exports[0]["performance"] == {**exports[1]["performance"], "fitSeconds": exports[0]["performance"]["fitSeconds"]}

# 2. Three chunks (2 + 2 + 2) equal the uninterrupted run; training learned the controlled workload.
a = chunk(s, 2)
b = chunk(s, 2, a["checkpoint"])
c = chunk(s, 2, b["checkpoint"])
results["three_chunks_equal_uninterrupted"] = diff(c["checkpoint"], whole["checkpoint"], "adapter.") == 0.0 and diff(c["checkpoint"], whole["checkpoint"], "optimizer.") == 0.0
details["losses"] = [round(e["trainLoss"], 4) for e in whole["epochs"]]
results["training_loss_decreases"] = whole["epochs"][-1]["trainLoss"] < 0.5 * meta_of(whole["checkpoint"])["initialTrainLoss"]

# 3. Optimizer state and RNG state are genuinely restored: discarding either changes the continuation (negative controls).
first_tensors = tensors_of(first["checkpoint"])
zeroed = {k: torch.zeros_like(v) for k, v in first_tensors.items() if k.endswith((".exp_avg", ".exp_avg_sq"))}
no_moments = chunk(s, N - K, rewrite(first["checkpoint"], tensor_changes=zeroed))
details["negative_controls"] = {"adapterOnlyMaxDiff": diff(no_moments["checkpoint"], whole["checkpoint"], "adapter.")}
other_rng = torch.Generator().manual_seed(12345).get_state()
reseeded = chunk(s, N - K, rewrite(first["checkpoint"], tensor_changes={"rng.torch": other_rng}))
details["negative_controls"]["otherDropoutRngMaxDiff"] = diff(reseeded["checkpoint"], whole["checkpoint"], "adapter.")
results["discarding_adamw_moments_changes_the_continuation"] = details["negative_controls"]["adapterOnlyMaxDiff"] > 1e-4
results["discarding_dropout_rng_changes_the_continuation"] = details["negative_controls"]["otherDropoutRngMaxDiff"] > 1e-6
restored = lb.restore_run(first["checkpoint"], s, rows, {"indices": prepared["indices"]}, lb.environment())
results["optimizer_step_counters_restored"] = all(float(st["step"]) == K * per_epoch for st in restored.optimizer.state_dict()["state"].values()) \
    and restored.steps == K * per_epoch and restored.completed == K

# 4. The base stays frozen and only adapters train; the checkpoint never carries base weights.
results["base_frozen_and_never_checkpointed"] = all(not p.requires_grad for n, p in restored.model.named_parameters() if "lora_" not in n) \
    and lf.base_digest(restored.model.unload()) == meta_of(first["checkpoint"])["base"]["digest"] \
    and all(k.startswith(("adapter.", "optimizer.", "rng.", "best.")) for k in first_tensors) \
    and all(".lora_A." in k or ".lora_B." in k for k in first_tensors if k.startswith(("adapter.", "best.")))

# 5. Evaluation and the adapter artifact: language-modelling metrics per partition; adapter-only bytes marked as an adapter.
evaluation = run("evaluate", s, artifact=exports[0]["artifact"], prepared={"indices": prepared["indices"]})
details["metrics"] = {f"{m['partition']}/{m['name']}": round(m["value"], 5) for m in evaluation["metrics"]}
results["evaluates_lm_metrics_per_partition"] = len(evaluation["metrics"]) == 9 and all(math.isfinite(m["value"]) for m in evaluation["metrics"]) \
    and all(math.isclose(details["metrics"][f"{p}/perplexity"], round(math.exp(next(m["value"] for m in evaluation["metrics"] if m["partition"] == p and m["name"] == "token_cross_entropy")), 5), rel_tol=1e-4) for p in ("train", "validation", "test"))
env = exports[0]["environment"]
artifact_meta = lf.read_metadata(base64.b64decode(exports[0]["artifact"]))
results["artifact_is_marked_adapter_on_its_base"] = env["artifact_kind"] == "lora_adapter" and env["base_model"] == lf.BASE_ID \
    and env["base_model_digest"] == artifact_meta["base"]["digest"] and exports[0]["format"] == "meticule-lora-adapter-v1" \
    and artifact_meta["training"]["optimizerSteps"] == N * per_epoch and len(base64.b64decode(exports[0]["artifact"])) < 64 * 1024
results["predictions_follow_the_counting_rule"] = sum(p[i] == (int(r["seq"].split(" ")[i]) + 1) % 8 for p, r in zip(predictions[0], probe) for i in range(len(p))) \
    / sum(len(p) for p in predictions[0]) > 0.9
# The pre-export model (the restored final run) and the exported adapter on a fresh base give the same predictions.
final = lb.restore_run(whole["checkpoint"], s, rows, {"indices": prepared["indices"]}, lb.environment())
results["exported_adapter_matches_pre_export_model"] = lb.predict_next_tokens(final.model, lb.sequences(s, probe, final.hp)) == predictions[0] \
    and torch.equal(lb.token_scores(final.model, final.train)[0] * torch.ones(1), lb.token_scores(lb.load_artifact(exports[0]["artifact"], s)[0], final.train)[0] * torch.ones(1))
best_export = run("export", s, [], checkpoint=whole["checkpoint"], restoreBest=True)
best_epoch = meta_of(whole["checkpoint"])["best"]["epoch"]
results["export_restores_best_epoch_adapter"] = best_export["performance"]["bestEpoch"] == best_epoch and \
    all(torch.equal(v, tensors_of(whole["checkpoint"])[f"best.{k}"]) for k, v in safetensors.torch.load(base64.b64decode(best_export["artifact"])).items())

# 6. Refusals. Checkpoints: corrupted, incompatible, inconsistent; none restarts training from zero.
ck = first["checkpoint"]
meta, tensors = lb.read_checkpoint(ck)
adapter_key = next(k for k in tensors if k.startswith("adapter."))
cases = {
    "truncated": (lambda: chunk(s, 1, ck[: len(ck) // 2]), "InvalidCheckpoint"),
    "not_base64": (lambda: chunk(s, 1, "@@@@"), "InvalidCheckpoint"),
    "header_length_lies": (lambda: chunk(s, 1, base64.b64encode(struct.pack("<Q", 2 ** 40) + base64.b64decode(ck)[8:]).decode()), "InvalidCheckpoint"),
    "oversized": (lambda: chunk(s, 1, "A" * (lb.LIMITS["maxCheckpointBytes"] * 2)), "InvalidCheckpoint"),
    "adapter_artifact_as_checkpoint": (lambda: chunk(s, 1, exports[0]["artifact"]), "IncompatibleCheckpoint"),
    "other_format": (lambda: chunk(s, 1, rewrite(ck, {"format": "pytorch-checkpoint-v1"})), "IncompatibleCheckpoint"),
    "other_schema": (lambda: chunk(s, 1, rewrite(ck, {"schema": 2})), "IncompatibleCheckpoint"),
    "other_runtime": (lambda: chunk(s, 1, rewrite(ck, {"environment": {**meta["environment"], "peft": "0.0.1"}})), "IncompatibleCheckpoint"),
    "other_rank": (lambda: chunk(spec(rank=8), 1, ck), "IncompatibleCheckpoint"),
    "other_targets": (lambda: chunk(spec(targetModules=["q_proj", "k_proj"]), 1, ck), "IncompatibleCheckpoint"),
    "other_seed": (lambda: chunk(spec(seed=4), 1, ck), "IncompatibleCheckpoint"),
    "other_data": (lambda: chunk(s, 1, ck, data=[{"seq": "1 2 3 4"}] + rows[1:]), "IncompatibleCheckpoint"),
    "other_split": (lambda: chunk(s, 1, ck, prep={"indices": {**prepared["indices"], "train": prepared["indices"]["train"][1:]}}), "IncompatibleCheckpoint"),
    "other_base_digest": (lambda: chunk(s, 1, rewrite(ck, {"base": {**meta["base"], "digest": "0" * 64}})), "IncompatibleCheckpoint"),
    "other_learning_rate_in_optimizer": (lambda: chunk(s, 1, rewrite(ck, {"optimizer": {**meta["optimizer"], "lr": 0.05}})), "IncompatibleCheckpoint"),
    "mixed_precision": (lambda: chunk(s, 1, rewrite(ck, {"precision": "mixed"})), "IncompatibleCheckpoint"),
    "wrong_adapter_shape": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={adapter_key: torch.zeros(5, 32)})), "IncompatibleCheckpoint"),
    "missing_optimizer_tensor": (lambda: chunk(s, 1, rewrite(ck, drop=("optimizer.0.exp_avg",))), "InvalidCheckpoint"),
    "extra_base_weight": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={"adapter.base_model.model.lm_head.weight": torch.zeros(32, 32)})), "InvalidCheckpoint"),
    "nan_adapter": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={adapter_key: torch.full_like(tensors[adapter_key], float("nan"))})), "InvalidCheckpoint"),
    "negative_second_moment": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={"optimizer.0.exp_avg_sq": -torch.ones_like(tensors["optimizer.0.exp_avg_sq"])})), "InvalidCheckpoint"),
    "step_counter_disagrees": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={"optimizer.0.step": torch.tensor(99.0)})), "InvalidCheckpoint"),
    "epoch_inconsistent_with_steps": (lambda: chunk(s, 1, rewrite(ck, {"completedEpoch": K + 1})), "InvalidCheckpoint"),
    "epoch_beyond_request": (lambda: chunk(s, 1, rewrite(ck, {"completedEpoch": N + 1})), "InvalidCheckpoint"),
    "malformed_python_rng": (lambda: chunk(s, 1, rewrite(ck, {"rng": {**meta["rng"], "python": [3, [1, 2], None]}})), "InvalidCheckpoint"),
    "wrong_rng_dtype": (lambda: chunk(s, 1, rewrite(ck, tensor_changes={"rng.torch": tensors["rng.torch"].float()})), "InvalidCheckpoint"),
    "best_without_tensors": (lambda: chunk(s, 1, rewrite(ck, {"best": None})), "InvalidCheckpoint"),
    "missing_metadata_field": (lambda: chunk(s, 1, rewrite(ck, {"optimizerSteps": None})), "InvalidCheckpoint"),
    "chunk_past_requested_epochs": (lambda: chunk(s, N - K + 1, ck), "ValueError"),
    "cuda_resume": (lambda: chunk(spec(device="cuda"), 1, ck), "RunnerError(cuda_unavailable)"),
}
outcomes = {name: refusal(fn) for name, (fn, _) in cases.items()}
details["checkpoint_refusals"] = outcomes
results["every_bad_checkpoint_refused_with_its_class"] = all((outcomes[n] or "").startswith(k) for n, (_, k) in cases.items())

# Artifacts: the adapter loader refuses foreign or mismatched artifacts, and a checkpoint is not an artifact.
artifact = exports[0]["artifact"]
artifact_cases = {
    "checkpoint_as_artifact": lambda: run("predict", s, probe, artifact=ck),
    "other_rank_snapshot": lambda: run("predict", spec(rank=8), probe, artifact=artifact),
    "garbage": lambda: run("predict", s, probe, artifact=base64.b64encode(b"not an adapter").decode()),
    "oversized": lambda: run("predict", s, probe, artifact="A" * (lf.LIMITS["maxAdapterBytes"] * 2)),
    "foreign_runtime_environment": lambda: run("predict", s, probe, artifact=artifact, expectedEnvironment={**env, "peft": "0.0.1"}),
    "transformers_reference_environment": lambda: run("predict", s, probe, artifact=artifact, expectedEnvironment={**env, "backend": "transformers", "artifact_format": "transformers-reference-v1"}),
}
artifact_outcomes = {name: refusal(fn) for name, fn in artifact_cases.items()}
details["artifact_refusals"] = artifact_outcomes
results["foreign_or_mismatched_artifacts_refused"] = all((v or "").startswith("IncompatibleArtifact") for v in artifact_outcomes.values())

# Configuration, data and device: refused before any training.
bad_hp = [{"rank": 0}, {"rank": 17}, {"alpha": 0}, {"alpha": float("inf")}, {"dropout": 0.6}, {"learningRate": float("nan")}, {"learningRate": 0.5},
          {"batchSize": 33}, {"epochs": 0}, {"epochs": 101}, {"maxSequenceLength": 1}, {"maxSequenceLength": 33}, {"targetModules": ["gate_proj"]},
          {"targetModules": []}, {"baseModel": "gpt2"}, {"trust_remote_code": True}, {"modelPath": "../x"}]
details["config_refusals"] = {json.dumps(h, default=str): refusal(lambda h=h: run("prepare", spec(**h))) for h in bad_hp}
results["invalid_configurations_refused"] = all((v or "").startswith("ValueError") for v in details["config_refusals"].values())
bad_rows = {"out_of_vocab": [{"seq": "1 32"}], "too_long": [{"seq": " ".join(["1"] * 33)}], "single_token": [{"seq": "5"}], "not_ids": [{"seq": "a b"}],
            "double_space": [{"seq": "1  2"}], "number_cell": [{"seq": 12}], "leading_zero": [{"seq": "01 2"}]}
details["data_refusals"] = {name: refusal(lambda d=d: run("prepare", s, d + rows[:9])) for name, d in bad_rows.items()}
results["invalid_token_data_refused"] = all(v == "RunnerError(invalid_tokens)" for v in details["data_refusals"].values())
results["shorter_max_sequence_length_enforced"] = refusal(lambda: run("prepare", spec(maxSequenceLength=8))) == "RunnerError(invalid_tokens)"
results["step_budget_enforced"] = refusal(lambda: run("train_chunk", spec(epochs=100, batchSize=1), prepared=prepared, epochs=1, state=None)) == "RunnerError(invalid_tokens)"
results["cuda_and_mixed_refused"] = refusal(lambda: run("prepare", spec(device="cuda"))) == "RunnerError(cuda_unavailable)" and \
    refusal(lambda: run("prepare", {**s, "training": {**s["training"], "precision": "mixed"}})) == "RunnerError(unsupported_device_or_precision)"
results["target_or_preprocessing_refused"] = (refusal(lambda: run("prepare", {**s, "dataset": {"schema": {**s["dataset"]["schema"], "target": "seq"}}})) or "").startswith("ValueError") \
    and (refusal(lambda: run("prepare", {**s, "pipeline": {"definition": {"features": ["seq"], "steps": [{"operation": "standard_scale"}]}}})) or "").startswith("ValueError")

# Divergence: a non-finite loss stops training with an error and produces no checkpoint (fault injected: one adapter weight
# set to infinity in memory, since bounded hyperparameters cannot diverge this workload reliably).
broken = lb.fresh_run(s, rows, {"indices": prepared["indices"]})
next(p for n, p in broken.model.named_parameters() if ".lora_B." in n).data.fill_(float("inf"))
results["divergence_refused"] = "diverged" in (refusal(lambda: lb.run_epochs(broken, 1)) or "")

# 7. Determinism and contract: the same snapshot gives identical checkpoint tensors and metadata (wall-clock time aside).
once, twice = chunk(s, 1)["checkpoint"], chunk(s, 1)["checkpoint"]
timeless = lambda m: {k: v for k, v in m.items() if k != "elapsedSeconds"}  # noqa: E731
results["same_seed_identical_checkpoint"] = timeless(meta_of(once)) == timeless(meta_of(twice)) and diff(once, twice, "") == 0.0 \
    and set(tensors_of(once)) == set(tensors_of(twice))
results["runner_declares_language_modeling"] = lb.MODELS == {"lora_causal_lm": "language_modeling"} and "language_modeling" in common.TASKS \
    and common.requested_metrics({}, "language_modeling", labelled=False) == ["token_cross_entropy", "perplexity"]

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=str))
