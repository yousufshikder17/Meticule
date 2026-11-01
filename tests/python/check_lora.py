"""LoRA fine-tuning foundation checks (Phase 6B.1): real PEFT adapters on the tiny offline base. Nothing is mocked.

Prints one JSON object of named results and diagnostics. The reload checks use freshly built base models, and one of them
runs in a separate Python process that sees only the adapter bytes.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(PYTHON_DIR))
import lora_finetune as lf  # noqa: E402
import ml_runtime_common as common  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from peft.tuners.lora import LoraLayer  # noqa: E402

results, details = {}, {}


def refusal(fn):
    try:
        fn()
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"


# 1. The base model alone: a valid forward pass of the expected shape, on CPU, single-threaded, nothing downloaded.
base = lf.build_base()
data = lf.synthetic_sequences()
with torch.no_grad():
    base_logits = base(input_ids=data[:4]).logits
results["base_forward_pass_valid"] = tuple(base_logits.shape) == (4, lf.DATA_LENGTH, 32) and bool(torch.isfinite(base_logits).all())
results["cpu_only_single_threaded"] = all(p.device.type == "cpu" for p in base.parameters()) and torch.get_num_threads() == 1

# 2-7. Training: PEFT LoRA attached exactly to q_proj/v_proj, base frozen, adapters get gradients and change, loss falls.
model, cfg, report = lf.train()
details["report"] = {k: v for k, v in report.items() if k != "lossHistory"}
details["lossHistory"] = [round(v, 5) for v in report["lossHistory"][::20]] + [round(report["lossHistory"][-1], 5)]
lora_layers = {name: module for name, module in model.named_modules() if isinstance(module, LoraLayer)}
results["peft_model_with_lora_layers"] = isinstance(model, PeftModel) and model.active_adapter == "default" and len(lora_layers) == 4
results["lora_on_exactly_the_target_modules"] = sorted(name.split(".")[-1] for name in lora_layers) == ["q_proj", "q_proj", "v_proj", "v_proj"] \
    and all(module.r["default"] == 4 and module.scaling["default"] == 8.0 / 4 for module in lora_layers.values())
results["base_parameters_frozen"] = all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
# Each adapted 32x32 projection gains A (r x 32) and B (32 x r): 2 layers x 2 projections x 2 x 4 x 32 = 1024.
counts = report["parameters"]
results["trainable_count_is_exactly_the_adapters"] = counts["trainable"] == 1024 and counts["total"] == report["base"]["parameters"] + 1024 \
    and abs(counts["trainablePercent"] - 100 * 1024 / counts["total"]) < 1e-9
g = report["gradients"]
results["adapters_receive_finite_gradients"] = g["adapterTensorsWithGradient"] == g["adapterTensors"] == 8 and g["adapterGradientsFinite"] \
    and g["adapterGradientNormTotal"] > 0
results["frozen_base_receives_no_gradient"] = g["frozenTensorsWithGradient"] == 0
results["optimizer_updates_every_adapter_tensor"] = report["adapterTensorsChanged"] == 8
# Tensor-by-tensor in train(), and independently: removing the adapters leaves a base whose digest is the pre-training one.
results["frozen_base_weights_unchanged"] = report["frozenBaseUnchanged"] and lf.base_digest(model.unload()) == report["base"]["digest"]
model, _, _ = lf.train()  # unload() detached the adapters; train again for the artifact checks
results["training_loss_decreases"] = report["finalLoss"] < 0.1 * report["initialLoss"]

# 8. Export: adapter tensors only, far smaller than the base, with the base identity and LoRA settings in the metadata.
blob = lf.export_adapter(model, cfg, report["base"]["digest"])
tensors = safetensors.torch.load(blob)
metadata = lf.read_metadata(blob)
base_bytes = sum(p.numel() * 4 for p in base.parameters())
details["adapter"] = {"bytes": len(blob), "tensors": len(tensors), "baseBytes": base_bytes, "metadata": metadata}
results["export_holds_only_adapter_tensors"] = len(tensors) == 8 and all(".lora_A." in n or ".lora_B." in n for n in tensors) \
    and sum(t.numel() for t in tensors.values()) == 1024 and len(blob) < base_bytes / 4
results["export_records_base_identity_and_lora"] = metadata["base"] == {"id": lf.BASE_ID, "config": lf.BASE_CONFIG, "digest": report["base"]["digest"]} \
    and metadata["lora"] == {"rank": 4, "alpha": 8.0, "dropout": 0.0, "targetModules": ["q_proj", "v_proj"]} and metadata["format"] == lf.ADAPTER_FORMAT

# 9. Reload into a freshly built base, in this process and in a separate one: the adapted model's outputs are reproduced.
probe = lf.synthetic_sequences(count=32, seed=7)
expected = lf.logits(model, probe)
fresh_base = lf.build_base()
results["fresh_base_is_a_new_object_without_adapters"] = fresh_base is not model.get_base_model() and not any("lora_" in n for n, _ in fresh_base.named_parameters())
reloaded = lf.load_adapter(blob, fresh_base)
details["reload_max_abs_diff"] = float((lf.logits(reloaded, probe) - expected).abs().max())
results["reload_in_fresh_model_matches"] = details["reload_max_abs_diff"] <= 1e-6
with tempfile.TemporaryDirectory() as directory:
    adapter_path = Path(directory) / "adapter.safetensors"
    adapter_path.write_bytes(blob)
    child = ("import sys, json; sys.path.insert(0, sys.argv[1]); import lora_finetune as lf; "
             "m = lf.load_adapter(open(sys.argv[2], 'rb').read(), lf.build_base()); "
             "print(json.dumps(lf.logits(m, lf.synthetic_sequences(count=32, seed=7)).flatten().tolist()))")
    out = subprocess.run([sys.executable, "-I", "-c", child, str(PYTHON_DIR), str(adapter_path)], capture_output=True, text=True, timeout=300, check=True)
    other = torch.tensor(json.loads(out.stdout.strip().splitlines()[-1])).reshape(expected.shape)
details["cross_process_max_abs_diff"] = float((other - expected).abs().max())
results["reload_in_separate_process_matches"] = details["cross_process_max_abs_diff"] <= 1e-6
results["adapter_changes_the_base_output"] = float((lf.logits(lf.build_base(), probe) - expected).abs().max()) > 1e-2


def resave(changes=None, meta_changes=None, drop=None):
    """A modified adapter: same tensors and metadata unless changed. Used for the refusal checks."""
    t = {k: v for k, v in tensors.items() if k != drop}
    t.update(changes or {})
    m = {**metadata, **(meta_changes or {})}
    return safetensors.torch.save(t, metadata={"meticule": json.dumps(m, sort_keys=True)})


# 10. Incompatible: another base architecture, another base initialisation, a tampered LoRA config or library version.
first_a = next(n for n in tensors if ".lora_A." in n)
cases = {
    "other_architecture": (lambda: lf.load_adapter(blob, lf.build_base({**lf.BASE_CONFIG, "hidden_size": 48, "intermediate_size": 96})), "IncompatibleArtifact"),
    "other_base_weights": (lambda: lf.load_adapter(blob, lf.build_base(seed=lf.BASE_SEED + 1)), "IncompatibleArtifact"),
    "other_base_id": (lambda: lf.load_adapter(resave(meta_changes={"base": {**metadata["base"], "id": "gpt2"}}), lf.build_base()), "IncompatibleArtifact"),
    "rank_does_not_match_tensors": (lambda: lf.load_adapter(resave(meta_changes={"lora": {**metadata["lora"], "rank": 8}}), lf.build_base()), "IncompatibleArtifact"),
    "unsupported_target_in_metadata": (lambda: lf.load_adapter(resave(meta_changes={"lora": {**metadata["lora"], "targetModules": ["gate_proj"]}}), lf.build_base()), "IncompatibleArtifact"),
    "target_set_does_not_match_tensors": (lambda: lf.load_adapter(resave(meta_changes={"lora": {**metadata["lora"], "targetModules": ["q_proj", "k_proj"]}}), lf.build_base()), "IncompatibleArtifact"),
    "missing_adapter_tensor": (lambda: lf.load_adapter(resave(drop=first_a), lf.build_base()), "IncompatibleArtifact"),
    "extra_base_tensor": (lambda: lf.load_adapter(resave({"base_model.model.lm_head.weight": torch.zeros(32, 32)}), lf.build_base()), "IncompatibleArtifact"),
    "wrong_tensor_shape": (lambda: lf.load_adapter(resave({first_a: torch.zeros(5, 32)}), lf.build_base()), "IncompatibleArtifact"),
    "other_peft_minor": (lambda: lf.load_adapter(resave(meta_changes={"versions": {**metadata["versions"], "peft": "0.99.0"}}), lf.build_base()), "IncompatibleArtifact"),
    "other_format": (lambda: lf.load_adapter(resave(meta_changes={"format": "meticule-lora-adapter-v9"}), lf.build_base()), "IncompatibleArtifact"),
    # 11. Corrupted, non-finite, oversized or not bytes at all (a path is never opened).
    "truncated": (lambda: lf.load_adapter(blob[: len(blob) // 2], lf.build_base()), "IncompatibleArtifact"),
    "random_bytes": (lambda: lf.load_adapter(os.urandom(4096), lf.build_base()), "IncompatibleArtifact"),
    "header_length_lies": (lambda: lf.load_adapter(struct.pack("<Q", 2 ** 40) + blob[8:], lf.build_base()), "IncompatibleArtifact"),
    "no_metadata": (lambda: lf.load_adapter(safetensors.torch.save(tensors), lf.build_base()), "IncompatibleArtifact"),
    "nan_tensor": (lambda: lf.load_adapter(resave({first_a: torch.full_like(tensors[first_a], float("nan"))}), lf.build_base()), "IncompatibleArtifact"),
    "fp64_tensor": (lambda: lf.load_adapter(resave({first_a: tensors[first_a].double()}), lf.build_base()), "IncompatibleArtifact"),
    "oversized": (lambda: lf.load_adapter(blob + b"\0" * lf.LIMITS["maxAdapterBytes"], lf.build_base()), "IncompatibleArtifact"),
    "path_instead_of_bytes": (lambda: lf.load_adapter("../../adapter.safetensors", lf.build_base()), "IncompatibleArtifact"),
}
outcomes = {name: refusal(fn) for name, (fn, _) in cases.items()}
details["adapter_refusals"] = outcomes
results["incompatible_adapters_refused"] = all((outcomes[n] or "").startswith(k) for n, (_, k) in list(cases.items())[:11])
results["corrupted_adapters_refused"] = all((outcomes[n] or "").startswith(k) for n, (_, k) in list(cases.items())[11:])

# 12. Invalid configuration and data are refused before any training.
bad_configs = [{"rank": 0}, {"rank": 17}, {"rank": 4.0}, {"rank": True}, {"alpha": 0}, {"alpha": float("inf")}, {"alpha": 65},
               {"dropout": -0.1}, {"dropout": 0.6}, {"dropout": float("nan")}, {"learningRate": 0}, {"learningRate": float("nan")},
               {"learningRate": 0.5}, {"batchSize": 0}, {"batchSize": 33}, {"steps": 0}, {"steps": 501}, {"seed": -1}, {"seed": 2 ** 31},
               {"targetModules": []}, {"targetModules": ["gate_proj"]}, {"targetModules": ["lm_head"]}, {"targetModules": ["q_proj", "q_proj"]},
               {"targetModules": "q_proj"}, {"targetModules": [".*"]}, {"model": "gpt2"}, {"trust_remote_code": True}, {"base": "/models/x"}]
config_refusals = {json.dumps(c, default=str): refusal(lambda c=c: lf.resolve_config(c)) for c in bad_configs}
details["config_refusals"] = config_refusals
results["invalid_configurations_refused"] = all((m or "").startswith("ValueError") for m in config_refusals.values())
bad_data = {"out_of_vocab": [[0, 32]], "negative": [[0, -1]], "one_dimensional": [1, 2, 3], "ragged": [[1, 2], [3]], "floats": [[0.5, 1.0]],
            "too_long": [[0] * 33], "too_short": [[0]], "too_many": [[0, 1]] * 1025, "empty": []}
data_refusals = {name: refusal(lambda d=d: lf.validate_sequences(d, 32)) for name, d in bad_data.items()}
details["data_refusals"] = data_refusals
results["invalid_sequences_refused"] = all((m or "").startswith("ValueError") for m in data_refusals.values())
results["training_budget_enforced"] = "budget" in (refusal(lambda: lf.train({"steps": 500, "batchSize": 32}, torch.zeros(64, 32, dtype=torch.int64))) or "")

# 13. Determinism: the same seed gives a byte-identical adapter and loss history; another seed gives another adapter.
again_model, again_cfg, again_report = lf.train()
results["same_seed_byte_identical_adapter"] = lf.export_adapter(again_model, again_cfg, again_report["base"]["digest"]) == blob \
    and again_report["lossHistory"] == report["lossHistory"]
other_model, other_cfg, other_report = lf.train({"seed": 1})
results["other_seed_other_adapter"] = lf.export_adapter(other_model, other_cfg, other_report["base"]["digest"]) != blob \
    and other_report["base"]["digest"] == report["base"]["digest"]

# 14. Dropout and every allowed target train too, and nothing was downloaded into the private Hugging Face cache.
wide_model, _, wide = lf.train({"rank": 8, "alpha": 16, "dropout": 0.1, "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"], "steps": 60})
details["all_targets"] = {k: wide[k] for k in ("parameters", "initialLoss", "finalLoss")}
results["all_targets_with_dropout_train"] = wide["parameters"]["trainable"] == 2 * 4 * 2 * 8 * 32 and wide["finalLoss"] < wide["initialLoss"] and wide["frozenBaseUnchanged"]
results["offline_nothing_downloaded"] = os.environ.get("HF_HUB_OFFLINE") == "1" and not any(Path(lf._CACHE).rglob("*"))
results["adapter_error_is_contract_class"] = issubclass(common.IncompatibleArtifact, ValueError)

print(json.dumps({"results": results, "details": details}, allow_nan=False, default=str))
