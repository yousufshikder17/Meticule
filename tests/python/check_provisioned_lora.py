"""Real offline provisioned model/adapter training and inference; hostile asset refusals."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import safetensors.torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_provisioning_fixtures import build
import lora_backend as lb
import ml_runtime_common as common
import transformers_manifest as manifest

results = {}
spec = {"algorithm": "lora_causal_lm", "backend": "peft", "seed": 7,
        "hyperparameters": {"baseModel": "provisioned-llama", "pretrainedModel": "approved-llama", "epochs": 2, "batchSize": 4},
        "training": {"device": "cpu", "precision": "fp32"}, "evaluation": {"metrics": ["token_cross_entropy"]},
        "split": {"strategy": "random", "train": 0.5, "validation": 0.25, "test": 0.25, "stratify": False},
        "dataset": {"schema": {"columns": [{"name": "seq", "type": "tokens"}]}},
        "pipeline": {"definition": {"features": ["seq"], "steps": []}}}
rows = [{"seq": "0 1 2 3 4 5 6 7"} for _ in range(12)]


def refused(call):
    try:
        call()
    except (common.RunnerError, common.IncompatibleArtifact, common.IncompatibleCheckpoint, ValueError):
        return True
    return False


with tempfile.TemporaryDirectory(prefix="meticule-provisioned-tests-") as root:
    directory = build(root)

    def call(command, snapshot=spec, **extra):
        return lb.execute({"command": command, "snapshot": snapshot, "rows": rows, "modelRoot": root, **extra})

    prepared = call("prepare")
    first = call("train_chunk", prepared=prepared, epochs=1, state=None)
    second = call("train_chunk", prepared=prepared, epochs=1, state=first["checkpoint"])
    artifact = call("export", checkpoint=second["checkpoint"], restoreBest=False)
    predictions = call("predict", artifact=artifact["artifact"])["predictions"]
    metadata = lb.lf.read_metadata(base64.b64decode(artifact["artifact"]))
    results["real_provisioned_train_resume_inference"] = len(predictions) == len(rows) and second["completedEpoch"] == 2
    results["model_and_tokenizer_identity"] = metadata["base"]["id"] == "approved-llama" and len(metadata["base"]["tokenizerDigest"]) == 64
    results["honest_generated_provenance"] = metadata["base"]["source"]["repository"] == "meticule-fixtures/generated-not-pretrained"
    for name in ("../approved-llama", "https://example.com/model", "a/b", "C:\\model"):
        invalid = {**spec, "hyperparameters": {**spec["hyperparameters"], "pretrainedModel": name}}
        results["name_refused_" + name] = refused(lambda: call("prepare", invalid))
    original_config = (directory / "config.json").read_bytes()
    original_tokenizer = (directory / "tokenizer.json").read_bytes()
    original_weights = (directory / "model.safetensors").read_bytes()
    for field, value in (("model_type", "gpt2"), ("hidden_size", 1000000), ("auto_map", {"AutoModel": "evil.py"}),
                         ("__dict__", {"hidden_size": 1000000}), ("quantization_config", {"load_in_8bit": True}),
                         ("bos_token_id", 32), ("rope_scaling", {"type": "yarn"})):
        raw = json.loads(original_config)
        raw[field] = value
        (directory / "config.json").write_text(json.dumps(raw))
        manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
        results["config_refused_" + field] = refused(lambda: call("prepare"))
    raw = json.loads(original_config)
    raw["num_hidden_layers"] = 8
    (directory / "config.json").write_text(json.dumps(raw))
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    large_checkpoint = {**spec, "hyperparameters": {**spec["hyperparameters"], "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"]}}
    results["checkpoint_tensor_budget"] = refused(lambda: call("prepare", large_checkpoint))
    (directory / "config.json").write_bytes(original_config)
    # A token vocabulary that no longer matches the embedding table, even with updated operator hashes.
    tokenizer = json.loads(original_tokenizer)
    tokenizer["model"]["vocab"].pop("t31")
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer))
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    results["mismatched_tokenizer"] = refused(lambda: call("prepare"))
    (directory / "tokenizer.json").write_bytes(original_tokenizer)
    manifest.write_manifest(str(directory), "causal-lm", "llama", "meticule-fixtures/generated-not-pretrained", "0" * 40)
    results["same_provisioned_assets_reload"] = call("predict", artifact=artifact["artifact"])["predictions"] == predictions
    # Correctly hashed replacement tokenizer is a different base identity: an old adapter cannot attach to it.
    tokenizer = json.loads(original_tokenizer)
    tokenizer["model"]["vocab"]["t1"], tokenizer["model"]["vocab"]["t2"] = 2, 1
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer))
    manifest.write_manifest(str(directory), "causal-lm", "llama", "meticule-fixtures/generated-not-pretrained", "0" * 40)
    results["adapter_tokenizer_identity_mismatch"] = refused(lambda: call("predict", artifact=artifact["artifact"]))
    (directory / "tokenizer.json").write_bytes(original_tokenizer)
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    tensors = safetensors.torch.load(original_weights)
    tensors[next(iter(tensors))].reshape(-1)[0] = float("nan")
    safetensors.torch.save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    results["nonfinite_weights_with_valid_manifest"] = refused(lambda: call("prepare"))
    tensors = safetensors.torch.load(original_weights)
    key = next(iter(tensors))
    tensors[key] = tensors[key].to(dtype=lb.torch.int64)
    safetensors.torch.save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    results["nonfloating_weights_with_valid_manifest"] = refused(lambda: call("prepare"))
    tensors = safetensors.torch.load(original_weights)
    key = next(iter(tensors))
    tensors[key] = tensors[key].reshape(-1)[:1].contiguous()
    safetensors.torch.save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    results["mismatched_weight_shapes_with_valid_manifest"] = refused(lambda: call("train_chunk", prepared=prepared, epochs=1, state=None))
    (directory / "model.safetensors").write_bytes(original_weights)
    manifest.write_manifest(str(directory), "causal-lm", "llama", "fixture", "0" * 40)
    (directory / "model.safetensors").write_bytes(b"corrupt")
    results["corrupt_weights_hash"] = refused(lambda: call("prepare"))
    results["missing_model"] = refused(lambda: lb.execute({"command": "prepare", "snapshot": spec, "rows": rows, "modelRoot": str(Path(root) / "absent")}))
print(json.dumps({"results": results, "evidence": "real CPU training and inference on generated local safetensors assets"}))
if not all(results.values()):
    sys.exit(1)
