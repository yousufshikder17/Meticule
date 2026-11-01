"""Bounded Llama assets through Meticule's existing operator-provisioned manifest boundary."""
import contextlib
import os
import tempfile
import math
import struct

from safetensors import safe_open

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

import ml_runtime_common as common
import transformers_manifest as manifest
import lora_quantization as quant

BASE_ID = "provisioned-llama"
MAX_PARAMETERS = 32_000_000
MAX_ADAPTER_BYTES = 512 * 1024


def configuration(raw, hp):
    bounds = {"vocab_size": (2, 65536), "hidden_size": (8, 512), "intermediate_size": (8, 2048),
              "num_hidden_layers": (1, 8), "num_attention_heads": (1, 16), "max_position_embeddings": (32, 8192)}
    for key, (low, high) in bounds.items():
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise manifest.mismatch()
    optional = {"num_key_value_heads", "tie_word_embeddings", "initializer_range", "rms_norm_eps", "attention_bias", "mlp_bias",
                "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id", "use_cache", "hidden_act", "head_dim"}
    metadata = {"model_type", "architectures", "transformers_version", "dtype", "torch_dtype", "_name_or_path",
                "rope_parameters", "rope_scaling", "rope_theta", "pretraining_tp"}
    if set(raw) - set(bounds) - optional - metadata or raw.get("pretraining_tp", 1) != 1 or raw.get("rope_scaling") is not None:
        raise manifest.mismatch()
    for key in ("tie_word_embeddings", "attention_bias", "mlp_bias", "use_cache"):
        if key in raw and not isinstance(raw[key], bool):
            raise manifest.mismatch()
    for key in ("initializer_range", "rms_norm_eps", "attention_dropout"):
        if key in raw and (not isinstance(raw[key], (int, float)) or isinstance(raw[key], bool)
                           or not math.isfinite(raw[key]) or not 0 <= raw[key] <= 1):
            raise manifest.mismatch()
    rope = raw.get("rope_parameters") or {"rope_type": "default", "rope_theta": raw.get("rope_theta", 10000.0)}
    if not isinstance(rope, dict) or set(rope) - {"rope_type", "rope_theta"} or rope.get("rope_type") != "default" \
            or not isinstance(rope.get("rope_theta"), (int, float)) or isinstance(rope.get("rope_theta"), bool) \
            or not math.isfinite(rope["rope_theta"]) or not 1 <= rope["rope_theta"] <= 1e9:
        raise manifest.mismatch()
    hidden, heads = raw["hidden_size"], raw["num_attention_heads"]
    kv = raw.get("num_key_value_heads", heads)
    if not isinstance(kv, int) or isinstance(kv, bool) or not 1 <= kv <= heads or heads % kv or hidden % heads \
            or raw.get("architectures") != ["LlamaForCausalLM"] or raw.get("model_type") != "llama" \
            or raw.get("hidden_act", "silu") != "silu" or raw.get("head_dim", hidden // heads) != hidden // heads \
            or (hidden // heads) % 2 or any(key in raw for key in ("quantization_config", "auto_map", "transformers_weights")):
        raise manifest.mismatch()
    # Conservative upper bound: two embeddings (even if tied), Q/K/V/O, MLP and norms.
    parameters = 2 * raw["vocab_size"] * hidden + raw["num_hidden_layers"] * (4 * hidden ** 2 + 3 * hidden * raw["intermediate_size"] + 8 * hidden)
    if parameters > MAX_PARAMETERS:
        raise manifest.mismatch()
    adapter_bytes = sum(hp["rank"] * (hidden + (hidden if target in ("q_proj", "o_proj") else hidden * kv // heads))
                        for target in hp["targetModules"]) * raw["num_hidden_layers"] * 4
    if adapter_bytes > MAX_ADAPTER_BYTES or hp["maxSequenceLength"] > raw["max_position_embeddings"]:
        raise manifest.mismatch()
    # Adapter, best adapter, and three AdamW tensors per parameter plus bounded RNG state must fit checkpoint v1.
    if 2 * raw["num_hidden_layers"] * len(hp["targetModules"]) * 5 + 5 > 256:
        raise manifest.mismatch()
    safe = {key: raw[key] for key in set(bounds) | optional if key in raw}
    safe["rope_parameters"] = rope
    return parameters, safe


@contextlib.contextmanager
def provision(root, hp, device):
    """Each command reads a new verified private copy. Names never become paths supplied by a tenant."""
    if not isinstance(hp.get("pretrainedModel"), str):
        raise common.RunnerError("pretrained_model_unavailable", "A provisioned model name is required")
    with tempfile.TemporaryDirectory(prefix="meticule-peft-base-") as directory:
        try:
            provenance = manifest.verified_copy(root, hp["pretrainedModel"], {"causal-lm"}, {"llama"}, directory)
            raw = manifest.read_json(os.path.join(directory, "config.json"))
            parameters, safe = configuration(raw, hp)
            weights = os.path.join(directory, "model.safetensors")
            if not os.path.isfile(weights) or os.path.getsize(weights) > MAX_PARAMETERS * 4 + 1024 * 1024:
                raise manifest.mismatch()
            with open(weights, "rb") as handle:
                if struct.unpack("<Q", handle.read(8))[0] > 256 * 1024:
                    raise manifest.mismatch()
            count = 0
            with safe_open(weights, framework="pt", device="cpu") as handle:
                if len(handle.keys()) > 512:
                    raise manifest.mismatch()
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    count += tensor.numel()
                    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16) or count > MAX_PARAMETERS \
                            or not torch.isfinite(tensor).all():
                        raise manifest.mismatch()
            tokenizer_path = os.path.join(directory, "tokenizer.json")
            if os.path.getsize(tokenizer_path) > manifest.MAX_JSON_BYTES:
                raise manifest.mismatch()
            tokenizer = Tokenizer.from_file(tokenizer_path)
            vocabulary = tokenizer.get_vocab(with_added_tokens=True)
            if len(vocabulary) != raw["vocab_size"] or set(vocabulary.values()) != set(range(raw["vocab_size"])):
                raise manifest.mismatch()
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                value = raw.get(key)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < raw["vocab_size"]):
                    raise manifest.mismatch()
            tokenizer_digest = manifest.sha256_file(tokenizer_path)
        except manifest.ManifestError as error:
            raise common.RunnerError(error.code, "Provisioned model or tokenizer validation failed") from None
        except Exception:
            raise common.RunnerError("pretrained_model_mismatch", "Provisioned model or tokenizer validation failed") from None
        # Includes full-vocabulary logits at evaluation batch size 64, forward/backward activations and dense load overhead.
        estimate = parameters * 16 + 64 * hp["maxSequenceLength"] * raw["vocab_size"] * 12 + 256 * 1024 ** 2
        quant.preflight({k: hp[k] for k in quant.DEFAULTS if k in hp}, device, estimate, hp.get("maxMemoryMiB", 6144) * 1024 ** 2)
        yield {"directory": directory, "config": safe, "identity": {"id": provenance["name"], "digest": provenance["digest"],
               "config": safe, "tokenizerDigest": tokenizer_digest, "source": provenance["source"], "family": "llama"}}


def load(context):
    try:
        model, info = LlamaForCausalLM.from_pretrained(context["directory"], local_files_only=True, trust_remote_code=False,
                                                      use_safetensors=True, dtype=torch.float32, output_loading_info=True,
                                                      config=LlamaConfig(**context["config"]))
    except torch.cuda.OutOfMemoryError:
        raise
    except (ValueError, RuntimeError, KeyError, OSError):
        raise common.RunnerError("pretrained_model_mismatch", "Provisioned weights do not match the supported architecture") from None
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise common.RunnerError("pretrained_model_mismatch", "Provisioned weights do not exactly match the supported architecture")
    if sum(p.numel() for p in model.parameters()) > MAX_PARAMETERS:
        raise common.RunnerError("pretrained_model_mismatch", "Provisioned model exceeds the parameter bound")
    return model.eval()
