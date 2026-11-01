"""Optional CUDA quantization boundary for the existing PEFT runner; never downloads assets."""
import importlib.metadata
import math
import tempfile

import torch

import ml_runtime_common as common

DEFAULTS = {"quantization": "none", "computeDtype": "fp32", "doubleQuantization": False}
PINS = {"torch": "2.9.1", "transformers": "5.17.0", "peft": "0.21.2", "accelerate": "1.15.0",
        "safetensors": "0.8.0", "bitsandbytes": "0.50.2"}


def settings(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unsupported quantization configuration")
    value = {**DEFAULTS, **raw}
    if value["quantization"] not in ("none", "nf4", "fp4") or value["computeDtype"] not in ("fp32", "bf16") \
            or not isinstance(value["doubleQuantization"], bool):
        raise ValueError("Unsupported quantization configuration")
    if value["quantization"] == "none" and (value["computeDtype"] != "fp32" or value["doubleQuantization"]):
        raise ValueError("Quantization options require four-bit quantization")
    return value


def dependencies():
    versions = {}
    for package, expected in PINS.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise common.RunnerError("quantization_runtime_unavailable", "The pinned quantization runtime is not installed") from None
        if versions[package].split("+")[0] != expected:
            raise common.RunnerError("quantization_runtime_incompatible", "The quantization runtime version is not supported")
    return versions


def preflight(config, device, estimated_bytes, memory_limit_bytes=None):
    """Memory is advisory, not isolation or a guarantee that execution will fit."""
    config = settings(config)
    if not isinstance(estimated_bytes, int) or isinstance(estimated_bytes, bool) or estimated_bytes < 0:
        raise ValueError("Invalid memory estimate")
    if not isinstance(device, str) or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported device")
    if config["quantization"] != "none" and device != "cuda":
        raise common.RunnerError("cuda_unsupported", "Four-bit adapter training requires CUDA")
    if device == "cpu":
        return {"device": "cpu", "estimatedBytes": estimated_bytes, "memoryEnforcement": "advisory"}
    if not torch.cuda.is_available():
        raise common.RunnerError("cuda_unavailable", "CUDA is unavailable")
    if memory_limit_bytes is not None and (not isinstance(memory_limit_bytes, int) or isinstance(memory_limit_bytes, bool)
                                          or memory_limit_bytes <= 0):
        raise ValueError("Invalid memory limit")
    capability = torch.cuda.get_device_capability(0)
    if config["quantization"] != "none":
        versions = dependencies()
        # The pinned Windows/Linux CUDA 12.8 wheels target sm70 and newer.
        if capability < (7, 0) or torch.version.cuda != "12.8":
            raise common.RunnerError("cuda_unsupported", "Unsupported quantization CUDA runtime or compute capability")
        if config["computeDtype"] == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise common.RunnerError("cuda_unsupported", "BF16 is unavailable on this device")
    else:
        versions = {}
    free, total = torch.cuda.mem_get_info(0)
    available = min(free, memory_limit_bytes) if memory_limit_bytes is not None else free
    if estimated_bytes > math.floor(available * 0.8):
        raise common.RunnerError("cuda_memory_budget_exceeded", "Estimated memory exceeds the advisory CUDA memory budget")
    return {"device": "cuda", "estimatedBytes": estimated_bytes, "freeBytes": free, "totalBytes": total,
            "capability": list(capability), "runtime": torch.version.cuda, "versions": versions,
            "memoryEnforcement": "advisory"}


def load_quantized(base, config):
    """Quantize a trusted, locally constructed Llama base through the real Transformers/bitsandbytes loader.

    A temporary safetensors representation avoids a hand-written parallel quantization implementation. Provisioned
    models will enter through the existing verified-copy manifest boundary, never a model name or remote URL.
    """
    from transformers import BitsAndBytesConfig, LlamaForCausalLM
    from peft import prepare_model_for_kbit_training

    config = settings(config)
    if config["quantization"] == "none":
        raise ValueError("Quantized loading requires four-bit quantization")
    preflight(config, "cuda", sum(p.numel() for p in base.parameters()) * 8 + 256 * 1024 ** 2)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=config["quantization"],
                                    bnb_4bit_compute_dtype=torch.bfloat16 if config["computeDtype"] == "bf16" else torch.float32,
                                    bnb_4bit_use_double_quant=config["doubleQuantization"])
    with tempfile.TemporaryDirectory(prefix="meticule-quantized-") as directory:
        base.save_pretrained(directory, safe_serialization=True)
        model = LlamaForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True, quantization_config=quantization, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if not matches_configuration(model, config) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Quantized base was not loaded and frozen")
    model.config.use_cache = False
    return model


def matches_configuration(model, config):
    config = settings(config)
    if config["quantization"] == "none" or not getattr(model, "is_loaded_in_4bit", False):
        return False
    import bitsandbytes as bnb
    layers = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
    dtype = torch.bfloat16 if config["computeDtype"] == "bf16" else torch.float32
    return bool(layers) and all(
        layer.weight.quant_state is not None and layer.weight.quant_state.quant_type == config["quantization"]
        and layer.weight.quant_state.nested == config["doubleQuantization"] and layer.compute_dtype == dtype for layer in layers)
