"""Bounded LoRA fine-tuning foundation (Phase 6B.1): Hugging Face PEFT adapters on a tiny, locally constructed transformer.

One supported base, `meticule-tiny-llama-v1`: a LlamaForCausalLM built from a fixed config and seed, never downloaded and
never loaded from a path. PEFT attaches LoRA adapters to the allowlisted attention projections, the base stays frozen, and
an explicit AdamW loop trains only the adapter on bounded token sequences (CPU, fp32, single-threaded, deterministic).

The adapter artifact (`meticule-lora-adapter-v1`) is a safetensors file: only the LoRA tensors, with JSON metadata naming
the base (id, config, weight digest) and the LoRA configuration. Loading takes bytes, never a path, bounds the size before
parsing, validates the metadata and every tensor against a freshly built compatible base, and refuses anything else.

Phase 6B.2 runs this through the Meticule lifecycle as the `peft` backend (python/lora_backend.py), which adds token
datasets, chunked training with full-state checkpoints, and the registry and serving paths.
"""
import os
import shutil
import tempfile
import atexit

# CUDA determinism needs this before the first cuBLAS context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# As in transformers_backend: an empty per-process cache and an offline hub, before transformers or peft is imported.
_CACHE = tempfile.mkdtemp(prefix="meticule-lora-")
atexit.register(shutil.rmtree, _CACHE, True)
os.environ.update({"HF_HOME": _CACHE, "HF_HUB_CACHE": os.path.join(_CACHE, "hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import peft  # noqa: E402
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import ml_runtime_common as common  # noqa: E402

ADAPTER_FORMAT = "meticule-lora-adapter-v1"
ADAPTER_SCHEMA = 1
BASE_ID = "meticule-tiny-llama-v1"
BASE_SEED = 1234
# initializer_range 0.3: with Llama's default 0.02 the tied, frozen output embedding is too small to express a confident
# prediction, so no adapter could fit even the controlled workload.
BASE_CONFIG = {"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
               "num_key_value_heads": 4, "max_position_embeddings": 64, "initializer_range": 0.3, "tie_word_embeddings": True}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LIMITS = {"maxRank": 16, "maxAlpha": 64.0, "maxDropout": 0.5, "maxLearningRate": 0.1, "maxBatchSize": 32, "maxSteps": 500,
          "maxSeed": 2 ** 31 - 1, "maxSequences": 1024, "maxSequenceLength": 32, "maxTrainingTokens": 500 * 32 * 16,
          "maxBaseParameters": 100_000, "maxAdapterBytes": 1024 * 1024, "maxHeaderBytes": 64 * 1024}
DEFAULTS = {"rank": 4, "alpha": 8.0, "dropout": 0.0, "learningRate": 0.02, "batchSize": 16, "steps": 200, "seed": 0,
            "targetModules": ["q_proj", "v_proj"]}
DATA_SEED, DATA_COUNT, DATA_LENGTH, DATA_PERIOD = 99, 256, 16, 8


def configure():
    """Single-threaded and deterministic, so a seeded run gives the same adapter in every process."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


configure()


def is_int(value, low, high):
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def is_number(value, low, high, open_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return (low < value if open_low else low <= value) and value <= high


def resolve_config(raw=None):
    """The validated LoRA and training configuration. Unknown keys, wrong types, non-finite values and out-of-range values fail."""
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unsupported LoRA configuration key")
    cfg = {**DEFAULTS, **raw}
    if not is_int(cfg["rank"], 1, LIMITS["maxRank"]):
        raise ValueError("LoRA rank out of range")
    if not is_number(cfg["alpha"], 0.0, LIMITS["maxAlpha"], open_low=True):
        raise ValueError("LoRA alpha out of range")
    if not is_number(cfg["dropout"], 0.0, LIMITS["maxDropout"]):
        raise ValueError("LoRA dropout out of range")
    if not is_number(cfg["learningRate"], 0.0, LIMITS["maxLearningRate"], open_low=True):
        raise ValueError("Learning rate out of range")
    if not is_int(cfg["batchSize"], 1, LIMITS["maxBatchSize"]) or not is_int(cfg["steps"], 1, LIMITS["maxSteps"]):
        raise ValueError("Batch size or steps out of range")
    if not is_int(cfg["seed"], 0, LIMITS["maxSeed"]):
        raise ValueError("Seed out of range")
    targets = cfg["targetModules"]
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t in TARGET_MODULES for t in targets) \
            or len(set(targets)) != len(targets):
        raise ValueError("Unsupported LoRA target modules")
    return {**cfg, "alpha": float(cfg["alpha"]), "dropout": float(cfg["dropout"]), "learningRate": float(cfg["learningRate"]),
            "targetModules": sorted(targets)}


def build_base(config=None, seed=BASE_SEED):
    """The base model, built from configuration with a fixed seed. Nothing is read from disk or the network."""
    config = BASE_CONFIG if config is None else config
    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(**config))
    if sum(p.numel() for p in model.parameters()) > LIMITS["maxBaseParameters"]:
        raise ValueError("Base model exceeds the parameter limit")
    return model.eval()


def base_config(model):
    """The recorded base configuration: the BASE_CONFIG keys, read back from the model."""
    return {key: getattr(model.config, key) for key in BASE_CONFIG}


def base_digest(model):
    """SHA-256 over every base tensor's name, dtype, shape and bytes, in name order. Taken before adapters are attached."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def synthetic_sequences(count=DATA_COUNT, length=DATA_LENGTH, period=DATA_PERIOD, seed=DATA_SEED):
    """The controlled workload: each sequence counts upward through `period` tokens from a random start (t+1 mod period)."""
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, period, (count, 1), generator=generator)
    return (starts + torch.arange(length)) % period


def validate_sequences(sequences, vocab_size):
    """A bounded 2-D integer token matrix with every id inside the vocabulary. Lists or a torch tensor; anything else fails."""
    try:
        tensor = sequences if isinstance(sequences, torch.Tensor) else torch.tensor(sequences)
    except (TypeError, ValueError, RuntimeError):
        raise ValueError("Token sequences must be a rectangular integer matrix") from None
    if tensor.dtype not in (torch.int64, torch.int32) or tensor.dim() != 2:
        raise ValueError("Token sequences must be a rectangular integer matrix")
    rows, length = tensor.shape
    if not 1 <= rows <= LIMITS["maxSequences"] or not 2 <= length <= LIMITS["maxSequenceLength"]:
        raise ValueError("Token sequence count or length out of range")
    if int(tensor.min()) < 0 or int(tensor.max()) >= vocab_size:
        raise ValueError("Token id outside the vocabulary")
    return tensor.to(torch.int64)


def lora_config(cfg):
    return LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"], target_modules=list(cfg["targetModules"]),
                      bias="none")


def attach_lora(base, cfg):
    """PEFT LoRA on the allowlisted projections. Adapter init (lora_A) is seeded; lora_B starts at zero, as LoRA defines."""
    torch.manual_seed(cfg["seed"])
    model = get_peft_model(base, lora_config(cfg))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    adapted = [name for name, module in model.named_modules() if hasattr(module, "lora_A") and name.split(".")[-1] in cfg["targetModules"]]
    # Only adapter matrices may train, and every targeted projection in every layer must carry one.
    if not trainable or any(".lora_A." not in n and ".lora_B." not in n for n in trainable) \
            or len(adapted) != len(cfg["targetModules"]) * base.config.num_hidden_layers:
        raise RuntimeError("LoRA did not attach exactly to the requested projections")
    return model


def parameter_counts(model):
    if hasattr(model, "get_nb_trainable_parameters"):
        # PEFT accounts for packed Params4bit storage; raw numel undercounts the logical base parameters.
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainablePercent": 100.0 * trainable / total}


def mean_loss(model, data):
    model.eval()
    with torch.no_grad():
        return float(model(input_ids=data, labels=data).loss)


def train(raw_config=None, sequences=None):
    """Fine-tunes LoRA adapters on the frozen base. Returns (peft model, configuration, report). Base weights never change."""
    cfg = resolve_config(raw_config)
    started = time.perf_counter()
    base = build_base()
    data = validate_sequences(synthetic_sequences() if sequences is None else sequences, base.config.vocab_size)
    if cfg["steps"] * cfg["batchSize"] * data.shape[1] > LIMITS["maxTrainingTokens"]:
        raise ValueError("Steps times batch size times sequence length exceeds the training budget")
    digest = base_digest(base)
    base_parameters = sum(p.numel() for p in base.parameters())
    frozen = [(name, p, p.detach().clone()) for name, p in base.named_parameters()]  # the same Parameter objects PEFT wraps
    model = attach_lora(base, cfg)
    adapters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    initial_adapters = {n: p.detach().clone() for n, p in adapters}
    optimizer = torch.optim.AdamW([p for _, p in adapters], lr=cfg["learningRate"], weight_decay=0.0)
    generator = torch.Generator().manual_seed(cfg["seed"])
    initial_loss = mean_loss(model, data)
    history, gradients = [], None
    model.train()
    for step in range(cfg["steps"]):
        batch = data[torch.randint(0, data.shape[0], (cfg["batchSize"],), generator=generator)]
        loss = model(input_ids=batch, labels=batch).loss
        if not torch.isfinite(loss):
            raise ValueError("LoRA training diverged: non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradients is None:  # the first backward pass: adapters get finite gradients, the frozen base gets none at all
            norms = {n: float(p.grad.norm()) for n, p in adapters if p.grad is not None}
            gradients = {"adapterTensors": len(adapters), "adapterTensorsWithGradient": len(norms),
                         "adapterGradientsFinite": all(math.isfinite(v) for v in norms.values()),
                         "adapterGradientNormTotal": math.sqrt(sum(v * v for v in norms.values())),
                         "frozenTensorsWithGradient": sum(p.grad is not None for _, p, _ in frozen)}
        optimizer.step()
        history.append(loss.item())
    final_loss = mean_loss(model, data)
    unchanged = all(torch.equal(p.detach(), before) for _, p, before in frozen) and all(not p.requires_grad for _, p, _ in frozen)
    changed = sum(not torch.equal(p.detach(), initial_adapters[n]) for n, p in adapters)
    if not unchanged:
        raise RuntimeError("A frozen base parameter changed during LoRA training")
    report = {"base": {"id": BASE_ID, "digest": digest, "parameters": base_parameters},
              "parameters": parameter_counts(model), "targetModules": cfg["targetModules"], "adaptedModules": len(cfg["targetModules"]) * base.config.num_hidden_layers,
              "steps": cfg["steps"], "initialLoss": initial_loss, "finalLoss": final_loss, "lossHistory": history,
              "gradients": gradients, "frozenBaseUnchanged": unchanged, "adapterTensorsChanged": changed,
              "seconds": time.perf_counter() - started, "device": "cpu"}
    return model, cfg, report


# ---- adapter artifact ----------------------------------------------------------------------------------------------

def versions():
    return {"peft": peft.__version__, "torch": torch.__version__.split("+")[0], "transformers": transformers.__version__, "safetensors": safetensors.__version__}


def export_adapter(model, cfg, digest, provenance=None, identity=None, quantization=None):
    """Only the LoRA tensors (PEFT's adapter state dict) plus metadata. The base weights are never written. `provenance`
    (6B.2: the training fingerprint, epochs and steps) is recorded under `training` and never read back by load_adapter."""
    base = model.get_base_model()
    quantization_versions = None
    if bool(getattr(base, "is_loaded_in_4bit", False)) != (quantization is not None):
        raise ValueError("Quantized adapter export requires its explicit quantization configuration")
    if quantization is not None:
        import lora_quantization
        if not lora_quantization.matches_configuration(base, quantization):
            raise ValueError("Quantized adapter export configuration does not match the base")
        quantization_versions = lora_quantization.dependencies()
    tensors = {name: tensor.detach().contiguous().cpu() for name, tensor in get_peft_model_state_dict(model).items()}
    if any(".lora_A." not in n and ".lora_B." not in n for n in tensors):
        raise RuntimeError("Adapter state holds a non-adapter tensor")
    lora = {key: cfg[key] for key in ("rank", "alpha", "dropout", "targetModules")}
    metadata = {"format": ADAPTER_FORMAT, "schema": ADAPTER_SCHEMA, "base": {"id": BASE_ID, "config": base_config(model.get_base_model()), "digest": digest},
                "lora": lora, "versions": versions()}
    if identity is not None:
        metadata["base"] = identity
    if quantization is not None:
        metadata["quantization"] = quantization
        metadata["quantizationVersions"] = quantization_versions
    if provenance is not None:
        metadata["training"] = provenance
    blob = safetensors.torch.save(tensors, metadata={"meticule": json.dumps(metadata, sort_keys=True, allow_nan=False)})
    if len(blob) > LIMITS["maxAdapterBytes"]:
        raise ValueError("Adapter exceeds the artifact size limit")
    return blob


def read_metadata(blob):
    """The safetensors header, parsed by hand within bounds so nothing is decoded before its size is known."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) > LIMITS["maxAdapterBytes"]:
        raise common.IncompatibleArtifact("oversized or invalid adapter")
    try:
        (size,) = struct.unpack("<Q", bytes(blob[:8]))
        if size > LIMITS["maxHeaderBytes"] or 8 + size > len(blob):
            raise ValueError("header out of bounds")
        header = json.loads(bytes(blob[8:8 + size]))
        metadata = json.loads(header["__metadata__"]["meticule"])
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
    except Exception:
        raise common.IncompatibleArtifact("unreadable adapter") from None
    return metadata


def load_adapter(blob, base, identity=None, quantization=None):
    """Attaches a stored adapter to `base`, a freshly built model. Every check runs before any adapter tensor is assigned."""
    metadata = read_metadata(blob)
    if metadata.get("format") != ADAPTER_FORMAT or metadata.get("schema") != ADAPTER_SCHEMA:
        raise common.IncompatibleArtifact("unsupported adapter format or schema")
    if metadata.get("quantization") != quantization:
        raise common.IncompatibleArtifact("adapter quantization does not match the requested base")
    if quantization is not None:
        import lora_quantization
        try:
            matches = lora_quantization.matches_configuration(base, quantization) and metadata.get("quantizationVersions") == lora_quantization.dependencies()
        except (ValueError, TypeError, AttributeError, ImportError, common.RunnerError):
            matches = False
        if not matches:
            raise common.IncompatibleArtifact("adapter needs a matching quantized base")
    recorded = metadata.get("versions") or {}
    current = versions()
    # PEFT owns the adapter tensor names, so its minor version must match; the rest is recorded for audit.
    if str(recorded.get("peft", "")).split(".")[:2] != current["peft"].split(".")[:2]:
        raise common.IncompatibleArtifact("adapter was written by an incompatible PEFT version")
    stored_base = metadata.get("base") or {}
    expected_base = identity if identity is not None else {"id": BASE_ID, "config": base_config(base), "digest": base_digest(base)}
    if stored_base != expected_base:
        raise common.IncompatibleArtifact("adapter belongs to a different base model")
    try:
        cfg = resolve_config({**metadata["lora"], "seed": 0})
    except (KeyError, TypeError, ValueError):
        raise common.IncompatibleArtifact("adapter LoRA configuration is not supported") from None
    try:
        tensors = safetensors.torch.load(bytes(blob))
    except Exception:
        raise common.IncompatibleArtifact("unreadable adapter") from None
    model = attach_lora(base, cfg)
    expected = {name: tuple(t.shape) for name, t in get_peft_model_state_dict(model).items()}
    if {name: tuple(t.shape) for name, t in tensors.items()} != expected:
        raise common.IncompatibleArtifact("adapter tensors do not match the LoRA configuration")
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors.values()):
        raise common.IncompatibleArtifact("adapter holds non-finite or non-fp32 values")
    result = set_peft_model_state_dict(model, tensors)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise common.IncompatibleArtifact("adapter tensors do not match the LoRA configuration")
    return model.eval()


def logits(model, sequences):
    model.eval()
    with torch.no_grad():
        return model(input_ids=validate_sequences(sequences, model.config.vocab_size)).logits
