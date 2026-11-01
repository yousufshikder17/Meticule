"""PEFT LoRA backend (Phase 6B.2): durable LoRA fine-tuning of `meticule-tiny-llama-v1` over the Meticule runner protocol.

The model and adapter mechanics are lora_finetune (6B.1): the base is built from a fixed config and seed, PEFT attaches LoRA
to allowlisted attention projections, the base stays frozen, and the adapter artifact is safetensors with its base identity.
This runner adds what the lifecycle needs: one `tokens` column of integer sequences, epoch-shuffled AdamW training in
bounded chunks (prepare, train, train_chunk, export, evaluate, predict, like the other neural runners), and a full-state
checkpoint so a resumed run continues exactly: adapter weights, AdamW moments and step counts, the shuffle generator and the
torch, NumPy and Python RNG states, progress, and the base, configuration and data fingerprints.

Checkpoints are safetensors plus JSON metadata, bounded and validated in full before any tensor is assigned; nothing is
unpickled and torch.load is never used. CPU/CUDA adapters and AdamW stay fp32; optional four-bit base computation uses
fp32 or BF16. Provisioned Llama assets reuse the verified manifest boundary. Nothing is downloaded or runs remote code.
"""
import runner_parent_guard
runner_parent_guard.start_parent_guard()  # before ML imports, including Windows venv-launcher cancellation
import lora_finetune as lf  # pins the Hugging Face libraries offline before they are imported

import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import peft  # noqa: E402
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402

import ml_runtime_common as common  # noqa: E402
import lora_quantization as quant  # noqa: E402
import lora_provisioning as provisioning  # noqa: E402

BACKEND = "peft"
ARTIFACT_FORMAT = lf.ADAPTER_FORMAT
ARTIFACT_SCHEMA = lf.ADAPTER_SCHEMA
PROTOCOL = "meticule-peft-v1"
CHECKPOINT_FORMAT = "peft-lora-checkpoint-v1"
CHECKPOINT_SCHEMA = 1
RUNTIME_KEYS = ("torch", "transformers", "peft", "safetensors", "numpy")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": {PROTOCOL: ARTIFACT_SCHEMA}, "schemas": {ARTIFACT_SCHEMA},
            "runtime_keys": RUNTIME_KEYS}
MODELS = {"lora_causal_lm": "language_modeling"}
BASE_MODELS = {lf.BASE_ID: (lf.BASE_CONFIG, lf.BASE_SEED)}
# Mirrors LORA_LIMITS in src/ml/peft-models.ts; steps and tokens need the data, so only the runner checks them.
LIMITS = {"maxRows": 1024, "maxEpochs": 100, "maxSequenceLength": 32, "maxSteps": 500, "maxTrainingTokens": 256_000,
          "maxPredictionRows": 64, "maxCheckpointBytes": 4 * 1024 * 1024, "maxHeaderBytes": 256 * 1024, "maxTensors": 256}
DEFAULTS = {"baseModel": lf.BASE_ID, "rank": 4, "alpha": 8.0, "dropout": 0.0, "targetModules": ["q_proj", "v_proj"], "learningRate": 0.02,
            "batchSize": 16, "epochs": 10, "maxSequenceLength": 32}
ADAMW = {"betas": [0.9, 0.999], "eps": 1e-8, "weightDecay": 0.0}
TOKENS = re.compile(r"(0|[1-9][0-9]{0,5})( (0|[1-9][0-9]{0,5})){0,511}", re.ASCII)  # domain.ts TOKENS_CELL; fullmatch
RunnerError = common.RunnerError


def environment():
    return common.base_environment([__file__, lf.__file__, quant.__file__, provisioning.__file__, provisioning.manifest.__file__, runner_parent_guard.__file__, common.__file__], BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                                    "safetensors": safetensors.__version__, "numpy": np.__version__})


def check_compatible(recorded, current):
    common.check_compatible(recorded, current, CONTRACT)


def resolve_device(spec):
    """Explicit single visible CUDA device or CPU; adapter/optimizer state remains fp32."""
    training = spec.get("training") or {}
    device = training.get("device", "cpu")
    if device not in ("cpu", "cuda") or training.get("precision", "fp32") != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")
    hp = spec.get("hyperparameters") or {}
    cfg = quant.settings({k: hp[k] for k in quant.DEFAULTS if k in hp})
    # Tiny-base estimate, deliberately conservative. Provisioned bases require their own parameter estimate.
    quant.preflight(cfg, device, 256 * 1024 ** 2, hp.get("maxMemoryMiB", 6144) * 1024 ** 2)
    return device


def resolve_hyperparameters(spec):
    """The catalog's allowlist again, at the runner boundary: unknown keys, wrong types and out-of-range values fail."""
    hp = {**DEFAULTS, **(spec.get("hyperparameters") or {})}
    if set(hp) - set(DEFAULTS) - set(quant.DEFAULTS) - {"maxMemoryMiB", "gradientAccumulationSteps", "pretrainedModel"}:
        raise ValueError("Unsupported LoRA hyperparameter")
    quant.settings({k: hp[k] for k in quant.DEFAULTS if k in hp})
    if not lf.is_int(hp.get("maxMemoryMiB", 6144), 256, 65536) or not lf.is_int(hp.get("gradientAccumulationSteps", 1), 1, 32):
        raise ValueError("Resource bounds are invalid")
    if hp["baseModel"] not in BASE_MODELS and hp["baseModel"] != provisioning.BASE_ID:
        raise ValueError("Unsupported base model")
    if hp["baseModel"] == provisioning.BASE_ID:
        if not isinstance(hp.get("pretrainedModel"), str) or not provisioning.manifest.NAME.fullmatch(hp["pretrainedModel"]):
            raise ValueError("A provisioned model identifier is required")
    elif hp.get("pretrainedModel") is not None:
        raise ValueError("Provisioned model name requires the provisioned Llama base")
    if not lf.is_int(hp["epochs"], 1, LIMITS["maxEpochs"]) or not lf.is_int(hp["maxSequenceLength"], 2, LIMITS["maxSequenceLength"]):
        raise ValueError("Epochs or sequence length out of range")
    cfg = lf.resolve_config({key: hp[key] for key in ("rank", "alpha", "dropout", "learningRate", "batchSize", "targetModules")})
    return {**hp, **{key: cfg[key] for key in ("rank", "alpha", "dropout", "learningRate", "batchSize", "targetModules")}}


def lora_settings(hp, seed):
    return {"rank": hp["rank"], "alpha": hp["alpha"], "dropout": hp["dropout"], "targetModules": hp["targetModules"], "seed": seed}


def feature(spec):
    """The one token column. TypeScript validation guarantees it; the runner refuses anything else regardless."""
    features = spec["pipeline"]["definition"]["features"]
    types = {c["name"]: c["type"] for c in spec["dataset"]["schema"]["columns"]}
    if len(features) != 1 or types.get(features[0]) != "tokens" or spec["pipeline"]["definition"]["steps"] or spec["dataset"]["schema"].get("target") is not None:
        raise ValueError("LoRA fine-tuning needs exactly one tokens column, no preprocessing and no target")
    return features[0]


def sequences(spec, rows, hp, indices=None):
    """Parsed token ids of the given rows. A cell outside the grammar, the vocabulary or the length bounds is invalid_tokens."""
    column = feature(spec)
    vocab = spec["_base"]["config"]["vocab_size"] if hp["baseModel"] == provisioning.BASE_ID else BASE_MODELS[hp["baseModel"]][0]["vocab_size"]
    chosen = range(len(rows)) if indices is None else indices
    if len(chosen) > LIMITS["maxRows"]:
        raise RunnerError("invalid_tokens", "Too many token sequences")
    parsed = []
    for i in chosen:
        cell = rows[i].get(column) if isinstance(rows[i], dict) else None
        if not isinstance(cell, str) or not TOKENS.fullmatch(cell):
            raise RunnerError("invalid_tokens", "A token sequence is not a list of integer ids")
        ids = [int(t) for t in cell.split(" ")]
        if not 2 <= len(ids) <= hp["maxSequenceLength"] or max(ids) >= vocab:
            raise RunnerError("invalid_tokens", "A token sequence is outside the model's vocabulary or length bounds")
        parsed.append(ids)
    return parsed


def batch_tensors(seqs):
    """Right-padded ids, attention mask and labels (-100 on padding, which the loss ignores)."""
    width = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.int64)
    mask = torch.zeros((len(seqs), width), dtype=torch.int64)
    for row, s in enumerate(seqs):
        ids[row, :len(s)] = torch.tensor(s)
        mask[row, :len(s)] = 1
    return ids, mask, ids.masked_fill(mask == 0, -100)


def token_scores(model, seqs, block=64):
    """Summed next-token cross-entropy, predicted-token count and correct greedy predictions over `seqs` (eval mode)."""
    model.eval()
    total, count, correct = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(seqs), block):
            ids, mask, labels = batch_tensors(seqs[start:start + block])
            ids, mask, labels = (t.to(next(model.parameters()).device) for t in (ids, mask, labels))
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            target = labels[:, 1:]
            total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100, reduction="sum"))
            valid = target != -100
            count += int(valid.sum())
            correct += int(((logits.argmax(-1) == target) & valid).sum())
    return total, count, correct


def mean_loss(model, seqs):
    total, count, _ = token_scores(model, seqs)
    loss = total / count
    if not math.isfinite(loss):
        raise ValueError("LoRA training diverged: non-finite loss")
    return loss


def fingerprint(spec, hp, prepared, train, validation, base_digest):
    """Identity of one training configuration and its data. A checkpoint only continues the run that wrote it."""
    payload = {"algorithm": spec["algorithm"], "hyperparameters": hp, "seed": spec["seed"], "feature": feature(spec),
               "fit": prepared["indices"]["train"], "validation": prepared["indices"]["validation"],
               "data": hashlib.sha256(json.dumps([train, validation]).encode()).hexdigest(), "base": base_digest}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_budget(hp, train):
    steps = hp["epochs"] * math.ceil(math.ceil(len(train) / hp["batchSize"]) / hp.get("gradientAccumulationSteps", 1))
    tokens = hp["epochs"] * sum(len(s) for s in train)
    if steps > LIMITS["maxSteps"] or tokens > LIMITS["maxTrainingTokens"]:
        raise RunnerError("invalid_tokens", "The training data and epochs exceed the LoRA step or token budget")


class Run:
    """Everything one training run carries between chunks. Built fresh or restored from a checkpoint."""

    def __init__(self, spec, hp, prepared, train, validation):
        self.spec, self.hp, self.prepared, self.train, self.validation = spec, hp, prepared, train, validation
        base, self.base_identity = build_base(spec, hp)
        self.base_digest = self.base_identity["digest"]
        self.device = resolve_device(spec)
        self.quantization = quant.settings({k: hp[k] for k in quant.DEFAULTS if k in hp})
        if self.quantization["quantization"] != "none":
            base = quant.load_quantized(base, self.quantization)
        else:
            base = base.to(self.device)
        self.model = lf.attach_lora(base, lora_settings(hp, spec["seed"]))  # seeds torch, then initialises lora_A
        self.names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=hp["learningRate"],
                                           betas=tuple(ADAMW["betas"]), eps=ADAMW["eps"], weight_decay=ADAMW["weightDecay"])
        self.sampler = torch.Generator().manual_seed(spec["seed"])
        random.seed(spec["seed"])
        np.random.seed(spec["seed"] % 2 ** 32)
        self.fingerprint = fingerprint(spec, hp, prepared, train, validation, self.base_digest)
        self.completed, self.steps, self.elapsed = 0, 0, 0.0
        self.initial = self.last_train = self.last_validation = None
        self.best = None  # {"epoch", "value", "state": adapter state dict}

    def adapter_state(self):
        return {name: t.detach().clone() for name, t in get_peft_model_state_dict(self.model).items()}


def fresh_run(spec, rows, prepared):
    hp = resolve_hyperparameters(spec)
    resolve_device(spec)
    if not prepared["indices"]["train"]:
        raise ValueError("Training partition is empty")
    train = sequences(spec, rows, hp, prepared["indices"]["train"])
    validation = sequences(spec, rows, hp, prepared["indices"]["validation"])
    check_budget(hp, train)
    run = Run(spec, hp, prepared, train, validation)
    run.initial = mean_loss(run.model, train)
    return run


def run_epochs(run, count):
    """`count` epochs, each one seeded shuffle of the training rows in batches; a chunk always ends on a completed step."""
    summaries = []
    for _ in range(count):
        started = time.perf_counter()
        order = torch.randperm(len(run.train), generator=run.sampler).tolist()
        run.model.train()
        accumulation = run.hp.get("gradientAccumulationSteps", 1)
        batches = math.ceil(len(order) / run.hp["batchSize"])
        run.optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), run.hp["batchSize"]):
            batch = start // run.hp["batchSize"]
            ids, mask, labels = batch_tensors([run.train[i] for i in order[start:start + run.hp["batchSize"]]])
            ids, mask, labels = (t.to(run.device) for t in (ids, mask, labels))
            loss = run.model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("LoRA training diverged: non-finite loss")
            group_start = (batch // accumulation) * accumulation * run.hp["batchSize"]
            group_indices = order[group_start:group_start + accumulation * run.hp["batchSize"]]
            group_tokens = sum(len(run.train[i]) - 1 for i in group_indices)
            (loss * ((labels[:, 1:] != -100).sum() / group_tokens)).backward()
            if (batch + 1) % accumulation == 0 or batch + 1 == batches:
                run.optimizer.step()
                run.optimizer.zero_grad(set_to_none=True)
                run.steps += 1
        train_loss = mean_loss(run.model, run.train)
        validation_loss = mean_loss(run.model, run.validation) if run.validation else None
        elapsed = time.perf_counter() - started
        run.completed += 1
        run.elapsed += elapsed
        run.last_train, run.last_validation = train_loss, validation_loss
        if validation_loss is not None and (run.best is None or validation_loss < run.best["value"]):
            run.best = {"epoch": run.completed, "value": validation_loss, "state": run.adapter_state()}
        summaries.append({"epoch": run.completed, "trainLoss": train_loss, "validationLoss": validation_loss, "elapsedSeconds": elapsed})
    return summaries


# ---- checkpoints ---------------------------------------------------------------------------------------------------

def save_run(run, env):
    tensors = {f"adapter.{k}": v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(run.model).items()}
    for index, state in run.optimizer.state_dict()["state"].items():
        for key in ("step", "exp_avg", "exp_avg_sq"):
            tensors[f"optimizer.{index}.{key}"] = state[key].detach().cpu().clone().contiguous()
    tensors["rng.torch"] = torch.get_rng_state()
    tensors["rng.sampler"] = run.sampler.get_state()
    if run.device == "cuda":
        tensors["rng.cuda"] = torch.cuda.get_rng_state(0)
    _, keys, pos, has_gauss, cached = np.random.get_state()
    tensors["rng.numpy"] = torch.from_numpy(np.array(keys, dtype=np.uint32).view(np.int32).copy())
    if run.best is not None:
        tensors.update({f"best.{k}": v.cpu().contiguous() for k, v in run.best["state"].items()})
    version, state, gauss = random.getstate()
    metadata = {"format": CHECKPOINT_FORMAT, "schema": CHECKPOINT_SCHEMA, "backend": BACKEND, "algorithm": run.spec["algorithm"],
                "base": run.base_identity,
                "hyperparameters": run.hp, "fingerprint": run.fingerprint, "completedEpoch": run.completed, "optimizerSteps": run.steps,
                "elapsedSeconds": run.elapsed, "initialTrainLoss": run.initial, "lastTrainLoss": run.last_train,
                "lastValidationLoss": run.last_validation, "best": None if run.best is None else {"epoch": run.best["epoch"], "value": run.best["value"]},
                "optimizer": {"parameters": run.names, "lr": run.hp["learningRate"], **ADAMW},
                "rng": {"python": [version, list(state), gauss], "numpy": {"pos": int(pos), "hasGauss": int(has_gauss), "cachedGaussian": float(cached)}},
                "device": run.device, "precision": "fp32", "environment": env}
    blob = safetensors.torch.save(tensors, metadata={"meticule": json.dumps(metadata, sort_keys=True, allow_nan=False)})
    return base64.b64encode(blob).decode()


def read_checkpoint(encoded):
    """(metadata, tensors). The size, header and tensor count are bounded before anything is decoded or allocated."""
    if not isinstance(encoded, str) or len(encoded) > LIMITS["maxCheckpointBytes"] * 4 // 3 + 4:
        raise common.InvalidCheckpoint("oversized checkpoint")
    try:
        raw = base64.b64decode(encoded, validate=True)
        (size,) = struct.unpack("<Q", raw[:8])
        if size > LIMITS["maxHeaderBytes"] or 8 + size > len(raw):
            raise ValueError("header out of bounds")
        header = json.loads(raw[8:8 + size])
        if len(header) - 1 > LIMITS["maxTensors"]:
            raise ValueError("too many tensors")
        metadata = json.loads(header["__metadata__"]["meticule"])
        tensors = safetensors.torch.load(raw)
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
    except Exception:
        raise common.InvalidCheckpoint("unreadable checkpoint") from None
    return metadata, tensors


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validated_checkpoint(encoded, spec, env, run):
    """Every refusal before any state is restored. `run` is a freshly built run for this snapshot (expected shapes and
    identity); its fingerprint is compared only when it carries the data (train_chunk, not export)."""
    meta, tensors = read_checkpoint(encoded)
    if meta.get("format") != CHECKPOINT_FORMAT or meta.get("schema") != CHECKPOINT_SCHEMA:
        raise common.IncompatibleCheckpoint("unsupported checkpoint format or schema")
    try:
        common.check_compatible(meta.get("environment"), env, CONTRACT)
    except common.IncompatibleArtifact as error:
        raise common.IncompatibleCheckpoint(str(error)) from None
    try:
        if meta["backend"] != BACKEND or meta["algorithm"] != spec["algorithm"] or meta["hyperparameters"] != run.hp:
            raise common.IncompatibleCheckpoint("Checkpoint belongs to a different training configuration")
        if meta["base"] != run.base_identity:
            raise common.IncompatibleCheckpoint("Checkpoint belongs to a different base model")
        if run.train is not None and meta["fingerprint"] != run.fingerprint:
            raise common.IncompatibleCheckpoint("Checkpoint belongs to a different training configuration or dataset")
        if meta["device"] != run.device or meta["precision"] != "fp32":
            raise common.IncompatibleCheckpoint("Checkpoint device or precision is not supported")
        if meta["optimizer"] != {"parameters": run.names, "lr": run.hp["learningRate"], **ADAMW}:
            raise common.IncompatibleCheckpoint("Checkpoint optimizer does not match the configuration")
        completed, steps = meta["completedEpoch"], meta["optimizerSteps"]
        per_epoch = None if run.train is None else math.ceil(math.ceil(len(run.train) / run.hp["batchSize"]) / run.hp.get("gradientAccumulationSteps", 1))
        if not lf.is_int(completed, 1, run.hp["epochs"]) or not lf.is_int(steps, completed, LIMITS["maxSteps"]) \
                or (per_epoch is not None and steps != completed * per_epoch):
            raise common.InvalidCheckpoint("Checkpoint progress is invalid")
        progress = [meta["elapsedSeconds"], meta["initialTrainLoss"], meta["lastTrainLoss"]] + ([] if meta["lastValidationLoss"] is None else [meta["lastValidationLoss"]])
        best = meta["best"]
        if not all(is_number(v) for v in progress) or meta["elapsedSeconds"] < 0 \
                or (best is not None and not (lf.is_int(best["epoch"], 1, completed) and is_number(best["value"]))):
            raise common.InvalidCheckpoint("Checkpoint progress is invalid")
        python_rng, numpy_rng = meta["rng"]["python"], meta["rng"]["numpy"]
        if not (isinstance(python_rng, list) and len(python_rng) == 3 and python_rng[0] == 3 and isinstance(python_rng[1], list) and len(python_rng[1]) == 625
                and lf.is_int(python_rng[0], 3, 3) and all(lf.is_int(v, 0, 2 ** 32 - 1) for v in python_rng[1][:-1])
                and lf.is_int(python_rng[1][-1], 0, 624) and (python_rng[2] is None or is_number(python_rng[2]))
                and lf.is_int(numpy_rng["pos"], 0, 624) and numpy_rng["hasGauss"] in (0, 1) and is_number(numpy_rng["cachedGaussian"])):
            raise common.InvalidCheckpoint("Checkpoint random state is malformed")
    except (KeyError, TypeError, AttributeError):
        raise common.InvalidCheckpoint("Checkpoint metadata is malformed") from None
    adapter = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(run.model).items()}
    expected = {f"adapter.{k}": s for k, s in adapter.items()}
    expected.update({f"optimizer.{i}.{key}": (() if key == "step" else adapter_shape) for i, adapter_shape in
                     enumerate(tuple(p.shape) for p in run.model.parameters() if p.requires_grad) for key in ("step", "exp_avg", "exp_avg_sq")})
    expected.update({"rng.torch": tuple(torch.get_rng_state().shape), "rng.sampler": tuple(torch.Generator().get_state().shape), "rng.numpy": (624,)})
    if run.device == "cuda":
        expected["rng.cuda"] = tuple(torch.cuda.get_rng_state(0).shape)
        if run.quantization["quantization"] != "none":
            current = artifact_environment(env, run.base_digest, run.hp, run.device)
            for key in (*quant.PINS, "quantization", "quantization_compute_dtype", "quantization_double"):
                if meta["environment"].get(key) != current.get(key):
                    raise common.IncompatibleCheckpoint("Checkpoint quantization configuration or runtime differs")
        for key in ("cuda_runtime", "device_capability", "device_name"):
            if meta["environment"].get(key) != artifact_environment(env, run.base_digest, run.hp, run.device).get(key):
                raise common.IncompatibleCheckpoint("Checkpoint GPU is incompatible")
    if best is not None:
        expected.update({f"best.{k}": s for k, s in adapter.items()})
    if set(tensors) != set(expected):
        raise common.InvalidCheckpoint("Checkpoint contents do not match its metadata")
    if any(tuple(tensors[k].shape) != s for k, s in expected.items() if k.startswith(("adapter.", "best."))):
        raise common.IncompatibleCheckpoint("Checkpoint adapter does not match the LoRA configuration")
    for name, t in tensors.items():
        kind = name.split(".")[0]
        want = {"rng": torch.int32 if name == "rng.numpy" else torch.uint8}.get(kind, torch.float32)
        if tuple(t.shape) != expected[name] or t.dtype != want or (want == torch.float32 and not torch.isfinite(t).all()):
            raise common.InvalidCheckpoint("Checkpoint holds malformed or non-finite tensors")
        if name.endswith(".exp_avg_sq") and bool((t < 0).any()) or name.endswith(".step") and float(t) != steps:
            raise common.InvalidCheckpoint("Checkpoint optimizer state is inconsistent")
    # Exercise restoration on isolated generators before assigning any checkpoint state to the run or global RNGs.
    try:
        random.Random().setstate((python_rng[0], tuple(python_rng[1]), python_rng[2]))
        np.random.RandomState().set_state(("MT19937", tensors["rng.numpy"].numpy().view(np.uint32), numpy_rng["pos"], numpy_rng["hasGauss"], numpy_rng["cachedGaussian"]))
        torch.Generator().set_state(tensors["rng.torch"])
        torch.Generator().set_state(tensors["rng.sampler"])
        if run.device == "cuda":
            torch.Generator(device="cuda").set_state(tensors["rng.cuda"])
    except (ValueError, TypeError, RuntimeError, OverflowError):
        raise common.InvalidCheckpoint("Checkpoint random state cannot be restored") from None
    return meta, tensors


def load_adapter_tensors(model, state):
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise common.IncompatibleCheckpoint("Checkpoint adapter does not match the LoRA configuration")


def restore_run(encoded, spec, rows, prepared, env):
    """Rebuilds the exact run a checkpoint froze. A checkpoint that does not fit is refused; it never becomes a fresh start."""
    run = fresh_run(spec, rows, prepared)
    meta, tensors = validated_checkpoint(encoded, spec, env, run)
    python_rng = meta["rng"]["python"]
    try:
        random.setstate((python_rng[0], tuple(python_rng[1]), python_rng[2]))
    except (ValueError, TypeError, OverflowError):
        raise common.InvalidCheckpoint("Checkpoint Python random state cannot be restored") from None
    load_adapter_tensors(run.model, {k[len("adapter."):]: v for k, v in tensors.items() if k.startswith("adapter.")})
    template = run.optimizer.state_dict()
    template["state"] = {i: {key: tensors[f"optimizer.{i}.{key}"] for key in ("step", "exp_avg", "exp_avg_sq")} for i in range(len(run.names))}
    run.optimizer.load_state_dict(template)
    torch.set_rng_state(tensors["rng.torch"])
    if run.device == "cuda":
        torch.cuda.set_rng_state(tensors["rng.cuda"], 0)
    run.sampler.set_state(tensors["rng.sampler"])
    numpy_rng = meta["rng"]["numpy"]
    np.random.set_state(("MT19937", tensors["rng.numpy"].numpy().view(np.uint32), numpy_rng["pos"], numpy_rng["hasGauss"], numpy_rng["cachedGaussian"]))
    run.completed, run.steps, run.elapsed = meta["completedEpoch"], meta["optimizerSteps"], meta["elapsedSeconds"]
    run.initial, run.last_train, run.last_validation = meta["initialTrainLoss"], meta["lastTrainLoss"], meta["lastValidationLoss"]
    if meta["best"] is not None:
        run.best = {"epoch": meta["best"]["epoch"], "value": meta["best"]["value"], "state": {k[len("best."):]: v for k, v in tensors.items() if k.startswith("best.")}}
    return run


# ---- adapter artifact and inference ----------------------------------------------------------------------------------

def artifact_environment(env, digest, hp, device="cpu"):
    """Recorded with the attempt: the runtime, CPU provenance, and the fields that mark this model as an adapter on a base."""
    quantization_versions = quant.dependencies() if hp.get("quantization", "none") != "none" else {}
    return {**env, **quantization_versions, "device_requested": device, "device_actual": device, "cuda_runtime": str(torch.version.cuda) if device == "cuda" else "none",
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "none",
            "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if device == "cuda" else "none",
            "quantization": hp.get("quantization", "none"), "quantization_compute_dtype": hp.get("computeDtype", "fp32"),
            "quantization_double": str(hp.get("doubleQuantization", False)).lower(), "memory_enforcement": "advisory",
            "precision": "fp32", "artifact_kind": "lora_adapter", "base_model": hp["baseModel"], "base_model_digest": digest,
            "lora_rank": str(hp["rank"]), "lora_target_modules": ",".join(hp["targetModules"])}


def performance(meta_like, counts):
    result = {"fitSeconds": float(meta_like["elapsedSeconds"]), "epochsCompleted": float(meta_like["completedEpoch"]),
              "optimizerSteps": float(meta_like["optimizerSteps"]), "initialTrainLoss": meta_like["initialTrainLoss"],
              "finalTrainLoss": meta_like["lastTrainLoss"], "trainableParameters": float(counts["trainable"]), "totalParameters": float(counts["total"])}
    if meta_like["lastValidationLoss"] is not None:
        result["finalValidationLoss"] = meta_like["lastValidationLoss"]
    if meta_like.get("best") is not None:
        result["bestEpoch"] = float(meta_like["best"]["epoch"])
    return result


def export(run, state, env, progress):
    """The 6B.1 adapter artifact (adapter tensors, LoRA settings, base identity) plus training provenance."""
    if state is not None:
        load_adapter_tensors(run.model, state)
    provenance = {"fingerprint": run.fingerprint, "epochs": progress["completedEpoch"], "optimizerSteps": progress["optimizerSteps"]}
    blob = lf.export_adapter(run.model, lora_settings(run.hp, run.spec["seed"]), run.base_digest, provenance, identity=run.base_identity,
                             quantization=run.quantization if run.quantization["quantization"] != "none" else None)
    env_export = artifact_environment(env, run.base_digest, run.hp, run.device)
    if "tokenizerDigest" in run.base_identity:
        env_export.update({"base_model_name": run.base_identity["id"], "tokenizer_digest": run.base_identity["tokenizerDigest"],
                           "base_revision": run.base_identity["source"]["revision"], "base_repository": run.base_identity["source"]["repository"]})
    return {"artifact": base64.b64encode(blob).decode(), "format": ARTIFACT_FORMAT, "environment": env_export, "resolvedHyperparameters": run.hp,
            "performance": performance(progress, lf.parameter_counts(run.model))}


def load_artifact(encoded, spec):
    """The adapter on a freshly built base, after lora_finetune's full validation and a match with the snapshot's settings."""
    hp = resolve_hyperparameters(spec)
    if not isinstance(encoded, str) or len(encoded) > lf.LIMITS["maxAdapterBytes"] * 4 // 3 + 4:
        raise common.IncompatibleArtifact("oversized or invalid adapter")
    try:
        blob = base64.b64decode(encoded, validate=True)
    except Exception:
        raise common.IncompatibleArtifact("unreadable adapter") from None
    metadata = lf.read_metadata(blob)
    if metadata.get("lora") != {"rank": hp["rank"], "alpha": hp["alpha"], "dropout": hp["dropout"], "targetModules": hp["targetModules"]} \
            or (metadata.get("base") or {}).get("id") != (hp.get("pretrainedModel") if hp["baseModel"] == provisioning.BASE_ID else hp["baseModel"]):
        raise common.IncompatibleArtifact("adapter does not match the snapshot's LoRA configuration")
    device = resolve_device(spec)
    qcfg = quant.settings({k: hp[k] for k in quant.DEFAULTS if k in hp})
    if metadata.get("quantization", quant.DEFAULTS) != qcfg:
        raise common.IncompatibleArtifact("Adapter quantization configuration differs")
    if qcfg["quantization"] != "none" and metadata.get("quantizationVersions") != quant.dependencies():
        raise common.IncompatibleArtifact("Adapter quantization runtime differs")
    base, identity = build_base(spec, hp)
    if metadata.get("base") != identity:
        raise common.IncompatibleArtifact("Adapter base or tokenizer identity differs")
    base = quant.load_quantized(base, qcfg) if qcfg["quantization"] != "none" else base.to(device)
    return lf.load_adapter(blob, base, identity=identity, quantization=qcfg if qcfg["quantization"] != "none" else None), hp


def predict_next_tokens(model, seqs):
    """Greedy next token after every position of each sequence (eval mode, padding ignored)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(seqs), 64):
            block = seqs[start:start + 64]
            ids, mask, _ = batch_tensors(block)
            ids, mask = (t.to(next(model.parameters()).device) for t in (ids, mask))
            best = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            out.extend(best[row, :len(s)].tolist() for row, s in enumerate(block))
    return out


def _execute(request):
    env = environment()
    spec, rows = request["snapshot"], request["rows"]
    if request.get("expectedEnvironment") is not None:
        check_compatible(request["expectedEnvironment"], env)  # recorded in PostgreSQL, checked before any artifact bytes are read
    command = request["command"]
    if command == "prepare":
        hp = resolve_hyperparameters(spec)
        resolve_device(spec)
        sequences(spec, rows, hp)  # every row is validated before any split or training
        return {**common.split_partitions(spec, rows, None), "environment": env}
    if command == "train":  # one-shot, for direct callers: a fresh run for every epoch, then the artifact
        run = fresh_run(spec, rows, request["prepared"])
        run_epochs(run, run.hp["epochs"])
        progress = {"elapsedSeconds": run.elapsed, "completedEpoch": run.completed, "optimizerSteps": run.steps, "initialTrainLoss": run.initial,
                    "lastTrainLoss": run.last_train, "lastValidationLoss": run.last_validation}
        return export(run, None, env, progress)
    if command == "train_chunk":
        prepared = {"indices": request["prepared"]["indices"]}
        epochs = request["epochs"]
        if not lf.is_int(epochs, 1, LIMITS["maxEpochs"]):
            raise ValueError("Chunk size out of range")
        run = fresh_run(spec, rows, prepared) if request.get("state") is None else restore_run(request["state"], spec, rows, prepared, env)
        if run.completed + epochs > run.hp["epochs"]:
            raise ValueError("Chunk would train past the requested epochs")
        summaries = run_epochs(run, epochs)
        env_run = artifact_environment(env, run.base_digest, run.hp, run.device)
        return {"checkpoint": save_run(run, env_run), "checkpointFormat": CHECKPOINT_FORMAT, "checkpointSchema": CHECKPOINT_SCHEMA,
                "completedEpoch": run.completed, "epochs": summaries, "environment": env_run}
    if command == "export":
        hp = resolve_hyperparameters(spec)
        resolve_device(spec)
        shell = Run(spec, hp, {"indices": {"train": [], "validation": []}}, [], [])
        shell.train = None  # no data at export: the fingerprint was checked by every chunk that resumed
        meta, tensors = validated_checkpoint(request["checkpoint"], spec, env, shell)
        shell.fingerprint = meta["fingerprint"]
        prefix = "best." if request.get("restoreBest") and meta["best"] is not None else "adapter."
        return export(shell, {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}, env, meta)
    if command == "evaluate":
        model, hp = load_artifact(request["artifact"], spec)
        names = common.requested_metrics(spec, MODELS[spec["algorithm"]], labelled=False)
        metrics = []
        for partition in ("train", "validation", "test"):
            total, count, correct = token_scores(model, sequences(spec, rows, hp, request["prepared"]["indices"][partition]))
            entropy = total / count
            values = {"token_cross_entropy": entropy, "perplexity": math.exp(min(entropy, 700.0)), "next_token_accuracy": correct / count}
            metrics.extend({"name": n, "partition": partition, "value": values[n], "direction": common.DIRECTIONS[n]} for n in names)
        return {"metrics": metrics, "performance": {}}
    if command == "predict":
        if len(rows) > LIMITS["maxPredictionRows"]:
            raise RunnerError("invalid_tokens", "Too many token sequences for one prediction")
        model, hp = load_artifact(request["artifact"], spec)
        return {"predictions": predict_next_tokens(model, sequences(spec, rows, hp))}
    raise ValueError("Unknown backend command")


def build_base(spec, hp):
    if hp["baseModel"] == provisioning.BASE_ID:
        return provisioning.load(spec["_base"]), spec["_base"]["identity"]
    config, seed = BASE_MODELS[hp["baseModel"]]
    base = lf.build_base(config, seed)
    return base, {"id": hp["baseModel"], "config": lf.base_config(base), "digest": lf.base_digest(base)}


def execute(request):
    try:
        spec = request["snapshot"]
        if any(key in spec for key in ("distillation", "teacher", "execution")) or "teacherArtifact" in request:
            raise RunnerError("unsupported_task", "This public backend supports local adapter fine-tuning only")
        hp = resolve_hyperparameters(spec)
        if hp["baseModel"] == provisioning.BASE_ID:
            with provisioning.provision(request.get("modelRoot", ""), hp, resolve_device(spec)) as context:
                return _execute({**request, "snapshot": {**spec, "_base": context}})
        return _execute(request)
    except torch.cuda.OutOfMemoryError:
        # The canonical processor fails this attempt terminally; no automatic smaller-batch retry.
        raise RunnerError("cuda_out_of_memory", "CUDA memory was exhausted") from None
    finally:
        # A request owns a fresh process. Cancellation/lease-loss kills it; the driver reclaims its allocations.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=MODELS.get))
