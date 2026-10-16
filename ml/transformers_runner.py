"""Hugging Face Transformers backend (CPU, fp32): pretrained text, vision and audio models provisioned under a platform-wide
model root, used as fixed classifiers, encoders and a speech recogniser, or as frozen encoders under a trained softmax head
(linear_heads). Audio clips are decoded by audio_common (the media_common walker) and resampled to the model's rate.

The runner never downloads anything and never runs remote code. Every command that touches a model verifies the
provisioned directory against its manifest and loads only a private per-call copy (transformers_manifest.verified_copy).
Models load with concrete classes, safetensors only, fp32, and any missing or mismatched weight is refused instead of being
initialised at random. The artifact (`transformers-reference-v1`) is JSON: a model reference with its verified digest,
labels and the trained head; never the base weights. Loading an artifact re-verifies the provisioned model's digest.
"""
import atexit
import os
import shutil
import tempfile

# Hugging Face caches go to a per-call empty directory (removed at exit), and the hub stays offline, before transformers
# is imported. The TypeScript runner environment sets the offline variables too.
_CACHE = tempfile.mkdtemp(prefix="meticule-hf-")
atexit.register(shutil.rmtree, _CACHE, True)
os.environ.update({"HF_HOME": _CACHE, "HF_HUB_CACHE": os.path.join(_CACHE, "hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})

import base64
import binascii
import gc
import io
import json
import sys
import time
import warnings

import joblib
import numpy as np
import PIL
import safetensors
import scipy
import scipy.special
import sklearn
import threadpoolctl
import tokenizers
import torch
import transformers
from PIL import Image
from transformers import (BertForSequenceClassification, BertModel, BertTokenizer, DistilBertForSequenceClassification, DistilBertModel,
                          DistilBertTokenizer, ViTForImageClassification, ViTImageProcessorPil, ViTModel, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor)

import audio_common
import linear_heads
import media_common
import ml_runtime_common as common
import transformers_manifest

transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
Image.MAX_IMAGE_PIXELS = 4096 * 4096

BACKEND = "transformers"
ARTIFACT_FORMAT = "transformers-reference-v1"
ARTIFACT_SCHEMA = 1
PROTOCOL = "meticule-transformers-v1"
RUNTIME_KEYS = ("torch", "transformers", "tokenizers", "safetensors", "pillow", "numpy", "scipy", "sklearn", "joblib", "threadpoolctl")
CONTRACT = {"backend": BACKEND, "artifact_format": ARTIFACT_FORMAT, "protocols": {PROTOCOL: ARTIFACT_SCHEMA}, "schemas": {ARTIFACT_SCHEMA},
            "runtime_keys": RUNTIME_KEYS}
MAX_PARAMETERS = 400_000_000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
RunnerError = common.RunnerError

TEXT_FAMILIES, IMAGE_FAMILIES, AUDIO_FAMILIES = frozenset({"bert", "distilbert"}), frozenset({"vit"}), frozenset({"wav2vec2"})
FAMILIES = {"text": TEXT_FAMILIES, "image": IMAGE_FAMILIES, "audio": AUDIO_FAMILIES}
MAX_TRANSCRIPT_CHARS = 10_000
# Mirrors src/ml/transformers-models.ts.
MODELS = {
    "text_classifier": {"task": "classification", "modality": "text", "learning": "pretrained", "kind": "text-classifier"},
    "text_embedder": {"task": "embedding", "modality": "text", "learning": "pretrained", "kind": "text-encoder"},
    "text_embedding_classifier": {"task": "classification", "modality": "text", "learning": "supervised", "kind": "text-encoder"},
    "image_classifier": {"task": "classification", "modality": "image", "learning": "pretrained", "kind": "image-classifier"},
    "image_embedder": {"task": "embedding", "modality": "image", "learning": "pretrained", "kind": "image-encoder"},
    "image_embedding_classifier": {"task": "classification", "modality": "image", "learning": "supervised", "kind": "image-encoder"},
    "audio_classifier": {"task": "classification", "modality": "audio", "learning": "pretrained", "kind": "audio-classifier"},
    "speech_recognizer": {"task": "transcription", "modality": "audio", "learning": "pretrained", "kind": "speech-ctc"},
}
# Concrete classes per (kind, family); encoders load without a pooling layer, since embeddings come from the hidden states.
CLASSES = {("text-encoder", "bert"): (BertModel, {"add_pooling_layer": False}), ("text-encoder", "distilbert"): (DistilBertModel, {}),
           ("text-classifier", "bert"): (BertForSequenceClassification, {}), ("text-classifier", "distilbert"): (DistilBertForSequenceClassification, {}),
           ("image-encoder", "vit"): (ViTModel, {"add_pooling_layer": False}), ("image-classifier", "vit"): (ViTForImageClassification, {}),
           ("audio-classifier", "wav2vec2"): (Wav2Vec2ForSequenceClassification, {}), ("speech-ctc", "wav2vec2"): (Wav2Vec2ForCTC, {})}
TOKENIZERS = {"bert": BertTokenizer, "distilbert": DistilBertTokenizer}
# name: (kind, low, high, default) or (kind, choices, default). Mirrors the TypeScript catalog; enforced again here.
SHARED = {"model": ("name",), "batchSize": ("int", 1, 64, 16)}
TEXT = {"maxLength": ("int", 8, 512, 128)}
HEAD = {"l2": ("float", 0.0, 100.0, 1.0), "maxIter": ("int", 10, 1000, 200)}
PARAMETERS = {
    "text_classifier": {**SHARED, **TEXT},
    "text_embedder": {**SHARED, **TEXT, "pooling": ("choice", ("mean", "cls"), "mean"), "normalize": ("bool", True)},
    "text_embedding_classifier": {**SHARED, **TEXT, "pooling": ("choice", ("mean", "cls"), "mean"), **HEAD},
    "image_classifier": dict(SHARED),
    "image_embedder": {**SHARED, "pooling": ("choice", ("cls", "mean"), "cls"), "normalize": ("bool", True)},
    "image_embedding_classifier": {**SHARED, "pooling": ("choice", ("cls", "mean"), "cls"), **HEAD},
    # Audio is forwarded one clip at a time (forward_audio), so it has no batch size.
    "audio_classifier": {"model": ("name",)},
    "speech_recognizer": {"model": ("name",)},
}
FIXED = {"pretrained_model_unavailable": "The pretrained model is not provisioned",
         "pretrained_model_mismatch": "The provisioned model does not match its manifest or this model"}


def environment():
    return common.base_environment([__file__, common.__file__, media_common.__file__, audio_common.__file__, linear_heads.__file__, transformers_manifest.__file__],
                                   BACKEND, ARTIFACT_FORMAT, ARTIFACT_SCHEMA, PROTOCOL,
                                   {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                                    "safetensors": safetensors.__version__, "pillow": PIL.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                                    "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__})


def resolve_hyperparameters(spec):
    algorithm = spec["algorithm"]
    supplied, allowed = spec.get("hyperparameters") or {}, PARAMETERS[algorithm]
    if set(supplied) - set(allowed):
        raise ValueError("Unsupported transformers hyperparameter")
    resolved = {}
    for name, rule in allowed.items():
        value = supplied.get(name, None if rule[0] == "name" else rule[-1])
        kind = rule[0]
        if kind == "name":
            if not isinstance(value, str) or not transformers_manifest.NAME.match(value):
                raise ValueError("A provisioned model name is required")
        elif kind == "int":
            if not isinstance(value, int) or isinstance(value, bool) or not rule[1] <= value <= rule[2]:
                raise ValueError(f"Hyperparameter {name} out of range")
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not rule[1] <= float(value) <= rule[2]:
                raise ValueError(f"Hyperparameter {name} out of range")
            value = float(value)
        elif kind == "choice":
            if value not in rule[1]:
                raise ValueError(f"Hyperparameter {name} out of range")
        elif not isinstance(value, bool):
            raise ValueError(f"Hyperparameter {name} out of range")
        resolved[name] = value
    return resolved


def resolve_device(spec):
    training = spec.get("training") or {}
    if training.get("device", "cpu") == "cuda":
        raise RunnerError("cuda_unsupported", "CUDA is not supported by the transformers backend")
    if training.get("device", "cpu") != "cpu" or training.get("precision", "fp32") != "fp32":
        raise RunnerError("unsupported_device_or_precision", "Unsupported device or precision")


def feature(spec):
    """The one feature column, which must be declared with the model's modality (never a tabular column read as text)."""
    features = spec["pipeline"]["definition"]["features"]
    if len(features) != 1 or spec["pipeline"]["definition"]["steps"]:
        raise ValueError("Transformers models read exactly one feature column and take no preprocessing steps")
    types = {c["name"]: c["type"] for c in spec["dataset"]["schema"]["columns"]}
    if types.get(features[0]) != MODELS[spec["algorithm"]]["modality"]:
        raise ValueError("The feature column type does not match the model's modality")
    return features[0]


# ---- loading a provisioned model -----------------------------------------------------------------------------------

class Pretrained:
    """A verified, loaded model and its tokenizer or image processor, with the manifest facts recorded about it."""

    def __init__(self, root, algorithm, name, seed):
        entry = MODELS[algorithm]
        self.work = tempfile.mkdtemp(prefix="meticule-model-")
        families = FAMILIES[entry["modality"]]
        try:
            self.info = transformers_manifest.verified_copy(root, name, {entry["kind"]}, families, self.work)
        except transformers_manifest.ManifestError as error:
            self.close()
            if error.code == "pretrained_model_unavailable":
                raise RunnerError("pretrained_model_unavailable", FIXED["pretrained_model_unavailable"]) from None
            raise RunnerError("pretrained_model_mismatch", FIXED["pretrained_model_mismatch"]) from None
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        cls, options = CLASSES[(self.info["kind"], self.info["family"])]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model, report = cls.from_pretrained(self.work, local_files_only=True, use_safetensors=True, trust_remote_code=False,
                                                         dtype=torch.float32, output_loading_info=True, **options)
            unexpected = [k for k in report["unexpected_keys"] if not k.startswith("pooler.")]
            if report["missing_keys"] or report["mismatched_keys"] or report["error_msgs"] or unexpected:
                raise ValueError
            if sum(p.numel() for p in self.model.parameters()) > MAX_PARAMETERS or any(p.dtype != torch.float32 for p in self.model.parameters()):
                raise ValueError
            self.model.eval()
            if entry["modality"] == "text":
                self.tokenizer = TOKENIZERS[self.info["family"]].from_pretrained(self.work, local_files_only=True)
            elif entry["modality"] == "image":
                self.processor = ViTImageProcessorPil.from_pretrained(self.work, local_files_only=True)
            elif self.info["kind"] == "speech-ctc":
                self.processor = Wav2Vec2Processor.from_pretrained(self.work, local_files_only=True)
                self.rate = int(self.processor.feature_extractor.sampling_rate)
            else:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.work, local_files_only=True)
                self.rate = int(self.processor.sampling_rate)
        except RunnerError:
            raise
        except Exception:
            self.close()
            raise RunnerError("pretrained_model_mismatch", FIXED["pretrained_model_mismatch"]) from None
        self.config = self.model.config

    def labels(self):
        count = int(self.config.num_labels)
        return [str(self.config.id2label[i]) for i in range(count)]

    def max_length(self, requested):
        """Truncation length: the smallest of the tokenizer's limit, the position-embedding table and the request."""
        return int(min(self.tokenizer.model_max_length, self.config.max_position_embeddings, requested))

    def close(self):
        self.__dict__.pop("model", None)
        gc.collect()
        shutil.rmtree(self.work, ignore_errors=True)


def open_image(cell):
    """Header-validated by media_common first; then Pillow with PNG/JPEG only, a pixel cap, and decompression-bomb
    warnings as errors. Any failure is invalid_media."""
    data = media_common.decode_base64(cell)
    _, width, height = media_common.image_header(data)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data), formats=["PNG", "JPEG"])
            if image.size != (width, height):
                raise ValueError
            image.load()
            return image.convert("RGB")
    except Exception:
        raise RunnerError("invalid_media", "Image could not be decoded") from None


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def forward_text(loaded, texts, hp):
    """Hidden states or logits for each batch, with truncation always on."""
    for batch in batches(texts, hp["batchSize"]):
        encoded = loaded.tokenizer([str(t) for t in batch], truncation=True, max_length=loaded.max_length(hp.get("maxLength", 512)), padding=True, return_tensors="pt")
        with torch.no_grad():
            yield loaded.model(**encoded), encoded["attention_mask"]


def forward_images(loaded, cells, hp):
    for batch in batches(cells, hp["batchSize"]):
        pixels = torch.cat([loaded.processor(images=open_image(cell), return_tensors="pt")["pixel_values"] for cell in batch])
        with torch.no_grad():
            yield loaded.model(pixel_values=pixels), None  # images have no padding: every patch token is real


def clips(loaded, cells):
    """Mono float32 clips at the model's sampling rate; the walker enforces the 0.1..30 s and format rules (C3)."""
    if not 8000 <= loaded.rate <= 48000:
        raise RunnerError("pretrained_model_mismatch", FIXED["pretrained_model_mismatch"])
    return [audio_common.decode(cell, loaded.rate) for cell in cells]


def forward_audio(loaded, cells, hp):
    """One clip per forward pass: zero padding changes a group-norm wav2vec2's outputs, so a batched clip's scores would
    depend on the other clips in its request."""
    for clip in clips(loaded, cells):
        inputs = loaded.processor(clip, sampling_rate=loaded.rate, return_tensors="pt")
        with torch.no_grad():
            yield loaded.model(input_values=inputs["input_values"]), None


FORWARD = {"text": forward_text, "image": forward_images, "audio": forward_audio}


def transcribe(loaded, cells):
    """Greedy CTC decoding, one clip per forward pass."""
    transcripts = []
    for clip in clips(loaded, cells):
        inputs = loaded.processor(clip, sampling_rate=loaded.rate, return_tensors="pt")
        with torch.no_grad():
            ids = loaded.model(input_values=inputs["input_values"]).logits.argmax(dim=-1)
        text = loaded.processor.batch_decode(ids)[0]
        if len(text) > MAX_TRANSCRIPT_CHARS:
            raise ValueError("Transcript exceeds the output bound")
        transcripts.append(text)
    return transcripts


def embeddings(loaded, values, hp, modality):
    """Pooled hidden states: mean over real tokens (text: attention mask; images: every patch token) or the first token."""
    if not values:
        return np.zeros((0, int(getattr(loaded.config, "hidden_size", getattr(loaded.config, "dim", 0)))))
    chunks = []
    for output, mask in FORWARD[modality](loaded, values, hp):
        hidden = output.last_hidden_state
        if hp["pooling"] == "cls":
            pooled = hidden[:, 0]
        else:
            weights = torch.ones(hidden.shape[:2]) if mask is None else mask
            weights = weights.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        chunks.append(pooled.numpy().astype(np.float64))
    vectors = np.vstack(chunks)
    if not np.isfinite(vectors).all():
        raise ValueError("Model produced non-finite embeddings")
    if hp.get("normalize"):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Cannot normalise a zero embedding")
        vectors = vectors / norms
    return vectors


def class_probabilities(loaded, values, hp, modality):
    """The pretrained head's softmax over its own labels."""
    if not values:
        return np.zeros((0, len(loaded.labels())))
    chunks = [scipy.special.softmax(output.logits.numpy().astype(np.float64), axis=1) for output, _ in FORWARD[modality](loaded, values, hp)]
    return np.vstack(chunks)


# ---- the reference artifact ------------------------------------------------------------------------------------------

def serialize(document):
    return base64.b64encode(json.dumps(document, allow_nan=False, sort_keys=True).encode("utf-8")).decode()


def read_artifact(artifact, env):
    try:
        raw = base64.b64decode(artifact, validate=True)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise common.IncompatibleArtifact("unreadable artifact") from None
    keys = {"schema", "backend", "task", "algorithm", "model", "labels", "maxLength", "pooling", "normalize", "embeddingDimension", "head", "environment"}
    if not isinstance(document, dict) or set(document) != keys:
        raise common.IncompatibleArtifact("malformed artifact")
    common.check_compatible(document["environment"], env, CONTRACT)
    model = document["model"]
    if (document["schema"] != ARTIFACT_SCHEMA or document["backend"] != BACKEND or document["algorithm"] not in MODELS
            or document["task"] != MODELS[document["algorithm"]]["task"] or not isinstance(model, dict)
            or set(model) != {"name", "kind", "family", "repository", "revision", "digest"}):
        raise common.IncompatibleArtifact("malformed artifact")
    return document


class Bundle:
    """An artifact made usable: the model it references, re-verified against its recorded digest, and its head."""

    def __init__(self, document, root, seed):
        self.document = document
        self.loaded = Pretrained(root, document["algorithm"], document["model"]["name"], seed)
        if self.loaded.info["digest"] != document["model"]["digest"]:
            self.loaded.close()
            raise RunnerError("pretrained_model_mismatch", FIXED["pretrained_model_mismatch"])
        self.head = None
        if document["head"] is not None:
            try:
                head = document["head"]
                self.classes = list(head["classes"])
                self.head = linear_heads.decode(head["arrays"], int(document["embeddingDimension"]), len(self.classes))
            except Exception:
                raise common.IncompatibleArtifact("malformed artifact") from None

    def hp(self):
        d = self.document
        return {"batchSize": 16, "maxLength": d["maxLength"] or 512, "pooling": d["pooling"], "normalize": d["normalize"]}


def outputs(bundle, values):
    """Predictions (labels or vectors) plus, for classifiers, probabilities in sorted class order for metrics."""
    algorithm = bundle.document["algorithm"]
    entry, hp = MODELS[algorithm], bundle.hp()
    if entry["task"] == "transcription":
        return {"transcripts": transcribe(bundle.loaded, values)}
    if entry["task"] == "embedding":
        vectors = embeddings(bundle.loaded, values, hp, entry["modality"])
        return {"vectors": vectors}
    if bundle.head is not None:
        proba = linear_heads.predict_proba(bundle.head, embeddings(bundle.loaded, values, hp, entry["modality"])) if values else np.zeros((0, len(bundle.classes)))
        classes = bundle.classes
    else:
        labels = bundle.loaded.labels()
        raw = class_probabilities(bundle.loaded, values, hp, entry["modality"])
        order = np.argsort(np.array(labels))
        proba, classes = raw[:, order], [labels[i] for i in order]
    prediction = [classes[i] for i in proba.argmax(axis=1)] if len(proba) else []
    return {"prediction": np.array(prediction), "proba": proba, "classes": np.array(classes)}


def measure(names, task, y, labelled, result):
    if task == "transcription":
        return {name: (common.wer if name == "wer" else common.cer)(list(y), result["transcripts"]) for name in names}
    if task == "embedding":
        return common.embedding_metrics(names, result["vectors"], y if labelled else None)
    return common.measure_predictions(names, y, result["prediction"], proba=result["proba"], scores=result["proba"], classes=result["classes"])


def execute(request):
    env = environment()
    spec = request["snapshot"]
    rows = request["rows"]
    if request.get("expectedEnvironment") is not None:
        common.check_compatible(request["expectedEnvironment"], env, CONTRACT)
    command = request["command"]
    algorithm = spec["algorithm"]
    entry = MODELS[algorithm]
    target = spec["dataset"]["schema"].get("target")
    labelled = target is not None
    column = feature(spec)
    resolve_device(spec)
    hp = resolve_hyperparameters(spec)
    root = request.get("modelRoot")
    if command == "prepare":
        Pretrained(root, algorithm, hp["model"], spec["seed"]).close()  # refuse an unprovisioned or mismatched model before any work
        y = np.array([row[target] for row in rows]) if labelled else None
        return {**common.split_partitions(spec, rows, y), "environment": env}
    if command == "train":
        started = time.perf_counter()
        indices = request["prepared"]["indices"]
        loaded = Pretrained(root, algorithm, hp["model"], spec["seed"])
        try:
            info = loaded.info
            document = {"schema": ARTIFACT_SCHEMA, "backend": BACKEND, "task": entry["task"], "algorithm": algorithm,
                        "model": {"name": info["name"], "kind": info["kind"], "family": info["family"], "repository": info["source"]["repository"],
                                  "revision": info["source"]["revision"], "digest": info["digest"]},
                        "labels": None, "maxLength": loaded.max_length(hp["maxLength"]) if entry["modality"] == "text" else None,
                        "pooling": hp.get("pooling"), "normalize": bool(hp.get("normalize", False)), "embeddingDimension": None, "head": None}
            provenance = {"model_name": info["name"], "model_kind": info["kind"], "model_family": info["family"], "model_digest": info["digest"],
                          "model_repository": info["source"]["repository"], "model_revision": info["source"]["revision"]}
            if entry["learning"] == "pretrained" and entry["task"] == "classification":
                # Binding a pretrained head: the dataset's classes must be labels the model already predicts.
                labels = loaded.labels()
                if not {str(row[target]) for row in rows} <= set(labels) or any(not isinstance(row[target], str) for row in rows):
                    raise RunnerError("pretrained_label_mismatch", "Dataset classes are not labels of the pretrained model")
                document["labels"] = labels
            elif entry["task"] == "transcription":
                # A pretrained recogniser fits nothing; the targets are the reference transcripts its metrics compare with.
                if any(not isinstance(row[target], str) for row in rows):
                    raise ValueError("Reference transcripts must be text")
            elif entry["task"] == "embedding":
                sample = [rows[indices["train"][0]][column]]
                document["embeddingDimension"] = int(embeddings(loaded, sample, hp, entry["modality"]).shape[1])
                provenance["embedding_dimension"] = str(document["embeddingDimension"])
            else:
                train_rows = [rows[i] for i in indices["train"]]
                classes = sorted({row[target] for row in train_rows})
                if len(classes) < 2:
                    raise ValueError("Classification needs at least two classes in the training partition")
                features = embeddings(loaded, [row[column] for row in train_rows], hp, entry["modality"])
                head = linear_heads.fit(features, [classes.index(row[target]) for row in train_rows], len(classes), hp["l2"], hp["maxIter"])
                document["embeddingDimension"] = int(features.shape[1])
                document["head"] = {"classes": classes, "arrays": linear_heads.encode(head)}
        finally:
            loaded.close()
        trained_env = {**env, **provenance}
        document["environment"] = trained_env
        return {"artifact": serialize(document), "format": ARTIFACT_FORMAT, "environment": trained_env, "resolvedHyperparameters": hp,
                "performance": {"fitSeconds": time.perf_counter() - started}}
    if command in ("evaluate", "predict"):
        bundle = Bundle(read_artifact(request["artifact"], env), root, spec["seed"])
        try:
            if command == "predict":
                result = outputs(bundle, [row[column] for row in rows])
                if entry["task"] == "transcription":
                    return {"predictions": result["transcripts"]}
                return {"predictions": result["vectors"].tolist() if entry["task"] == "embedding" else result["prediction"].tolist()}
            names = common.requested_metrics(spec, entry["task"], labelled)
            y = np.array([row[target] for row in rows]) if labelled else None
            prepared = request["prepared"]["indices"]
            metrics, performance = [], {}

            def emit(partition, values, **extra):
                metrics.extend({"name": n, "partition": partition, "value": v, "direction": common.DIRECTIONS[n], **extra} for n, v in values.items())

            for partition in ("train", "validation", "test"):
                indices = prepared[partition]
                started = time.perf_counter()
                emit(partition, measure(names, entry["task"], y[indices] if labelled else None, labelled, outputs(bundle, [rows[i][column] for i in indices])))
                if partition == "test":
                    performance["predictSeconds"] = time.perf_counter() - started
            cv = (spec.get("evaluation") or {}).get("cv")
            if cv:
                if entry["learning"] == "pretrained":
                    raise ValueError("Cross-validation is not supported for pretrained models")
                # The encoder is frozen, so every row is embedded once; each fold fits a fresh standardisation and head on its own rows.
                pool = np.array(prepared["train"])
                features = embeddings(bundle.loaded, [rows[i][column] for i in pool], bundle.hp(), entry["modality"])
                position = {int(i): k for k, i in enumerate(pool)}
                per_fold, fit_total = [], 0.0
                for number, held_out in enumerate(request["prepared"]["folds"]):
                    fit_rows = [int(i) for i in pool[~np.isin(pool, held_out)]]
                    classes = sorted({rows[i][target] for i in fit_rows})
                    started = time.perf_counter()
                    head = linear_heads.fit(features[[position[i] for i in fit_rows]], [classes.index(rows[i][target]) for i in fit_rows], len(classes), hp["l2"], hp["maxIter"])
                    fit_total += time.perf_counter() - started
                    proba = linear_heads.predict_proba(head, features[[position[int(i)] for i in held_out]])
                    result = {"prediction": np.array([classes[i] for i in proba.argmax(axis=1)]), "proba": proba, "classes": np.array(classes)}
                    values = common.measure_predictions(names, y[held_out], result["prediction"], proba=proba, scores=proba, classes=result["classes"])
                    per_fold.append(values)
                    emit("cv_fold", values, fold=number)
                for n in names:
                    series = np.array([f[n] for f in per_fold])
                    metrics.append({"name": n, "partition": "cv", "value": float(series.mean()), "direction": common.DIRECTIONS[n], "std": float(series.std())})
                performance["cvFitSeconds"] = fit_total
                performance["cvModelsBuilt"] = float(len(per_fold))
            return {"metrics": metrics, "performance": performance}
        finally:
            bundle.loaded.close()
    raise ValueError("Unknown backend command")


def task_of(algorithm):
    return MODELS[algorithm]["task"] if algorithm in MODELS else None


if __name__ == "__main__":
    sys.exit(common.serve(execute, task_of=task_of))
