"""Tiny, seeded Hugging Face fixtures for the transformers backend tests, built offline at test time. Nothing is downloaded.

    python hf_fixtures.py <model_root>      builds every fixture under <model_root> and prints a JSON summary

Good models: tiny BERT/DistilBERT encoders and sequence classifiers (labels negative/positive) with a small WordPiece vocabulary
and model_max_length set; tiny ViT encoder and classifier (labels red/blue) with a 32x32 PIL image processor; a BERT encoder
whose config says bfloat16 (it must still run in fp32). Malicious variants of a good encoder exercise every provisioning refusal.
These models are random: they prove the execution path, not semantic quality.
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import (BertConfig, BertForSequenceClassification, BertModel, BertTokenizer, DistilBertConfig,  # noqa: E402
                          DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer, ViTConfig, ViTForImageClassification,
                          ViTImageProcessorPil, ViTModel)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import transformers_manifest  # noqa: E402

transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
WORDS = ["good", "great", "happy", "love", "excellent", "bad", "awful", "sad", "hate", "terrible", "the", "movie", "was", "a", "film", "it",
         "very", "not", "red", "blue", "and", "plot", "acting"]
VOCAB = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
SENTIMENT = {0: "negative", 1: "positive"}
COLOURS = {0: "red", 1: "blue"}


def save_text(directory, model, tokenizer_class):
    model.save_pretrained(directory)
    tokenizer_class(vocab=dict(VOCAB), model_max_length=48).save_pretrained(directory)


def text_model(root, name, family, kind, seed, dtype=None):
    torch.manual_seed(seed)
    labels = {"id2label": SENTIMENT, "label2id": {v: k for k, v in SENTIMENT.items()}} if kind == "text-classifier" else {}
    if family == "bert":
        config = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37, max_position_embeddings=64, **labels)
        model = BertForSequenceClassification(config) if kind == "text-classifier" else BertModel(config, add_pooling_layer=False)
        tokenizer = BertTokenizer
    else:
        config = DistilBertConfig(vocab_size=len(VOCAB), dim=32, n_layers=2, n_heads=2, hidden_dim=37, max_position_embeddings=64, **labels)
        model = DistilBertForSequenceClassification(config) if kind == "text-classifier" else DistilBertModel(config)
        tokenizer = DistilBertTokenizer
    directory = os.path.join(root, name)
    save_text(directory, model, tokenizer)
    if dtype:
        rewrite_config(directory, lambda c: c.update({"dtype": dtype}))
    transformers_manifest.write_manifest(directory, kind, family, f"meticule-fixtures/{name}", "0" * 40, "apache-2.0")
    return directory


def image_model(root, name, kind, seed):
    torch.manual_seed(seed)
    labels = {"id2label": COLOURS, "label2id": {v: k for k, v in COLOURS.items()}} if kind == "image-classifier" else {}
    config = ViTConfig(image_size=32, patch_size=8, num_channels=3, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37, **labels)
    model = ViTForImageClassification(config) if kind == "image-classifier" else ViTModel(config, add_pooling_layer=False)
    directory = os.path.join(root, name)
    model.save_pretrained(directory)
    ViTImageProcessorPil(size={"height": 32, "width": 32}).save_pretrained(directory)
    transformers_manifest.write_manifest(directory, kind, "vit", f"meticule-fixtures/{name}", "0" * 40, "apache-2.0")
    return directory


def rewrite_config(directory, change):
    path = os.path.join(directory, "config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    change(config)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)


def rewrite_manifest(directory, change):
    path = os.path.join(directory, transformers_manifest.MANIFEST)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    change(manifest)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def variant(root, source, name):
    """A copy of a good model under a new name, re-manifested so that only the injected defect differs."""
    directory = os.path.join(root, name)
    shutil.copytree(source, directory)
    rewrite_manifest(directory, lambda m: m.update({"name": name}))
    return directory


def build(root):
    os.makedirs(root, exist_ok=True)
    good = {
        "tiny-bert-encoder": text_model(root, "tiny-bert-encoder", "bert", "text-encoder", 1),
        "tiny-distilbert-encoder": text_model(root, "tiny-distilbert-encoder", "distilbert", "text-encoder", 2),
        "tiny-bert-sentiment": text_model(root, "tiny-bert-sentiment", "bert", "text-classifier", 3),
        "tiny-distilbert-sentiment": text_model(root, "tiny-distilbert-sentiment", "distilbert", "text-classifier", 4),
        "tiny-bert-bf16-encoder": text_model(root, "tiny-bert-bf16-encoder", "bert", "text-encoder", 5, dtype="bfloat16"),
        "tiny-vit-encoder": image_model(root, "tiny-vit-encoder", "image-encoder", 6),
        "tiny-vit-colours": image_model(root, "tiny-vit-colours", "image-classifier", 7),
    }
    encoder = good["tiny-bert-encoder"]
    evil, skipped = {}, []
    # Files on disk that the manifest does not list.
    for name, filename, content in (("evil-unlisted-tokens", "added_tokens.json", b"{}"), ("evil-python-file", "modeling_evil.py", b"import os\n"),
                                    ("evil-pickle-bin", "pytorch_model.bin", b"\x80\x04"), ("evil-uppercase-bin", "PYTORCH_MODEL.BIN", b"\x80\x04")):
        directory = variant(root, encoder, name)
        with open(os.path.join(directory, filename), "wb") as handle:
            handle.write(content)
        evil[name] = "listing"
    directory = variant(root, encoder, "evil-symlink")
    try:
        os.symlink(os.path.join(directory, "config.json"), os.path.join(directory, "special_tokens_map.json"))
        rewrite_manifest(directory, lambda m: m["files"].update({"special_tokens_map.json": transformers_manifest.sha256_file(os.path.join(directory, "config.json"))}))
        evil["evil-symlink"] = "symlink"
    except OSError:
        shutil.rmtree(directory)
        skipped.append("evil-symlink: this OS refused to create a symbolic link without elevated rights")
    if os.name == "nt":
        # A directory junction is a reparse point too, and Windows creates one without elevated rights.
        directory = variant(root, encoder, "evil-junction")
        outside = os.path.join(root, "outside-the-model")
        os.makedirs(outside, exist_ok=True)
        import subprocess
        if subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(directory, "nested"), outside], capture_output=True).returncode == 0:
            evil["evil-junction"] = "reparse point"
        else:
            shutil.rmtree(directory)
            skipped.append("evil-junction: mklink /J failed")
    directory = variant(root, encoder, "evil-hash-mismatch")
    with open(os.path.join(directory, "model.safetensors"), "r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF]))
    evil["evil-hash-mismatch"] = "hash"
    directory = variant(root, encoder, "evil-wrong-family")
    rewrite_manifest(directory, lambda m: m.update({"family": "distilbert"}))
    evil["evil-wrong-family"] = "family"
    directory = variant(root, encoder, "evil-auto-map")
    rewrite_config(directory, lambda c: c.update({"auto_map": {"AutoModel": "modeling_evil.EvilModel"}}))
    rewrite_manifest(directory, lambda m: m["files"].update({"config.json": transformers_manifest.sha256_file(os.path.join(directory, "config.json"))}))
    evil["evil-auto-map"] = "config"
    directory = variant(root, encoder, "evil-sharded")
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"weight_map": {}}, handle)
    rewrite_manifest(directory, lambda m: m["files"].update({"model.safetensors.index.json": transformers_manifest.sha256_file(os.path.join(directory, "model.safetensors.index.json"))}))
    evil["evil-sharded"] = "allowlist"
    directory = variant(root, encoder, "evil-dotdot-key")
    rewrite_manifest(directory, lambda m: m["files"].update({"../config.json": m["files"]["config.json"]}))
    evil["evil-dotdot-key"] = "key"
    directory = variant(root, good["tiny-bert-sentiment"], "evil-wrong-kind")  # a classifier offered where an encoder is expected
    rewrite_manifest(directory, lambda m: m.update({"kind": "text-encoder"}))
    evil["evil-wrong-kind"] = "weights"
    return {"root": root, "good": sorted(good), "evil": evil, "skipped": skipped}


if __name__ == "__main__":
    print(json.dumps(build(os.path.abspath(sys.argv[1]))))
