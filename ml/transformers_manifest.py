"""Provisioned pretrained models: the `meticule-model.json` manifest, how operators write it, and how runners verify it.

Standard library only, so operators and test fixtures can use it without any ML framework. A model directory holds the
Hugging Face files plus the manifest:
    {"schema": 1, "name", "kind", "family", "source": {"repository", "revision"}, "license"?, "files": {"<relpath>": "<sha256>"}}

`verified_copy` is the only way a runner reaches model files. Before any framework touches them it checks the name, path
containment, file types, the exact file listing against the manifest, a filename allowlist, the config, the family and kind,
and the size; then it copies every file into a fresh private directory while hashing it, so the files loaded are exactly the
files verified (no check-then-load race). Every failure raises ManifestError with a fixed code and no path in its message.
"""
import hashlib
import json
import os
import re
import shutil
import stat
import sys

MANIFEST = "meticule-model.json"
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
KEY = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Exactly the files the supported families need: weights in safetensors, configs, tokenizers and processors. No code, no
# pickled weights (.bin, .pt, .pth, .pkl, .ckpt, .h5, .msgpack), no shard indexes.
ALLOWED_FILES = frozenset({"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json",
                           "merges.txt", "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
                           # transformers v5 saves a Wav2Vec2Processor (feature extractor + CTC tokenizer) as processor_config.json.
                           "processor_config.json"})
KINDS = frozenset({"text-encoder", "text-classifier", "image-encoder", "image-classifier", "audio-classifier", "speech-ctc"})
MAX_FILES = 64
MAX_TOTAL_BYTES = 2 * 1024 ** 3
MAX_JSON_BYTES = 16 * 1024 * 1024
REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: symlinks and junctions on Windows


class ManifestError(Exception):
    """`code` is pretrained_model_unavailable (not provisioned) or pretrained_model_mismatch (anything else)."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def mismatch():
    return ManifestError("pretrained_model_mismatch")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_plain(info, directory):
    """A regular file (or, for `directory`, a real directory): never a symlink, junction or other reparse point."""
    if getattr(info, "st_file_attributes", 0) & REPARSE_POINT or stat.S_ISLNK(info.st_mode):
        return False
    return stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)


def listing(model_dir):
    """Every file under the model directory as a '/'-joined relative path. Refuses links, specials and too many entries."""
    files, pending = [], [""]
    while pending:
        relative = pending.pop()
        with os.scandir(os.path.join(model_dir, relative) if relative else model_dir) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                path = f"{relative}/{entry.name}" if relative else entry.name
                if is_plain(info, directory=True):
                    pending.append(path)
                elif is_plain(info, directory=False):
                    files.append(path)
                else:
                    raise mismatch()
                if len(files) + len(pending) > MAX_FILES + 1:
                    raise mismatch()
    return files


def read_json(path, limit=MAX_JSON_BYTES):
    if os.path.getsize(path) > limit:
        raise mismatch()
    with open(path, "rb") as handle:
        value = json.loads(handle.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise mismatch()
    return value


def canonical_digest(kind, family, files):
    return hashlib.sha256(json.dumps({"kind": kind, "family": family, "files": files}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_manifest(manifest, name):
    if set(manifest) - {"schema", "name", "kind", "family", "source", "license", "files"} or manifest.get("schema") != 1 or manifest.get("name") != name:
        raise mismatch()
    source, files = manifest.get("source"), manifest.get("files")
    if manifest.get("kind") not in KINDS or not isinstance(manifest.get("family"), str) or not isinstance(source, dict) or set(source) != {"repository", "revision"}:
        raise mismatch()
    if not all(isinstance(source[k], str) and 0 < len(source[k]) <= 200 for k in source) or ("license" in manifest and not isinstance(manifest["license"], str)):
        raise mismatch()
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise mismatch()
    for key, digest in files.items():
        segments = key.split("/") if isinstance(key, str) else []
        if (not KEY.match(key) or any(s in (".", "..") for s in segments) or segments[-1].casefold() not in ALLOWED_FILES
                or not isinstance(digest, str) or not SHA256.match(digest)):
            raise mismatch()
    if len({k.casefold() for k in files}) != len(files) or "config.json" not in files:
        raise mismatch()


def verified_copy(root, name, kinds, families, destination):
    """Verifies the provisioned model `name` and copies it into `destination` (an empty private directory).

    Returns {"name", "kind", "family", "source", "license", "digest"}; never a path."""
    if not isinstance(name, str) or not NAME.match(name):
        raise mismatch()
    if not isinstance(root, str) or not os.path.isabs(root) or not os.path.isdir(root):
        raise ManifestError("pretrained_model_unavailable")
    real_root = os.path.realpath(root)
    model_dir = os.path.join(real_root, name)
    try:
        info = os.lstat(model_dir)
    except FileNotFoundError:
        raise ManifestError("pretrained_model_unavailable") from None
    if not is_plain(info, directory=True) or os.path.dirname(os.path.realpath(model_dir)) != real_root:
        raise mismatch()
    try:
        found = listing(model_dir)
        manifest = read_json(os.path.join(model_dir, MANIFEST))
        check_manifest(manifest, name)
        files = manifest["files"]
        if len({f.casefold() for f in found}) != len(found) or {f.casefold() for f in found} != {k.casefold() for k in files} | {MANIFEST}:
            raise mismatch()
        if manifest["kind"] not in kinds or manifest["family"] not in families:
            raise mismatch()
        config = read_json(os.path.join(model_dir, "config.json"))
        if "auto_map" in config or "transformers_weights" in config or config.get("model_type") != manifest["family"]:
            raise mismatch()
        if sum(os.path.getsize(os.path.join(model_dir, *k.split("/"))) for k in files) > MAX_TOTAL_BYTES:
            raise mismatch()
        for key, expected in files.items():
            target = os.path.join(destination, *key.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            digest = hashlib.sha256()
            with open(os.path.join(model_dir, *key.split("/")), "rb") as source, open(target, "xb") as copy:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    copy.write(block)
            if digest.hexdigest() != expected:
                raise mismatch()
    except ManifestError:
        raise
    except Exception:
        raise mismatch() from None
    return {"name": name, "kind": manifest["kind"], "family": manifest["family"], "source": dict(manifest["source"]),
            "license": manifest.get("license"), "digest": canonical_digest(manifest["kind"], manifest["family"], files)}


def write_manifest(model_dir, kind, family, repository, revision, license=None):
    """Operator helper: hashes every file in a model directory (named by its final path segment) and writes the manifest."""
    name = os.path.basename(os.path.normpath(model_dir))
    files = {path: sha256_file(os.path.join(model_dir, *path.split("/"))) for path in sorted(listing(model_dir)) if path != MANIFEST}
    manifest = {"schema": 1, "name": name, "kind": kind, "family": family, "source": {"repository": repository, "revision": revision}, "files": files}
    if license is not None:
        manifest["license"] = license
    check_manifest(manifest, name)
    with open(os.path.join(model_dir, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


if __name__ == "__main__":
    # python transformers_manifest.py <model_dir> <kind> <family> <repository> <revision> [license]
    if len(sys.argv) not in (6, 7):
        sys.exit("usage: transformers_manifest.py <model_dir> <kind> <family> <repository> <revision> [license]")
    print(json.dumps(write_manifest(*sys.argv[1:]), indent=2))
