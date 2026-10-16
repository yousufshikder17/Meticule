"""Real Hugging Face Transformers backend checks over tiny offline fixtures. Prints one JSON object of named results.

Builds the fixtures (tests/python/hf_fixtures.py) in a temporary model root, then drives the runner in-process. The tiny
models are random: these checks prove the execution path, provisioning safety and output contracts, not semantic quality.
Processor integration is in tests/integration/ml-transformers.test.ts.
"""
import base64
import io
import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENTINEL_HOME = tempfile.mkdtemp(prefix="meticule-sentinel-hf-home-")
os.environ["HF_HOME"] = SENTINEL_HOME  # the runner must replace this with its own per-call directory and write nothing here
sys.path.insert(0, str(HERE.parents[1] / "ml"))
sys.path.insert(0, str(HERE))
import transformers_runner as tb  # noqa: E402
import hf_fixtures  # noqa: E402
import ml_runtime_common as common  # noqa: E402
import numpy as np  # noqa: E402
import transformers_manifest  # noqa: E402

results, details = {}, {}
root = tempfile.mkdtemp(prefix="meticule-hf-fixtures-")
summary = hf_fixtures.build(root)
details["skipped"] = summary["skipped"]
UUID = "00000000-0000-4000-8000-000000000000"


def spec(algorithm, column_type, rows_target=True, hyperparameters=None, metrics=None, cv=None, target_type="string"):
    columns = [{"name": "x", "type": column_type}] + ([{"name": "label", "type": target_type}] if rows_target else [])
    return {"algorithm": algorithm, "backend": "transformers", "seed": 7, "hyperparameters": hyperparameters or {},
            "evaluation": {**({"metrics": metrics} if metrics else {}), **({"cv": cv} if cv else {})},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False},
            "dataset": {"schema": {"columns": columns, **({"target": "label"} if rows_target else {})}},
            "pipeline": {"definition": {"features": ["x"], "steps": []}}}


def run(command, s, rows, model_root=root, **extra):
    return tb.execute({"command": command, "snapshot": s, "rows": rows, "modelRoot": model_root, **extra})


def lifecycle(s, rows, model_root=root):
    prepared = run("prepare", s, rows, model_root)
    trained = run("train", s, rows, model_root, prepared=prepared)
    evaluated = run("evaluate", s, rows, model_root, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])
    return prepared, trained, evaluated


def refusal(fn):
    """The fixed refusal code a call raises, or None if it succeeds."""
    try:
        fn()
    except common.RunnerError as error:
        return error.code
    except common.IncompatibleArtifact:
        return "incompatible_artifact"
    except ValueError:
        return "value_error"
    return None


def metric(evaluated, name, partition="train"):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == name and m["partition"] == partition)


def png(rgb, size=32):
    """A solid-colour RGB PNG built with the standard library."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()


positive_words, negative_words = ["good", "great", "happy", "love", "excellent"], ["bad", "awful", "sad", "hate", "terrible"]
text_rows = [{"x": " ".join([(positive_words if i % 2 else negative_words)[(i + k) % 5] for k in range(3)] + ["the", "movie"]),
              "label": "positive" if i % 2 else "negative"} for i in range(60)]
temp_before = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("meticule-model-")}

# Provisioning refusals: every malicious variant, an unprovisioned name and an unusable root, each with its fixed code.
embedder = lambda model: spec("text_embedder", "text", False, {"model": model})
for name in summary["evil"]:
    results[f"provisioning_refuses_{name.replace('-', '_')}"] = refusal(lambda: run("prepare", embedder(name), text_rows)) == "pretrained_model_mismatch"
results["unprovisioned_model_is_unavailable"] = refusal(lambda: run("prepare", embedder("not-provisioned"), text_rows)) == "pretrained_model_unavailable"
results["relative_model_root_is_unavailable"] = refusal(lambda: run("prepare", embedder("tiny-bert-encoder"), text_rows, model_root="relative/root")) == "pretrained_model_unavailable"
results["path_like_model_names_refused"] = all(refusal(lambda m=m: run("prepare", embedder(m), text_rows)) == "value_error" for m in ("../tiny-bert-encoder", "Tiny", "a/b", ""))
results["missing_model_name_refused"] = refusal(lambda: run("prepare", spec("text_embedder", "text", False, {}), text_rows)) == "value_error"
results["classifier_offered_as_encoder_refused"] = refusal(lambda: run("prepare", spec("text_embedder", "text", False, {"model": "tiny-bert-sentiment"}), text_rows)) == "pretrained_model_mismatch"
results["unknown_hyperparameter_refused"] = refusal(lambda: run("prepare", spec("text_embedder", "text", False, {"model": "tiny-bert-encoder", "trust_remote_code": True}), text_rows)) == "value_error"
results["cuda_refused"] = refusal(lambda: run("prepare", {**embedder("tiny-bert-encoder"), "training": {"device": "cuda", "precision": "fp32"}}, text_rows)) == "cuda_unsupported"

# Pretrained classifiers bind their label set; outputs are model labels.
for family in ("bert", "distilbert"):
    s = spec("text_classifier", "text", True, {"model": f"tiny-{family}-sentiment"}, ["accuracy", "log_loss"])
    prepared, trained, evaluated = lifecycle(s, text_rows)
    predictions = run("predict", s, text_rows[:10], artifact=trained["artifact"])["predictions"]
    document = json.loads(base64.b64decode(trained["artifact"]))
    results[f"{family}_pretrained_classifier_binds_labels"] = document["labels"] == ["negative", "positive"] and set(predictions) <= {"negative", "positive"} and 0 <= metric(evaluated, "log_loss") < 10
mismatched = [{**r, "label": "neutral" if i == 0 else r["label"]} for i, r in enumerate(text_rows)]
s = spec("text_classifier", "text", True, {"model": "tiny-bert-sentiment"})
results["pretrained_label_mismatch_refused"] = refusal(lambda: run("train", s, mismatched, prepared=run("prepare", s, mismatched))) == "pretrained_label_mismatch"
results["pretrained_cross_validation_refused"] = refusal(lambda: lifecycle(spec("text_classifier", "text", True, {"model": "tiny-bert-sentiment"}, cv={"strategy": "kfold", "folds": 3}), text_rows)) == "value_error"

# Embeddings: shape, normalisation, determinism, distinct inputs, pooling, truncation of a 10,000-character row.
s = spec("text_embedder", "text", True, {"model": "tiny-bert-encoder"}, ["retrieval_precision_at_1", "retrieval_mrr"])
prepared, trained, evaluated = lifecycle(s, text_rows)
vectors = np.array(run("predict", s, text_rows[:6], artifact=trained["artifact"])["predictions"])
again = np.array(run("predict", s, text_rows[:6], artifact=trained["artifact"])["predictions"])
results["embedding_dimension_recorded"] = trained["environment"]["embedding_dimension"] == "32" and vectors.shape == (6, 32)
results["embeddings_are_unit_length"] = np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
results["embeddings_are_deterministic"] = np.array_equal(vectors, again)
results["different_inputs_give_different_vectors"] = not np.allclose(vectors[0], vectors[1])
results["labelled_embedding_metrics_bounded"] = 0.0 <= metric(evaluated, "retrieval_precision_at_1") <= 1.0 and 0.0 < metric(evaluated, "retrieval_mrr") <= 1.0
cls = spec("text_embedder", "text", True, {"model": "tiny-bert-encoder", "pooling": "cls"})
cls_vectors = np.array(run("predict", cls, text_rows[:2], artifact=run("train", cls, text_rows, prepared=prepared)["artifact"])["predictions"])
results["cls_and_mean_pooling_differ"] = not np.allclose(cls_vectors[0], vectors[0])
long_row = [{"x": ("good " * 2000)[:10000], "label": "positive"}]
results["ten_thousand_character_row_truncated_and_embedded"] = len(long_row[0]["x"]) == 10000 and len(run("predict", s, long_row, artifact=trained["artifact"])["predictions"][0]) == 32
unlabelled = spec("text_embedder", "text", False, {"model": "tiny-distilbert-encoder"})
_, unlabelled_trained, unlabelled_eval = lifecycle(unlabelled, [{"x": r["x"]} for r in text_rows])
results["unlabelled_embeddings_report_only_the_diagnostic"] = {m["name"] for m in unlabelled_eval["metrics"]} == {"mean_pairwise_cosine"}
bf16 = spec("text_embedder", "text", False, {"model": "tiny-bert-bf16-encoder"})
results["bfloat16_config_runs_in_fp32"] = len(run("predict", bf16, [{"x": "good movie"}], artifact=run("train", bf16, [{"x": r["x"]} for r in text_rows], prepared=run("prepare", bf16, [{"x": r["x"]} for r in text_rows]))["artifact"])["predictions"][0]) == 32
results["artifact_holds_no_base_weights"] = len(base64.b64decode(trained["artifact"])) < 20_000 and json.loads(base64.b64decode(trained["artifact"]))["model"]["digest"] == trained["environment"]["model_digest"]

# Trained heads over frozen encoders learn separable tasks.
head = spec("text_embedding_classifier", "text", True, {"model": "tiny-bert-encoder", "l2": 0.01}, ["accuracy"])
_, head_trained, head_eval = lifecycle(head, text_rows)
results["text_head_learns_separable_task"] = metric(head_eval, "accuracy") >= 0.95
details["text_head_test_accuracy"] = metric(head_eval, "accuracy", "test")
head_cv = spec("text_embedding_classifier", "text", True, {"model": "tiny-distilbert-encoder", "l2": 0.01}, ["accuracy"], cv={"strategy": "kfold", "folds": 3})
_, _, cv_eval = lifecycle(head_cv, text_rows)
results["head_cross_validation_fits_per_fold"] = cv_eval["performance"]["cvModelsBuilt"] == 3.0 and sum(m["partition"] == "cv_fold" for m in cv_eval["metrics"]) == 3
images = [{"x": png((200 + i % 40, 10, 10) if i % 2 else (10, 10, 200 + i % 40)), "label": "red" if i % 2 else "blue"} for i in range(40)]
image_head = spec("image_embedding_classifier", "image", True, {"model": "tiny-vit-encoder", "l2": 0.01}, ["accuracy"])
results["image_head_learns_red_versus_blue"] = metric(lifecycle(image_head, images)[2], "accuracy") >= 0.95
image_pretrained = spec("image_classifier", "image", True, {"model": "tiny-vit-colours"}, ["accuracy"])
_, image_trained, _ = lifecycle(image_pretrained, images)
results["pretrained_image_classifier_outputs_model_labels"] = set(run("predict", image_pretrained, images[:6], artifact=image_trained["artifact"])["predictions"]) <= {"red", "blue"}
image_embed = spec("image_embedder", "image", False, {"model": "tiny-vit-encoder"})
image_embed_artifact = run("train", image_embed, images, prepared=run("prepare", image_embed, images))["artifact"]
image_vectors = np.array(run("predict", image_embed, images[:4], artifact=image_embed_artifact)["predictions"])
results["image_embeddings_unit_length"] = image_vectors.shape == (4, 32) and np.allclose(np.linalg.norm(image_vectors, axis=1), 1.0)

# Images: undecodable and out-of-bounds media refused as invalid_media.
corrupt = png((1, 2, 3))
corrupt = base64.b64encode(base64.b64decode(corrupt)[:60] + b"\x00" * 40).decode()
results["undecodable_image_refused"] = refusal(lambda: run("predict", image_embed, [{"x": corrupt}], artifact=image_embed_artifact)) == "invalid_media"
huge = base64.b64encode(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 5000, 10, 8, 2, 0, 0, 0) + b"\x00" * 4 + struct.pack(">I", 0) + b"IEND" + b"\x00" * 4).decode()
results["oversized_image_refused"] = refusal(lambda: tb.open_image(huge)) == "invalid_media"

# A provisioned model that changes after training is refused at prediction (digest re-verified on every load).
altered = tempfile.mkdtemp(prefix="meticule-hf-altered-")
shutil.copytree(os.path.join(root, "tiny-bert-encoder"), os.path.join(altered, "tiny-bert-encoder"))
weights = os.path.join(altered, "tiny-bert-encoder", "model.safetensors")
blob = bytearray(open(weights, "rb").read()); blob[-1] ^= 0x01
open(weights, "wb").write(bytes(blob))
transformers_manifest.write_manifest(os.path.join(altered, "tiny-bert-encoder"), "text-encoder", "bert", "meticule-fixtures/tiny-bert-encoder", "0" * 40, "apache-2.0")
results["changed_model_digest_refused_at_predict"] = refusal(lambda: run("predict", s, text_rows[:2], model_root=altered, artifact=trained["artifact"])) == "pretrained_model_mismatch"
results["foreign_runtime_artifact_refused"] = refusal(lambda: run("predict", s, text_rows[:2], artifact=trained["artifact"], expectedEnvironment={**trained["environment"], "transformers": "0.0.1"})) == "incompatible_artifact"
results["tampered_artifact_refused"] = refusal(lambda: run("predict", s, text_rows[:2], artifact=base64.b64encode(b"{\"schema\": 1}").decode())) == "incompatible_artifact"

# Audio (M8): a pretrained wav2vec2 classifier binds its labels; a CTC recogniser decodes greedily. The CTC fixture's bias
# forces every clip to "h", so WER and CER have known values. Tiny random models: the path, not recognition quality.
def wav_cell(freq, seconds=0.3, rate=16000):
    t = np.arange(int(rate * seconds)) / rate
    data = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(data))
    return base64.b64encode(header + data).decode()


tones = [{"x": wav_cell(250 + 20 * (i % 5) if i % 2 else 2500 + 50 * (i % 5)), "label": "low" if i % 2 else "high"} for i in range(20)]
audio_cls = spec("audio_classifier", "audio", True, {"model": "tiny-wav2vec2-tones"}, ["accuracy"])
_, audio_trained, _ = lifecycle(audio_cls, tones)
audio_labels = run("predict", audio_cls, tones[:5], artifact=audio_trained["artifact"])["predictions"]
results["audio_classifier_binds_labels_and_outputs_them"] = json.loads(base64.b64decode(audio_trained["artifact"]))["labels"] == ["low", "high"] and set(audio_labels) <= {"low", "high"}
results["audio_classifier_label_mismatch_refused"] = refusal(lambda: run("train", audio_cls, [{**t, "label": "middle"} for t in tones], prepared=run("prepare", audio_cls, tones))) == "pretrained_label_mismatch"
bundle = tb.Bundle(json.loads(base64.b64decode(audio_trained["artifact"])), root, 1)
try:  # zero padding changes a group-norm wav2vec2's outputs, so a clip must give the same scores alone and beside a longer one
    alone = tb.outputs(bundle, [wav_cell(250, 0.2)])["proba"][0]
    results["audio_rows_do_not_depend_on_their_batch"] = np.array_equal(alone, tb.outputs(bundle, [wav_cell(2600, 1.5), wav_cell(250, 0.2)])["proba"][1])
finally:
    bundle.loaded.close()
speech = spec("speech_recognizer", "audio", True, {"model": "tiny-wav2vec2-ctc"}, ["wer", "cer"], target_type="text")
said = [{"x": wav_cell(300 + 40 * i), "label": ["h", "hi", "a b"][i % 3]} for i in range(15)]
_, speech_trained, speech_eval = lifecycle(speech, said)
transcripts = run("predict", speech, said[:4], artifact=speech_trained["artifact"])["predictions"]
results["ctc_decodes_the_forced_transcript"] = transcripts == ["h", "h", "h", "h"]
results["speech_is_deterministic"] = run("predict", speech, said[:4], artifact=speech_trained["artifact"])["predictions"] == transcripts
results["speech_lifecycle_reports_wer_and_cer"] = {m["name"] for m in speech_eval["metrics"]} == {"wer", "cer"} and all(0 <= m["value"] for m in speech_eval["metrics"])
results["undecodable_or_too_short_audio_refused"] = (refusal(lambda: run("predict", speech, [{"x": base64.b64encode(b"RIFF").decode()}], artifact=speech_trained["artifact"])) == "invalid_media"
                                                     and refusal(lambda: run("predict", speech, [{"x": wav_cell(300, seconds=0.01)}], artifact=speech_trained["artifact"])) == "invalid_media")

# No network: every socket is refused during a full train/evaluate/predict cycle, and nothing is written to HF_HOME.
real_socket = socket.socket


class NoNetwork(real_socket):
    def __init__(self, *args, **kwargs):
        raise OSError("network disabled by the check")


socket.socket = NoNetwork
try:
    _, offline_trained, _ = lifecycle(spec("text_embedding_classifier", "text", True, {"model": "tiny-bert-encoder"}, ["accuracy"]), text_rows)
    results["full_cycle_runs_without_network"] = len(run("predict", head, text_rows[:3], artifact=offline_trained["artifact"])["predictions"]) == 3
finally:
    socket.socket = real_socket
results["hf_home_untouched_and_cache_private"] = os.listdir(SENTINEL_HOME) == [] and os.environ["HF_HOME"] != SENTINEL_HOME and os.environ["HF_HUB_OFFLINE"] == "1"
results["per_call_model_copies_removed"] = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("meticule-model-")} <= temp_before

shutil.rmtree(root, ignore_errors=True)
shutil.rmtree(altered, ignore_errors=True)
print(json.dumps({"results": results, "details": details}))
