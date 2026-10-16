"""Real audio backend checks (NumPy/SciPy, the scikit-learn interpreter). Prints one JSON object of named results.

Decoder known answers, every audio vector of the shared media vector file, MFCC determinism and invariances, classifier
learning, output schemas and artifact refusals. Processor integration is in tests/integration/ml-audio.test.ts.
"""
import base64
import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import audio_runner as ab  # noqa: E402
import audio_common  # noqa: E402
import media_common  # noqa: E402
import ml_runtime_common as common  # noqa: E402

results, details = {}, {}
UUID = "00000000-0000-4000-8000-000000000000"


def wav(samples, rate, bits=16, fmt=1):
    """WAV bytes (base64) from float samples in [-1, 1], shaped (frames,) or (frames, channels)."""
    data = np.asarray(samples, dtype=np.float64)
    data = data[:, None] if data.ndim == 1 else data
    channels = data.shape[1]
    if fmt == 3:
        body = data.astype("<f4").tobytes()
    elif bits == 8:
        body = np.clip(np.round(data * 128 + 128), 0, 255).astype(np.uint8).tobytes()
    elif bits == 24:
        ints = np.clip(np.round(data * 8388608), -8388608, 8388607).astype(np.int32).reshape(-1)
        body = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in ints)
    else:
        scale = float(1 << (bits - 1))
        body = np.clip(np.round(data * scale), -scale, scale - 1).astype("<i2" if bits == 16 else "<i4").tobytes()
    align = channels * bits // 8
    fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, rate * align, align, bits)
    riff = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt_chunk + b"data" + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")
    return base64.b64encode(b"RIFF" + struct.pack("<I", len(riff)) + riff).decode()


def tone(freq, rate, seconds=0.5, amplitude=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def refuses(fn, kind=common.RunnerError, code=None):
    try:
        fn()
    except kind as error:
        return code is None or getattr(error, "code", None) == code
    except Exception:
        return False
    return False


# ---- decoder known answers -------------------------------------------------------------------------------------------
sine = tone(440, 8000)
decoded = audio_common.decode(wav(sine, 8000), 8000)
results["pcm16_round_trip_within_quantisation"] = decoded.shape == sine.shape and np.max(np.abs(decoded - sine)) <= 1.0 / 32768 + 1e-7
results["pcm8_24_32_and_float32_decode_the_same_tone"] = all(
    np.max(np.abs(audio_common.decode(wav(sine, 8000, bits=b, fmt=f), 8000) - sine)) <= tolerance
    for b, f, tolerance in ((8, 1, 1.0 / 128 + 1e-6), (24, 1, 1.0 / 8388608 + 1e-7), (32, 1, 1e-6), (32, 3, 1e-7)))
stereo = np.column_stack([sine, -0.5 * sine])
results["stereo_downmix_is_the_channel_mean"] = np.allclose(audio_common.decode(wav(stereo, 8000), 8000), 0.25 * sine, atol=2.0 / 32768)
resampled = audio_common.decode(wav(tone(1000, 44100, seconds=1.0), 44100), 16000)
spectrum = np.abs(np.fft.rfft(resampled))
peak = np.fft.rfftfreq(len(resampled), 1 / 16000)[spectrum.argmax()]
results["resampling_preserves_the_tone_frequency"] = len(resampled) == 16000 and abs(peak - 1000) <= 1.0
details["resampled_peak_hz"] = float(peak)

# ---- the shared media vector file: every audio vector gives the recorded verdict and facts --------------------------
vectors = [v for v in json.loads((ROOT / "tests/fixtures/media-vectors.json").read_text())["vectors"] if v["kind"] == "audio"]
expand = lambda value: value if isinstance(value, str) else "".join(text * count for text, count in value["parts"])
agreeing = 0
for vector in vectors:
    try:
        samples, rate = media_common.read_wav(media_common.decode_base64(expand(vector["input"])))
        agreeing += vector["verdict"] == "accept" and {"sampleRate": rate, "channels": samples.shape[1], "frames": samples.shape[0]} == vector["facts"]
    except common.RunnerError as error:
        agreeing += vector["verdict"] == "refuse" and error.code == "invalid_media"
results["every_shared_audio_vector_agrees"] = agreeing == len(vectors) and len(vectors) >= 40
details["audio_vectors"] = len(vectors)

# ---- MFCC determinism and invariances --------------------------------------------------------------------------------
settings = audio_common.resolve_settings({})
clip = wav(tone(300, 16000) + 0.1 * tone(2500, 16000), 16000)
first, second = audio_common.summary(clip, settings), audio_common.summary(clip, settings)
results["mfcc_summary_is_deterministic_and_sized"] = np.array_equal(first, second) and first.shape == (2 * settings["nMfcc"],)
# Scaling the signal by a multiplies every mel energy by a^2, shifting every log-mel band by 2 ln a. Under the orthonormal
# DCT-II a constant shift lands only on coefficient 0, as 2 ln a * sqrt(n_mels); every other mean and every std is unchanged.
# This holds while no band sits at the log floor, so the signal is broadband (noise plus a tone) and stored as float32.
a = 0.25
broadband = 0.2 * np.random.RandomState(1).randn(8000) + tone(300, 16000)
first = audio_common.summary(wav(broadband, 16000, bits=32, fmt=3), settings)
scaled = audio_common.summary(wav(a * broadband, 16000, bits=32, fmt=3), settings)
n = settings["nMfcc"]
shift = scaled[0] - first[0]
results["amplitude_scaling_moves_only_the_first_mean"] = (abs(shift - 2 * math.log(a) * math.sqrt(settings["nMels"])) < 1e-2
                                                         and np.max(np.abs(scaled[1:n] - first[1:n])) < 1e-2 and np.max(np.abs(scaled[n:] - first[n:])) < 1e-2)
details["first_mean_shift"] = float(shift)
results["different_tones_differ"] = not np.allclose(audio_common.summary(wav(tone(300, 16000), 16000), settings), audio_common.summary(wav(tone(3000, 16000), 16000), settings))
results["inconsistent_settings_refused"] = refuses(lambda: audio_common.resolve_settings({"frameMs": 10, "hopMs": 20}), ValueError) and refuses(lambda: audio_common.resolve_settings({"nMfcc": 40, "nMels": 20}), ValueError)


# ---- the backend: classifier learning, embeddings, output schemas, refusals ------------------------------------------
def spec(algorithm, labelled=True, hyperparameters=None, metrics=None, cv=None, column_type="audio"):
    columns = [{"name": "clip", "type": column_type}] + ([{"name": "label", "type": "string"}] if labelled else [])
    return {"algorithm": algorithm, "backend": "audio", "seed": 7, "hyperparameters": hyperparameters or {},
            "evaluation": {**({"metrics": metrics} if metrics else {}), **({"cv": cv} if cv else {})},
            "split": {"id": UUID, "strategy": "random", "train": 0.6, "validation": 0.2, "test": 0.2, "stratify": False},
            "dataset": {"schema": {"columns": columns, **({"target": "label"} if labelled else {})}}, "pipeline": {"definition": {"features": ["clip"], "steps": []}}}


def run(command, s, rows, **extra):
    return ab.execute({"command": command, "snapshot": s, "rows": rows, **extra})


def lifecycle(s, rows):
    prepared = run("prepare", s, rows)
    trained = run("train", s, rows, prepared=prepared)
    return prepared, trained, run("evaluate", s, rows, prepared=prepared, artifact=trained["artifact"], expectedEnvironment=trained["environment"])


def accuracy(evaluated, partition):
    return next(m["value"] for m in evaluated["metrics"] if m["name"] == "accuracy" and m["partition"] == partition)


rng = np.random.RandomState(0)
tones = [{"clip": wav(tone(200 + 15 * (i % 7), 16000, 0.3) if i % 2 else tone(2500 + 40 * (i % 7), 16000, 0.3), 16000), "label": "low" if i % 2 else "high"} for i in range(40)]
classifier = spec("audio_mfcc_classifier", hyperparameters={"l2": 0.1}, metrics=["accuracy", "log_loss"])
_, trained, evaluated = lifecycle(classifier, tones)
results["classifier_separates_low_and_high_tones"] = accuracy(evaluated, "train") >= 0.95 and accuracy(evaluated, "test") >= 0.9
noise = [{"clip": wav(0.3 * rng.randn(4800) if i % 2 else tone(600 + 20 * (i % 5), 16000, 0.3), 16000), "label": "noise" if i % 2 else "tone"} for i in range(40)]
results["classifier_separates_noise_and_tone"] = accuracy(lifecycle(spec("audio_mfcc_classifier", hyperparameters={"l2": 0.1}, metrics=["accuracy"]), noise)[2], "test") >= 0.9
predictions = run("predict", classifier, tones[:6], artifact=trained["artifact"])["predictions"]
results["classifier_predicts_training_labels"] = predictions == [t["label"] for t in tones[:6]]
cv = spec("audio_mfcc_classifier", hyperparameters={"l2": 0.1}, metrics=["accuracy"], cv={"strategy": "kfold", "folds": 3})
_, _, cv_eval = lifecycle(cv, tones)
results["classifier_cv_fits_a_fresh_head_per_fold"] = cv_eval["performance"]["cvModelsBuilt"] == 3.0 and sum(m["partition"] == "cv_fold" for m in cv_eval["metrics"]) == 3
embedder = spec("audio_features", labelled=False, hyperparameters={"nMfcc": 20})
_, embedded, embed_eval = lifecycle(embedder, [{"clip": t["clip"]} for t in tones])
vectors = np.array(run("predict", embedder, [{"clip": t["clip"]} for t in tones[:3]], artifact=embedded["artifact"])["predictions"])
results["embeddings_have_the_recorded_dimension"] = embedded["environment"]["embedding_dimension"] == "40" and vectors.shape == (3, 40)
results["unlabelled_embeddings_report_only_the_diagnostic"] = {m["name"] for m in embed_eval["metrics"]} == {"mean_pairwise_cosine"}
labelled_embed = spec("audio_features", metrics=["retrieval_precision_at_1"])
results["labelled_embeddings_retrieve_the_same_class"] = next(m["value"] for m in lifecycle(labelled_embed, tones)[2]["metrics"] if m["partition"] == "train") >= 0.9

# Artifact and request refusals.
results["corrupt_artifact_refused"] = refuses(lambda: run("predict", classifier, tones[:1], artifact=base64.b64encode(b"junk").decode()), common.IncompatibleArtifact)
results["foreign_runtime_refused"] = refuses(lambda: run("predict", classifier, tones[:1], artifact=trained["artifact"], expectedEnvironment={**trained["environment"], "scipy": "0.0.1"}), common.IncompatibleArtifact)
buffer = io.BytesIO()
np.savez(buffer, metadata=np.array(json.dumps({"schema": 1})), evil=np.array([{"x": 1}], dtype=object))
results["object_array_artifact_refused"] = refuses(lambda: run("predict", classifier, tones[:1], artifact=base64.b64encode(buffer.getvalue()).decode()), common.IncompatibleArtifact)
results["undecodable_clip_refused"] = refuses(lambda: run("predict", classifier, [{"clip": base64.b64encode(b"RIFF....WAVE").decode()}], artifact=trained["artifact"]), common.RunnerError, "invalid_media")
results["non_audio_feature_refused"] = refuses(lambda: run("prepare", spec("audio_features", labelled=False, column_type="text"), [{"clip": "x"}]), ValueError)
results["cuda_refused"] = refuses(lambda: run("prepare", {**classifier, "training": {"device": "cuda", "precision": "fp32"}}, tones), common.RunnerError, "cuda_unsupported")

print(json.dumps({"results": {k: bool(v) for k, v in results.items()}, "details": details}))
