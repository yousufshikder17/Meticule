"""The shared media and timestamp vectors through media_common and parse_timestamp, plus the WAV decoder's sample values.
Prints one JSON line: {"results": {name: bool}, "vectors": {id: bool}, "mismatches": [...]}.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import media_common as media  # noqa: E402
import ml_runtime_common as common  # noqa: E402

DOCUMENT = json.loads((ROOT / "tests" / "fixtures" / "media-vectors.json").read_text(encoding="utf-8"))
VECTORS = DOCUMENT["vectors"]
# The review's attack cases; each must stay in the file as a refusal.
ATTACKS = ["png-second-ihdr", "jpeg-lying-second-sof", "wav-second-data-chunk", "wav-lying-block-align", "wav-float-nan-sample",
           "wav-block-align-zero", "wav-10ms-clip", "b64-url-safe", "b64-whitespace", "b64-mid-string-padding", "ts-feb-30",
           "ts-hour-24", "ts-year-0000", "ts-offset-99-99"]
results, verdicts, mismatches, refusals = {}, {}, [], []


def expand(value):
    return value if isinstance(value, str) else "".join(text * count for text, count in value["parts"])


def facts_of(kind, text):
    if kind == "base64":
        data = media.decode_base64(text)
        return {"length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if kind == "image":
        image_format, width, height = media.image_header(media.decode_base64(text))
        return {"format": image_format, "width": width, "height": height}
    if kind == "audio":
        samples, rate = media.read_wav(media.decode_base64(text))
        return {"sampleRate": rate, "channels": samples.shape[1], "frames": samples.shape[0]}
    return {"epochMs": common.parse_timestamp(text) * 1000}


def agrees(vector, text):
    try:
        got = facts_of(vector["kind"], text)
    except common.RunnerError as error:
        refusals.append((vector, text, error))
        return vector["verdict"] == "refuse"
    except Exception as error:  # anything but a classified refusal is a defect, reported by vector id
        mismatches.append(f"{vector['id']}: {type(error).__name__}")
        return False
    if vector["verdict"] != "accept":
        return False
    if vector["kind"] == "timestamp":
        return abs(got["epochMs"] - vector["facts"]["epochMs"]) <= 0.002
    return got == vector["facts"]


for vector in VECTORS:
    verdicts[vector["id"]] = agrees(vector, expand(vector["input"]))
    if not verdicts[vector["id"]]:
        mismatches.append(vector["id"])

ids = [v["id"] for v in VECTORS]
results["vector_file_well_formed"] = len(set(ids)) == len(ids) and all(
    v["kind"] in ("base64", "image", "audio", "timestamp") and v["verdict"] in ("accept", "refuse") and v.get("note")
    and (("facts" in v) == (v["verdict"] == "accept")) for v in VECTORS)
limits = DOCUMENT["limits"]
results["limits_match_media_common"] = (limits["mediaMaxBytes"], limits["maxBase64Chars"], limits["maxSide"], limits["maxPixels"], limits["maxChunks"]) == \
    (media.MEDIA_MAX_BYTES, media.MAX_BASE64_CHARS, media.MAX_SIDE, media.MAX_PIXELS, media.MAX_CHUNKS) == (2097152, 2796204, 4096, 16777216, 10000)
by_id = {v["id"]: v for v in VECTORS}
results["every_attack_case_present"] = all(name in by_id and by_id[name]["verdict"] == "refuse" for name in ATTACKS)
for kind in ("base64", "image", "audio", "timestamp"):
    chosen = [v for v in VECTORS if v["kind"] == kind]
    results[f"{kind}_vectors_agree"] = len(chosen) >= 10 and {"accept", "refuse"} <= {v["verdict"] for v in chosen} and all(verdicts[v["id"]] for v in chosen)
results["media_refusals_are_invalid_media"] = all(e.code == "invalid_media" for v, _, e in refusals if v["kind"] != "timestamp")
results["timestamp_refusals_are_invalid_time_series"] = all(e.code == "invalid_time_series" for v, _, e in refusals if v["kind"] == "timestamp")
# Fixed texts: a handful of distinct messages, none carrying the refused cell.
results["refusal_messages_are_fixed"] = len({e.message for _, _, e in refusals}) <= 12 and all(
    len(text) < 4 or text not in e.message for _, text, e in refusals)


def wav(tag, channels, bits, payload, rate=8000):
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def decoded(tag, channels, bits, frame, frames=800):
    samples, rate = media.read_wav(wav(tag, channels, bits, frame * frames))
    return samples, rate


# Known sample values per width: the decoder scales PCM to [-1, 1) and keeps float samples as written.
cases = [(1, 8, bytes((0, 128, 255, 192)), [-1.0, 0.0, 127 / 128, 0.5]),
         (1, 16, struct.pack("<4h", -32768, 0, 32767, 16384), [-1.0, 0.0, 32767 / 32768, 0.5]),
         (1, 24, b"\x00\x00\x80" + b"\x00\x00\x00" + b"\xff\xff\x7f" + b"\x00\x00\xc0", [-1.0, 0.0, 8388607 / 8388608, -0.5]),
         (1, 32, struct.pack("<4i", -2147483648, 0, 2147483647, 1073741824), [-1.0, 0.0, 2147483647 / 2147483648, 0.5]),
         (3, 32, struct.pack("<4f", -1.5, 0.0, 0.25, 2.0), [-1.5, 0.0, 0.25, 2.0])]
scaled = True
for tag, bits, payload, expected in cases:
    samples, rate = decoded(tag, 1, bits, payload, 200)
    scaled = scaled and samples.dtype == np.float32 and samples.shape == (800, 1) and rate == 8000 \
        and np.allclose(samples[:4, 0], np.array(expected, dtype=np.float32), rtol=0, atol=1e-7)
results["wav_decoder_scales_each_width"] = bool(scaled)
stereo, _ = decoded(1, 2, 16, struct.pack("<2h", 16384, -16384))
results["wav_decoder_returns_frames_by_channels"] = stereo.shape == (800, 2) and bool(np.all(stereo[:, 0] == 0.5) and np.all(stereo[:, 1] == -0.5))
extensible = media.read_wav(media.decode_base64(expand(by_id["wav-extensible-pcm16-stereo"]["input"])))[0]
results["wav_decoder_accepts_extensible"] = extensible.shape == (800, 2) and extensible.dtype == np.float32

# Timestamps: parse_timestamp is UTC epoch seconds; epoch_micros is exact where a float of seconds is not.
results["parse_timestamp_is_utc_epoch_seconds"] = (common.parse_timestamp("1970-01-01T00:00:01.5Z") == 1.5
                                                   and common.parse_timestamp("1970-01-01T05:30:00+05:30") == 0.0
                                                   and common.parse_timestamp("1969-12-31T19:00:00-05:00") == 0.0)
results["epoch_micros_is_exact"] = (common.epoch_micros("2199-12-31T23:59:59.999999Z") - common.epoch_micros("2199-12-31T23:59:59.999998Z") == 1
                                    and common.epoch_micros("2024-01-01T00:00:00.000001Z") == 1704067200000001)

print(json.dumps({"results": results, "vectors": verdicts, "mismatches": mismatches}))
