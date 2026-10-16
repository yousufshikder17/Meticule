"""Writes tests/fixtures/media-vectors.json: the media and timestamp cases that src/ml/media.ts and every Python interpreter
that validates media (media_common.py, ml_runtime_common.parse_timestamp) must decide identically. Stdlib only and
deterministic: edit a case here and rerun (python tests/python/make_media_vectors.py); never edit the JSON by hand.
Accepted images are real files (PNG via zlib, baseline JPEG of flat 8x8 blocks), so pixel decoders can reuse them.
"""
import base64
import calendar
import hashlib
import json
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "media-vectors.json"
MEDIA_MAX_BYTES = 2 * 1024 * 1024
MAX_BASE64_CHARS = 4 * -(-MEDIA_MAX_BYTES // 3)
MAX_CHUNKS = 10_000
vectors = []


def b64(data):
    return base64.b64encode(data).decode()


def compact(text):
    """The cell as `input`: the text, or {"parts": [[text, count], ...]} when runs of a repeated unit (whole 4-character
    base64 groups, up to 8) shorten it. Expanding is "".join(text * count for text, count in parts)."""
    if len(text) < 512:
        return text
    groups = [text[i:i + 4] for i in range(0, len(text), 4)]
    parts, literal, i = [], [], 0
    while i < len(groups):
        size, repeats = 1, 1
        for width in range(1, 9):
            unit, count = groups[i:i + width], 1
            while len(unit) == width and groups[i + count * width:i + (count + 1) * width] == unit:
                count += 1
            if count >= 4 and count * width > size * repeats:
                size, repeats = width, count
        if repeats > 1:
            if literal:
                parts.append(["".join(literal), 1])
                literal = []
            parts.append(["".join(groups[i:i + size]), repeats])
            i += size * repeats
        else:
            literal.append(groups[i])
            i += 1
    if literal:
        parts.append(["".join(literal), 1])
    assert "".join(t * n for t, n in parts) == text
    return {"parts": parts} if len(json.dumps(parts)) < len(text) else text


def case(id, kind, text, verdict, note, **facts):
    vector = {"id": id, "kind": kind, "input": compact(text), "verdict": verdict, "note": note}
    if verdict == "accept":
        vector["facts"] = facts
    vectors.append(vector)


# ---- base64 (C1) ---------------------------------------------------------------------------------------------------

def base64_case(id, text, verdict, note):
    facts = {}
    if verdict == "accept":
        data = base64.b64decode(text, validate=True)
        facts = {"length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    case(id, "base64", text, verdict, note, **facts)


base64_case("b64-empty", "", "accept", "zero groups match the grammar")
base64_case("b64-three-bytes", "QUJD", "accept", "one full group")
base64_case("b64-one-byte-padded", "QQ==", "accept", "final group with two pads")
base64_case("b64-two-bytes-padded", "QUI=", "accept", "final group with one pad")
base64_case("b64-plus-slash-alphabet", "+/+/", "accept", "standard alphabet includes + and /")
base64_case("b64-non-canonical-bits", "QR==", "accept", "unused bits after the last byte are ignored, as the grammar allows")
base64_case("b64-url-safe", "-_-_", "refuse", "URL-safe alphabet")
base64_case("b64-whitespace", "QUJD QUJD", "refuse", "inner space")
base64_case("b64-line-break", "QUJD\nQUJD", "refuse", "MIME line break")
base64_case("b64-trailing-newline", "QUJD\n", "refuse", "a trailing newline is not part of the grammar")
base64_case("b64-mid-string-padding", "QQ==QUJD", "refuse", "'=' before the end")
base64_case("b64-padding-only", "====", "refuse", "a group of pads")
base64_case("b64-three-pads", "Q===", "refuse", "at most two pads")
base64_case("b64-missing-padding", "QQ", "refuse", "length not a multiple of 4")
base64_case("b64-length-five", "QUJDQ", "refuse", "length not a multiple of 4")
base64_case("b64-data-uri", "data:image/png;base64,QUJD", "refuse", "data: prefix")
base64_case("b64-non-ascii", "QUJé", "refuse", "non-ASCII character")
case("b64-max-decoded-2mib", "base64", "QUFB" * (MAX_BASE64_CHARS // 4 - 1) + "QUE=", "accept",
     "2,796,204 characters decoding to exactly 2 MiB", length=MEDIA_MAX_BYTES,
     sha256=hashlib.sha256(b"A" * MEDIA_MAX_BYTES).hexdigest())
case("b64-decoded-over-2mib", "base64", "QUFB" * (MAX_BASE64_CHARS // 4), "refuse",
     "2,796,204 characters without padding decode to 2 MiB + 1 byte")
case("b64-too-long", "base64", "QUFB" * (MAX_BASE64_CHARS // 4 + 1), "refuse", "longer than 2,796,204 characters")


# ---- PNG (C2) ------------------------------------------------------------------------------------------------------

def chunk(kind, payload=b""):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def ihdr(width, height, color=0):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))


def idat(width, height, color=0):
    row = b"\x00" + bytes(width * (3 if color == 2 else 1))  # filter 0, black pixels
    return chunk(b"IDAT", zlib.compress(row * height, 9))


def png(width, height, color=0, before=b"", after=b"", tail=b"", header=None):
    return (b"\x89PNG\r\n\x1a\n" + (header if header is not None else ihdr(width, height, color)) + before
            + idat(width, height, color) + after + chunk(b"IEND") + tail)


def image_case(id, data, verdict, note, size=None):
    kind = "png" if data[:4] == b"\x89PNG" else "jpeg"
    facts = {"format": kind, "width": size[0], "height": size[1]} if verdict == "accept" else {}
    case(id, "image", b64(data), verdict, note, **facts)


TEXT = chunk(b"tEXt", b"Comment\x00vector")
EMPTY = chunk(b"zzZz")  # unknown ancillary chunk: decoders skip it
image_case("png-gray-1x1", png(1, 1), "accept", "smallest image", (1, 1))
image_case("png-rgb-3x2", png(3, 2, color=2), "accept", "RGB", (3, 2))
image_case("png-max-width", png(4096, 1), "accept", "4096 wide", (4096, 1))
image_case("png-max-height", png(1, 4096), "accept", "4096 high", (1, 4096))
image_case("png-max-pixels", png(4096, 4096), "accept", "16,777,216 pixels", (4096, 4096))
image_case("png-ancillary-chunks", png(2, 2, before=TEXT, after=TEXT), "accept", "tEXt chunks before and after IDAT are skipped", (2, 2))
image_case("png-bytes-after-iend", png(2, 2, tail=b"trailing"), "accept", "the walk stops at IEND; later bytes are never read", (2, 2))
image_case("png-second-ihdr", png(2, 2, before=ihdr(4096, 4096)), "refuse", "a second IHDR could restate the size")
image_case("png-second-ihdr-after-idat", png(2, 2, after=ihdr(2, 2)), "refuse", "any later IHDR is refused")
image_case("png-width-zero", png(0, 1), "refuse", "width below 1")
image_case("png-height-zero", png(1, 0), "refuse", "height below 1")
image_case("png-width-4097", png(4097, 1), "refuse", "width above 4096")
image_case("png-height-4097", png(1, 4097), "refuse", "height above 4096")
image_case("png-first-chunk-not-ihdr", png(2, 2, header=TEXT + ihdr(2, 2)), "refuse", "IHDR must be the chunk at offset 8")
image_case("png-ihdr-length-12", png(2, 2, header=chunk(b"IHDR", struct.pack(">IIBBBB", 2, 2, 8, 0, 0, 0))), "refuse", "IHDR length must be 13")
full = png(2, 2)
image_case("png-chunk-beyond-buffer", full[:33] + struct.pack(">I", 1 << 20) + full[37:], "refuse", "IDAT length larger than the file")
image_case("png-length-field-max", full[:33] + b"\xff\xff\xff\xff" + full[37:], "refuse", "length 0xFFFFFFFF")
image_case("png-missing-iend", full[:-12], "refuse", "no IEND before the end")
image_case("png-truncated-chunk-header", full[:-6], "refuse", "the file ends inside a chunk header")
image_case("png-signature-and-ihdr-only", full[:33], "refuse", "no chunk after IHDR")
image_case("png-chunk-limit", png(2, 2, before=EMPTY * (MAX_CHUNKS - 3)), "accept", "exactly 10,000 chunks (IHDR, IDAT and IEND included)", (2, 2))
image_case("png-chunk-limit-exceeded", png(2, 2, before=EMPTY * (MAX_CHUNKS - 2)), "refuse", "10,001 chunks")
case("image-bad-signature", "image", b64(b"\x89PNX\r\n\x1a\n" + full[8:]), "refuse", "neither PNG nor JPEG magic")
case("image-gif", "image", b64(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"), "refuse", "GIF is not accepted")
case("image-empty", "image", "", "refuse", "valid base64 of nothing")
case("image-data-uri", "image", "data:image/png;base64," + b64(png(1, 1)), "refuse", "data: prefix fails the base64 grammar")
case("image-over-2mib", "image", b64(png(1, 1)[:-12] + chunk(b"zzZz", bytes(MEDIA_MAX_BYTES)) + chunk(b"IEND")), "refuse", "larger than 2 MiB")


# ---- JPEG (C2) -----------------------------------------------------------------------------------------------------

def segment(marker, payload=b""):
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def sof(width, height, marker=0xC0, components=1):
    return segment(marker, struct.pack(">BHHB", 8, height, width, components) + b"\x01\x11\x00" * components)


DQT = segment(0xDB, b"\x00" + b"\x01" * 64)
# One code ("0") per table: DC category 0 and AC end-of-block, so every block of the scan is the two bits "00".
DHT = segment(0xC4, b"\x00" + b"\x01" + bytes(15) + b"\x00" + b"\x10" + b"\x01" + bytes(15) + b"\x00")
SOS = segment(0xDA, b"\x01\x01\x00\x00\x3f\x00")
COM = segment(0xFE)


def scan(width, height):
    bits = 2 * -(-width // 8) * -(-height // 8)
    return bytes(bits // 8) + (bytes((0xFF >> (bits % 8),)) if bits % 8 else b"")  # pad with 1-bits


def jpeg(width, height, frame=None, before=b"", between=b"", sos=SOS):
    return (b"\xff\xd8" + before + DQT + (frame if frame is not None else sof(width, height)) + between + DHT + sos
            + scan(width, height) + b"\xff\xd9")


APP0 = segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
image_case("jpeg-baseline-8x8", jpeg(8, 8), "accept", "SOF0, one component", (8, 8))
image_case("jpeg-extended-16x24", jpeg(16, 24, frame=sof(16, 24, 0xC1)), "accept", "SOF1 carries the size too", (16, 24))
image_case("jpeg-max-width", jpeg(4096, 8), "accept", "4096 wide", (4096, 8))
image_case("jpeg-max-height", jpeg(8, 4096), "accept", "4096 high", (8, 4096))
image_case("jpeg-app0-and-comments", jpeg(8, 8, before=APP0 + segment(0xFE, b"note")), "accept", "APPn and COM segments are skipped", (8, 8))
image_case("jpeg-fill-bytes", jpeg(8, 8, between=b"\xff\xff"), "accept", "0xFF fill bytes may precede a marker", (8, 8))
image_case("jpeg-lying-second-sof", jpeg(8, 8, between=sof(4096, 4096)), "refuse", "a second SOFn could restate the size a decoder uses")
image_case("jpeg-second-sof-same-size", jpeg(8, 8, between=sof(8, 8, 0xC2)), "refuse", "exactly one SOFn before SOS")
image_case("jpeg-sof-beyond-buffer", b"\xff\xd8" + DQT + b"\xff\xc0\x00\x40" + struct.pack(">BHH", 8, 8, 8), "refuse", "SOF length larger than the file")
image_case("jpeg-sof-length-mismatch", jpeg(8, 8, frame=segment(0xC0, struct.pack(">BHHB", 8, 8, 8, 1) + b"\x01\x11\x00" + bytes(6))), "refuse", "SOF length must be 8 + 3 * components")
image_case("jpeg-sof-zero-components", jpeg(8, 8, frame=segment(0xC0, struct.pack(">BHHB", 8, 8, 8, 0))), "refuse", "at least one component")
image_case("jpeg-sof-five-components", jpeg(8, 8, frame=sof(8, 8, components=5)), "refuse", "at most four components")
image_case("jpeg-height-zero", jpeg(8, 8, frame=sof(8, 0)), "refuse", "height 0 (size deferred to a DNL marker)")
image_case("jpeg-width-4097", jpeg(4097, 8), "refuse", "width above 4096")
image_case("jpeg-no-sos", b"\xff\xd8" + DQT + sof(8, 8) + DHT + b"\xff\xd9", "refuse", "EOI before any SOS")
image_case("jpeg-sos-before-sof", b"\xff\xd8" + DQT + DHT + SOS + scan(8, 8) + b"\xff\xd9", "refuse", "SOS without a preceding SOFn")
image_case("jpeg-junk-between-segments", jpeg(8, 8, between=b"\x00"), "refuse", "a non-0xFF byte where a marker must start")
image_case("jpeg-segment-length-one", jpeg(8, 8, before=b"\xff\xfe\x00\x01"), "refuse", "segment length below 2")
image_case("jpeg-restart-before-sos", jpeg(8, 8, between=b"\xff\xd0"), "refuse", "standalone markers cannot precede SOS")
image_case("jpeg-soi-only", b"\xff\xd8\xff", "refuse", "magic bytes and nothing else")
image_case("jpeg-segment-limit", jpeg(8, 8, before=COM * (MAX_CHUNKS - 4)), "accept", "exactly 10,000 segments (DQT, SOF, DHT and SOS included)", (8, 8))
image_case("jpeg-segment-limit-exceeded", jpeg(8, 8, before=COM * (MAX_CHUNKS - 3)), "refuse", "10,001 segments")


# ---- WAV (C3) ------------------------------------------------------------------------------------------------------

def fmt(tag=1, channels=1, rate=8000, bits=16, block_align=None, byte_rate=None, extra=b""):
    block_align = channels * bits // 8 if block_align is None else block_align
    byte_rate = rate * block_align if byte_rate is None else byte_rate
    return struct.pack("<HHIIHH", tag, channels, rate, byte_rate, block_align, bits) + extra


def extensible(sub=1, channels=1, rate=8000, bits=16, cb_size=22, valid_bits=None, guid_tail=bytes.fromhex("000000001000800000aa00389b71")):
    extra = struct.pack("<HHI", cb_size, bits if valid_bits is None else valid_bits, 0) + struct.pack("<H", sub) + guid_tail
    return fmt(0xFFFE, channels, rate, bits, extra=extra)


def riff_chunk(kind, payload, pad=True):
    return kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if pad and len(payload) % 2 else b"")


def wav(fmt_payload, data, before=b"", after=b"", pad=True, riff_size=None, chunks=None):
    body = chunks if chunks is not None else before + riff_chunk(b"fmt ", fmt_payload) + riff_chunk(b"data", data, pad) + after
    return b"RIFF" + struct.pack("<I", len(body) + 4 if riff_size is None else riff_size) + b"WAVE" + body


def audio_case(id, data, verdict, note, rate=None, channels=None, frames=None):
    facts = {"sampleRate": rate, "channels": channels, "frames": frames} if verdict == "accept" else {}
    case(id, "audio", b64(data), verdict, note, **facts)


def silence(frames, block_align, unsigned=False):
    return (b"\x80" if unsigned else b"\x00") * (frames * block_align)


LIST = riff_chunk(b"LIST", b"INFOISFT\x05\x00\x00\x00test\x00\x00")
JUNK = riff_chunk(b"JUNK", b"")
NAN, INF = struct.pack("<f", float("nan")), struct.pack("<f", float("inf"))
audio_case("wav-pcm16-mono-100ms", wav(fmt(), silence(800, 2)), "accept", "exactly 0.1 s, the shortest clip", 8000, 1, 800)
audio_case("wav-pcm8-odd-with-pad", wav(fmt(bits=8), silence(801, 1, True)), "accept", "odd data chunk followed by its pad byte", 8000, 1, 801)
audio_case("wav-pcm8-odd-pad-absent-at-end", wav(fmt(bits=8), silence(801, 1, True), pad=False), "accept", "the pad byte may be missing only at the very end", 8000, 1, 801)
audio_case("wav-pcm24-stereo-16k", wav(fmt(channels=2, rate=16000, bits=24), silence(4000, 6)), "accept", "24-bit stereo", 16000, 2, 4000)
audio_case("wav-pcm32-mono-48k", wav(fmt(rate=48000, bits=32), silence(4800, 4)), "accept", "32-bit PCM at the highest rate", 48000, 1, 4800)
audio_case("wav-float32-stereo-44k1", wav(fmt(tag=3, channels=2, rate=44100, bits=32), silence(4410, 8)), "accept", "IEEE float", 44100, 2, 4410)
audio_case("wav-fmt-size-18", wav(fmt(extra=b"\x00\x00"), silence(800, 2)), "accept", "fmt with a zero cbSize", 8000, 1, 800)
audio_case("wav-extensible-pcm16-stereo", wav(extensible(channels=2), silence(800, 4)), "accept", "WAVE_FORMAT_EXTENSIBLE wrapping PCM", 8000, 2, 800)
audio_case("wav-extensible-float32", wav(extensible(sub=3, bits=32), silence(800, 4)), "accept", "WAVE_FORMAT_EXTENSIBLE wrapping IEEE float", 8000, 1, 800)
audio_case("wav-list-chunks-skipped", wav(fmt(), silence(800, 2), before=LIST, after=LIST), "accept", "LIST before fmt and after data", 8000, 1, 800)
audio_case("wav-30s", wav(fmt(bits=8), silence(240000, 1, True)), "accept", "exactly 30 s, the longest clip", 8000, 1, 240000)
audio_case("wav-chunk-limit", wav(fmt(bits=8), silence(800, 1, True), before=JUNK * (MAX_CHUNKS - 2)), "accept", "exactly 10,000 chunks (fmt and data included)", 8000, 1, 800)
audio_case("wav-chunk-limit-exceeded", wav(fmt(bits=8), silence(800, 1, True), before=JUNK * (MAX_CHUNKS - 1)), "refuse", "10,001 chunks")
audio_case("wav-second-data-chunk", wav(fmt(), silence(800, 2), after=riff_chunk(b"data", silence(800, 2))), "refuse", "exactly one data chunk")
audio_case("wav-second-fmt-chunk", wav(fmt(), silence(800, 2), before=riff_chunk(b"fmt ", fmt())), "refuse", "exactly one fmt chunk")
audio_case("wav-data-before-fmt", wav(None, None, chunks=riff_chunk(b"data", silence(800, 2)) + riff_chunk(b"fmt ", fmt())), "refuse", "fmt must precede data")
audio_case("wav-missing-data", wav(None, None, chunks=riff_chunk(b"fmt ", fmt())), "refuse", "no data chunk")
audio_case("wav-lying-block-align", wav(fmt(block_align=4), silence(400, 4)), "refuse", "blockAlign must equal channels * bits / 8")
audio_case("wav-block-align-zero", wav(fmt(block_align=0, byte_rate=0), silence(800, 2)), "refuse", "blockAlign 0 would make the duration infinite or NaN")
audio_case("wav-byte-rate-mismatch", wav(fmt(byte_rate=8000), silence(800, 2)), "refuse", "byteRate must equal rate * blockAlign")
audio_case("wav-partial-frame", wav(fmt(), silence(800, 2) + b"\x00"), "refuse", "data must be a whole number of frames")
audio_case("wav-float-nan-sample", wav(fmt(tag=3, bits=32), silence(799, 4) + NAN), "refuse", "float samples must be finite")
audio_case("wav-float-inf-sample", wav(fmt(tag=3, bits=32), INF + silence(799, 4)), "refuse", "float samples must be finite")
audio_case("wav-10ms-clip", wav(fmt(), silence(80, 2)), "refuse", "0.01 s is below 0.1 s")
audio_case("wav-just-under-100ms", wav(fmt(), silence(799, 2)), "refuse", "799 frames at 8000 Hz")
audio_case("wav-empty-data", wav(fmt(), b""), "refuse", "zero duration")
audio_case("wav-30s-plus-one-frame", wav(fmt(bits=8), silence(240001, 1, True), pad=False), "refuse", "above 30 s")
audio_case("wav-riff-size-mismatch", wav(fmt(), silence(800, 2), riff_size=1000), "refuse", "RIFF size must be the byte length - 8")
audio_case("wav-not-wave", wav(fmt(), silence(800, 2)).replace(b"WAVE", b"AVI ", 1), "refuse", "RIFF form type must be WAVE")
audio_case("wav-chunk-beyond-buffer", wav(None, None, chunks=riff_chunk(b"fmt ", fmt()) + b"data" + struct.pack("<I", 4000) + silence(800, 2)), "refuse", "data size larger than the rest of the file")
audio_case("wav-trailing-bytes", wav(None, None, chunks=riff_chunk(b"fmt ", fmt()) + riff_chunk(b"data", silence(800, 2)) + b"xyz"), "refuse", "bytes after the last chunk that cannot hold a chunk header")
audio_case("wav-fmt-too-short", wav(fmt()[:14], silence(800, 2)), "refuse", "fmt shorter than 16 bytes")
audio_case("wav-pcm12-bits", wav(fmt(bits=12, block_align=2, byte_rate=16000), silence(800, 2)), "refuse", "PCM bits must be 8, 16, 24 or 32")
audio_case("wav-float64", wav(fmt(tag=3, bits=64), silence(800, 8)), "refuse", "IEEE float must be 32-bit")
audio_case("wav-adpcm", wav(fmt(tag=2, bits=16), silence(800, 2)), "refuse", "only PCM and IEEE float")
audio_case("wav-three-channels", wav(fmt(channels=3), silence(800, 6)), "refuse", "at most two channels")
audio_case("wav-zero-channels", wav(fmt(channels=0, block_align=0, byte_rate=0), silence(800, 2)), "refuse", "at least one channel")
audio_case("wav-rate-7999", wav(fmt(rate=7999), silence(800, 2)), "refuse", "rate below 8000 Hz")
audio_case("wav-rate-48001", wav(fmt(rate=48001), silence(4801, 2)), "refuse", "rate above 48000 Hz")
audio_case("wav-extensible-bad-guid", wav(extensible(guid_tail=bytes(14)), silence(800, 2)), "refuse", "SubFormat must be PCM or IEEE_FLOAT")
audio_case("wav-extensible-mulaw", wav(extensible(sub=7), silence(800, 2)), "refuse", "SubFormat must be PCM or IEEE_FLOAT")
audio_case("wav-extensible-valid-bits", wav(extensible(valid_bits=12), silence(800, 2)), "refuse", "validBits must equal bits")
audio_case("wav-extensible-cbsize", wav(extensible(cb_size=0), silence(800, 2)), "refuse", "cbSize must be 22")
audio_case("wav-extensible-short-fmt", wav(fmt(tag=0xFFFE, extra=b"\x00\x00"), silence(800, 2)), "refuse", "extensible fmt needs 40 bytes")
case("audio-not-riff", "audio", b64(b"OggS" + bytes(60)), "refuse", "not a RIFF file")


# ---- timestamps (C4) ------------------------------------------------------------------------------------------------

def timestamp_case(id, text, verdict, note, utc=None, offset_minutes=0, micros=0):
    facts = {}
    if verdict == "accept":
        seconds = calendar.timegm(utc) - offset_minutes * 60
        facts = {"epochMs": seconds * 1000 + (micros // 1000 if micros % 1000 == 0 else micros / 1000)}
    case(id, "timestamp", text, verdict, note, **facts)


timestamp_case("ts-utc-z", "2024-03-01T12:34:56Z", "accept", "seconds, Z", (2024, 3, 1, 12, 34, 56))
timestamp_case("ts-no-seconds", "2024-03-01T12:34Z", "accept", "seconds are optional", (2024, 3, 1, 12, 34, 0))
timestamp_case("ts-positive-offset", "2024-01-01T00:00:00.123456+05:30", "accept", "fraction and positive offset", (2024, 1, 1, 0, 0, 0), 330, 123456)
timestamp_case("ts-negative-offset", "2023-12-31T19:00:00.5-05:00", "accept", "negative offset, short fraction", (2023, 12, 31, 19, 0, 0), -300, 500000)
timestamp_case("ts-minus-zero-offset", "2024-06-30T23:59:59-00:00", "accept", "-00:00 is UTC", (2024, 6, 30, 23, 59, 59))
timestamp_case("ts-leap-day", "2024-02-29T00:00:00Z", "accept", "2024 is a leap year", (2024, 2, 29, 0, 0, 0))
timestamp_case("ts-leap-day-2000", "2000-02-29T00:00:00Z", "accept", "2000 is a leap year", (2000, 2, 29, 0, 0, 0))
timestamp_case("ts-earliest", "1900-01-01T00:00:00Z", "accept", "first instant of 1900", (1900, 1, 1, 0, 0, 0))
timestamp_case("ts-latest", "2199-12-31T23:59:59.999999-23:59", "accept", "last day of 2199 with the largest offset: the year rule applies to the written year, though the UTC instant falls in 2200", (2199, 12, 31, 23, 59, 59), -1439, 999999)
timestamp_case("ts-feb-30", "2024-02-30T00:00:00Z", "refuse", "no 30 February")
timestamp_case("ts-feb-29-non-leap", "2023-02-29T00:00:00Z", "refuse", "2023 is not a leap year")
timestamp_case("ts-feb-29-1900", "1900-02-29T00:00:00Z", "refuse", "1900 is not a leap year (proleptic Gregorian)")
timestamp_case("ts-feb-29-2100", "2100-02-29T00:00:00Z", "refuse", "2100 is not a leap year")
timestamp_case("ts-april-31", "2024-04-31T00:00:00Z", "refuse", "April has 30 days")
timestamp_case("ts-month-13", "2024-13-01T00:00:00Z", "refuse", "month above 12")
timestamp_case("ts-month-0", "2024-00-10T00:00:00Z", "refuse", "month below 1")
timestamp_case("ts-day-0", "2024-01-00T00:00:00Z", "refuse", "day below 1")
timestamp_case("ts-hour-24", "2024-01-01T24:00:00Z", "refuse", "24:00 is not an hour")
timestamp_case("ts-minute-60", "2024-01-01T00:60:00Z", "refuse", "minute above 59")
timestamp_case("ts-leap-second", "2016-12-31T23:59:60Z", "refuse", "second 60 is refused")
timestamp_case("ts-year-0000", "0000-01-01T00:00:00Z", "refuse", "year below 1900")
timestamp_case("ts-year-1899", "1899-12-31T23:59:59Z", "refuse", "year below 1900")
timestamp_case("ts-year-2200", "2200-01-01T00:00:00Z", "refuse", "year above 2199")
timestamp_case("ts-offset-99-99", "2024-01-01T00:00:00+99:99", "refuse", "offset hours above 23")
timestamp_case("ts-offset-24-00", "2024-01-01T00:00:00+24:00", "refuse", "offset hours above 23")
timestamp_case("ts-offset-minutes-60", "2024-01-01T00:00:00-23:60", "refuse", "offset minutes above 59")
timestamp_case("ts-offset-no-colon", "2024-01-01T00:00:00+0530", "refuse", "offset must be +HH:MM")
timestamp_case("ts-naive", "2024-01-01T00:00:00", "refuse", "no zone")
timestamp_case("ts-date-only", "2024-01-01", "refuse", "no time")
timestamp_case("ts-space-separator", "2024-01-01 00:00:00Z", "refuse", "T separator required")
timestamp_case("ts-lowercase", "2024-01-01t00:00:00z", "refuse", "T and Z are upper case")
timestamp_case("ts-single-digit-fields", "2024-1-1T0:00:00Z", "refuse", "two-digit fields")
timestamp_case("ts-seven-digit-fraction", "2024-01-01T00:00:00.1234567Z", "refuse", "at most six fraction digits")
timestamp_case("ts-empty-fraction", "2024-01-01T00:00:00.Z", "refuse", "a dot needs digits")
timestamp_case("ts-fraction-without-seconds", "2024-01-01T00:00.5Z", "refuse", "a fraction needs seconds")
timestamp_case("ts-trailing-newline", "2024-01-01T00:00:00Z\n", "refuse", "nothing after the zone")
timestamp_case("ts-fullwidth-digits", "２０２４-01-01T00:00:00Z", "refuse", "ASCII digits only")
timestamp_case("ts-five-digit-year", "+12024-01-01T00:00:00Z", "refuse", "four-digit years only")
timestamp_case("ts-empty", "", "refuse", "empty")

OUT.parent.mkdir(exist_ok=True)
document = {
    "about": ("Shared media and timestamp vectors (spec amendments C1-C4); generated by tests/python/make_media_vectors.py, "
              "do not edit. `input` is the cell text, or {parts: [[text, count], ...]} meaning the concatenation of each text "
              "repeated count times. Accepted vectors carry the facts both sides must report: base64 -> decoded length and "
              "sha256; image -> format, width, height; audio -> sampleRate, channels, frames; timestamp -> epochMs (UTC; a "
              "fractional part carries microseconds, compare within 0.002 ms). Header walks visit at most maxChunks "
              "chunks or segments (PNG: IHDR to IEND inclusive; JPEG: markers after SOI up to and including SOS; WAV: "
              "chunks after the RIFF header)."),
    "limits": {"mediaMaxBytes": MEDIA_MAX_BYTES, "maxBase64Chars": MAX_BASE64_CHARS, "maxSide": 4096, "maxPixels": 4096 * 4096,
               "maxChunks": MAX_CHUNKS, "minAudioSeconds": 0.1, "maxAudioSeconds": 30, "minYear": 1900, "maxYear": 2199},
    "vectors": vectors,
}
assert len({v["id"] for v in vectors}) == len(vectors)
OUT.write_text(json.dumps(document, indent=1, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
print(f"{len(vectors)} vectors -> {OUT}")
