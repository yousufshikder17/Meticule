"""Strict validation of media cells: standard base64, PNG/JPEG headers and WAV audio decoded to float32. Stdlib + NumPy.

Mirrors src/ml/media.ts rule for rule; tests/fixtures/media-vectors.json holds the cases both sides must decide alike.
Every refusal is RunnerError("invalid_media") with a fixed message, never a library exception, path or cell value.
Image pixels are decoded later by the runner that needs them (Pillow, with these bounds re-checked); here only headers.
"""
import base64
import re
import struct

import numpy as np

from ml_runtime_common import RunnerError

MEDIA_MAX_BYTES = 2 * 1024 * 1024
MAX_BASE64_CHARS = 4 * -(-MEDIA_MAX_BYTES // 3)  # 2,796,204: the base64 length of 2 MiB
MAX_SIDE, MAX_PIXELS = 4096, 4096 * 4096
MAX_CHUNKS = 10_000  # chunks or segments a header walk visits, so a crafted file cannot buy unbounded work
PNG_MAGIC, JPEG_MAGIC = b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"
# Start-of-frame markers that carry the image size (C4, C8 and CC are DHT, JPG and DAC).
JPEG_SOF = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))
PCM, IEEE_FLOAT, EXTENSIBLE = 1, 3, 0xFFFE
# KSDATAFORMAT_SUBTYPE_PCM / _IEEE_FLOAT are {0000000X-0000-0010-8000-00AA00389B71}: two format bytes, then this.
GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")
_BASE64_BODY = re.compile(r"[A-Za-z0-9+/]*")


def _refuse(message):
    raise RunnerError("invalid_media", message)


def decode_base64(text):
    """Bytes of a media cell: standard base64 by the grammar
    ^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$ (no data: prefix, URL-safe alphabet, whitespace or
    inner padding), at most 2,796,204 characters and 2 MiB decoded. The grammar is checked as "length a multiple of 4,
    then alphabet up to one or two final '='", which is equivalent and avoids a backtracking regex on 2.8 MB cells."""
    if not isinstance(text, str) or len(text) > MAX_BASE64_CHARS or len(text) % 4:
        _refuse("Media cell is not valid base64")
    body = text[:-2] if text.endswith("==") else text[:-1] if text.endswith("=") else text
    if not _BASE64_BODY.fullmatch(body):
        _refuse("Media cell is not valid base64")
    data = base64.b64decode(text, validate=True)
    if len(data) > MEDIA_MAX_BYTES:
        _refuse("Media cell exceeds 2 MiB")
    return data


def _png_size(data):
    """IHDR must be the first chunk (length 13); every chunk lies inside the buffer; the walk ends at IEND (bytes after
    it are never read) and refuses any later IHDR."""
    if len(data) < 33 or data[8:16] != b"\x00\x00\x00\x0dIHDR":
        _refuse("Malformed PNG")
    pos, chunks = 8, 0
    while True:
        chunks += 1
        if chunks > MAX_CHUNKS or len(data) - pos < 12:
            _refuse("Malformed PNG")
        length, kind = struct.unpack_from(">I4s", data, pos)
        if length > len(data) - pos - 12 or (kind == b"IHDR" and pos != 8):
            _refuse("Malformed PNG")
        if kind == b"IEND":
            return struct.unpack_from(">II", data, 16)
        pos += 12 + length


def _jpeg_size(data):
    """Walks marker segments from SOI to the first SOS. 0xFF fill bytes may precede a marker; every other marker before
    SOS must carry a length (>= 2) that stays inside the buffer. Exactly one SOFn, before SOS, whose length is
    8 + 3 * components (1..4); a second SOFn (a size that could differ from the one checked) is refused."""
    pos, segments, size = 2, 0, None
    while True:
        segments += 1
        while len(data) - pos >= 2 and data[pos] == 0xFF and data[pos + 1] == 0xFF:
            pos += 1
        if segments > MAX_CHUNKS or len(data) - pos < 4 or data[pos] != 0xFF:
            _refuse("Malformed JPEG")
        marker, length = data[pos + 1], (data[pos + 2] << 8) | data[pos + 3]
        if marker in (0x00, 0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or length < 2 or length > len(data) - pos - 2:
            _refuse("Malformed JPEG")
        if marker in JPEG_SOF:
            components = data[pos + 9] if length >= 8 else 0
            if size is not None or not 1 <= components <= 4 or length != 8 + 3 * components:
                _refuse("Malformed JPEG")
            height, width = struct.unpack_from(">HH", data, pos + 5)
            size = (width, height)
        elif marker == 0xDA:
            if size is None:
                _refuse("Malformed JPEG")
            return size
        pos += 2 + length


def image_header(data):
    """("png" | "jpeg", width, height) of a PNG or JPEG from its header walk (C2): each side 1..4096, at most
    16,777,216 pixels."""
    if data[:8] == PNG_MAGIC:
        kind, (width, height) = "png", _png_size(data)
    elif data[:3] == JPEG_MAGIC:
        kind, (width, height) = "jpeg", _jpeg_size(data)
    else:
        _refuse("Image must be PNG or JPEG")
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE and width * height <= MAX_PIXELS):
        _refuse("Image dimensions out of bounds")
    return kind, width, height


def wav_layout(data):
    """The C3 walk. RIFF size == byte length - 8; every chunk lies inside the buffer and an odd-sized one is followed by
    a pad byte (absent only at the very end); other chunk types are skipped; exactly one `fmt ` before exactly one
    `data`; PCM 8/16/24/32-bit or IEEE float 32-bit, directly or as WAVE_FORMAT_EXTENSIBLE (cbSize 22, SubFormat
    PCM/IEEE_FLOAT, validBits == bits); 1-2 channels; 8000-48000 Hz; blockAlign == channels * bits / 8; byteRate ==
    rate * blockAlign; data a whole number of frames lasting 0.1 to 30 seconds; float samples finite.
    Returns {"format": "pcm" | "float", "bits", "channels", "rate", "frames", "offset"}."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE" or struct.unpack_from("<I", data, 4)[0] != len(data) - 8:
        _refuse("Malformed WAV")
    pos, chunks, fmt, body = 12, 0, None, None
    while pos < len(data):
        chunks += 1
        if chunks > MAX_CHUNKS or len(data) - pos < 8:
            _refuse("Malformed WAV")
        kind, size = struct.unpack_from("<4sI", data, pos)
        if size > len(data) - pos - 8:
            _refuse("Malformed WAV")
        if kind == b"fmt ":
            if fmt is not None or body is not None:
                _refuse("Malformed WAV")
            fmt = data[pos + 8:pos + 8 + size]
        elif kind == b"data":
            if fmt is None or body is not None:
                _refuse("Malformed WAV")
            body = (pos + 8, size)
        pos += 8 + size + (size & 1)
    if body is None or len(fmt) < 16:
        _refuse("Malformed WAV")
    tag, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == EXTENSIBLE:
        if len(fmt) < 40 or struct.unpack_from("<HH", fmt, 16) != (22, bits) or fmt[26:40] != GUID_TAIL or fmt[24:26] not in (b"\x01\x00", b"\x03\x00"):
            _refuse("Unsupported WAV format")
        tag = fmt[24]
    if not ((tag == PCM and bits in (8, 16, 24, 32)) or (tag == IEEE_FLOAT and bits == 32)):
        _refuse("Unsupported WAV format")
    if not (1 <= channels <= 2 and 8000 <= rate <= 48000):
        _refuse("Unsupported WAV format")
    if block_align != channels * bits // 8 or byte_rate != rate * block_align or body[1] % block_align:
        _refuse("Malformed WAV")
    frames = body[1] // block_align
    duration = frames / rate
    if not (duration >= 0.1 and duration <= 30):
        _refuse("Audio duration out of bounds")
    if tag == IEEE_FLOAT and not np.isfinite(np.frombuffer(data, "<f4", frames * channels, body[0])).all():
        _refuse("Audio samples must be finite")
    return {"format": "float" if tag == IEEE_FLOAT else "pcm", "bits": bits, "channels": channels, "rate": rate,
            "frames": frames, "offset": body[0]}


def read_wav(data):
    """(samples, rate) of a WAV that passes wav_layout: float32 samples shaped (frames, channels), PCM scaled to [-1, 1)
    (8-bit is unsigned around 128). Only this walker decodes audio; never scipy.io.wavfile."""
    layout = wav_layout(data)
    count, offset, bits = layout["frames"] * layout["channels"], layout["offset"], layout["bits"]
    if layout["format"] == "float":
        samples = np.frombuffer(data, "<f4", count, offset)
    elif bits == 8:
        samples = (np.frombuffer(data, np.uint8, count, offset).astype(np.float32) - 128) / 128
    elif bits == 24:
        raw = np.frombuffer(data, np.uint8, count * 3, offset).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (value - ((value & 0x800000) << 1)) / 8388608
    else:
        samples = np.frombuffer(data, "<i2" if bits == 16 else "<i4", count, offset) / float(1 << (bits - 1))
    return samples.astype(np.float32).reshape(layout["frames"], layout["channels"]), layout["rate"]
