"""Audio front-end (NumPy + SciPy + stdlib): WAV cells to fixed-length MFCC summary vectors.

Decoding goes only through media_common's chunk walker (never scipy.io.wavfile). Policy, all deterministic:
- channels: mono by the arithmetic mean of the channels;
- rate: resampled to the target rate with scipy.signal.resample_poly(up, down) over the reduced ratio (its default Kaiser
  anti-aliasing filter);
- clips must be at least 0.1 s (the walker enforces 0.1..30 s);
- features: frames of `frame_ms` with hop `hop_ms`, a periodic Hann window, |rfft|^2 power, an HTK mel filterbank of `n_mels`
  triangles from 0 Hz to Nyquist, log(max(mel energy, 1e-10)), DCT-II with orthonormal scaling keeping `n_mfcc`
  coefficients; the summary is each coefficient's mean then standard deviation over frames (length 2 * n_mfcc).
Clips are processed one at a time.
"""
import math

import numpy as np
import scipy.fft
import scipy.signal

import media_common

LOG_FLOOR = 1e-10
LIMITS = {"sampleRate": (8000, 48000), "nMfcc": (8, 40), "nMels": (16, 128), "frameMs": (10, 50), "hopMs": (5, 25)}
DEFAULTS = {"sampleRate": 16000, "nMfcc": 13, "nMels": 40, "frameMs": 25, "hopMs": 10}


def decode(cell, rate):
    """A WAV cell as mono float32 samples at `rate`."""
    samples, source_rate = media_common.read_wav(media_common.decode_base64(cell))
    mono = samples.mean(axis=1, dtype=np.float64)
    if source_rate != rate:
        common = math.gcd(rate, source_rate)
        mono = scipy.signal.resample_poly(mono, rate // common, source_rate // common)
    return mono.astype(np.float32)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, rate):
    """HTK-style triangular filters on the rfft bins, from 0 Hz to Nyquist."""
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(rate / 2.0), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / rate)
    bank = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        low, centre, high = edges[m], edges[m + 1], edges[m + 2]
        rising = (bins - low) / max(centre - low, 1e-12)
        falling = (high - bins) / max(high - centre, 1e-12)
        bank[m] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def mfcc(signal, rate, n_mfcc, n_mels, frame_ms, hop_ms):
    """(frames, n_mfcc) MFCC matrix. A clip shorter than one frame is zero-padded to one frame."""
    frame, hop = int(round(rate * frame_ms / 1000.0)), int(round(rate * hop_ms / 1000.0))
    n_fft = 1 << (frame - 1).bit_length()
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    count = 1 + (len(signal) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = signal[index].astype(np.float64) * scipy.signal.get_window("hann", frame, fftbins=True)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    log_mel = np.log(np.maximum(power @ mel_filterbank(n_mels, n_fft, rate).T, LOG_FLOOR))
    return scipy.fft.dct(log_mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]


def summary(cell, settings):
    """The fixed-length feature vector of one WAV cell: per-coefficient mean, then per-coefficient standard deviation."""
    coefficients = mfcc(decode(cell, settings["sampleRate"]), settings["sampleRate"], settings["nMfcc"], settings["nMels"], settings["frameMs"], settings["hopMs"])
    vector = np.concatenate([coefficients.mean(axis=0), coefficients.std(axis=0)])
    if not np.isfinite(vector).all():
        raise ValueError("Audio features are not finite")
    return vector


def features(cells, settings):
    """(rows, 2 * n_mfcc), one clip at a time."""
    width = 2 * settings["nMfcc"]
    return np.vstack([summary(cell, settings) for cell in cells]) if cells else np.zeros((0, width))


def resolve_settings(supplied):
    """Bounded front-end settings; frames must hold at least one hop and fit the frame inside 50 ms."""
    settings = {}
    for name, (low, high) in LIMITS.items():
        value = supplied.get(name, DEFAULTS[name])
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"Audio setting {name} out of range")
        settings[name] = value
    if settings["hopMs"] > settings["frameMs"] or settings["nMfcc"] > settings["nMels"]:
        raise ValueError("Audio settings are inconsistent")
    return settings
