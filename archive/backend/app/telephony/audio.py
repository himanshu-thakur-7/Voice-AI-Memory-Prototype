"""µ-law (G.711) audio helpers for the Twilio media path.

Twilio sends and expects **base64 G.711 µ-law, 8 kHz, mono, 20 ms / 160-byte frames**.
ElevenLabs Flash can emit ``ulaw_8000`` directly, so on a well-built pipeline almost
no resampling happens — which is exactly why audio transcoding is NOT a meaningful
latency source (single-digit ms). See docs/latency-budget.md.

We prefer the stdlib ``audioop`` for speed but ship a tiny pure-Python µ-law codec so
the prototype (and its tests) run anywhere, including Python 3.13 where ``audioop`` was
removed from the stdlib (install ``audioop-lts`` for the fast path in production).
"""

from __future__ import annotations

from collections.abc import Iterator

TWILIO_SAMPLE_RATE = 8000
FRAME_MS = 20
# 8000 samples/s * 0.02 s * 1 byte/sample(µ-law) = 160 bytes per 20 ms frame.
FRAME_BYTES = TWILIO_SAMPLE_RATE * FRAME_MS // 1000  # 160

try:  # fast path
    import audioop  # type: ignore

    _HAVE_AUDIOOP = True
except ImportError:  # pragma: no cover - exercised only where audioop is absent
    _HAVE_AUDIOOP = False


# ── pure-Python G.711 µ-law fallback ─────────────────────────────────────────
_BIAS = 0x84
_CLIP = 32635


def _linear_to_ulaw_sample(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > _CLIP:
        sample = _CLIP
    sample += _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _ulaw_to_linear_sample(u: int) -> int:
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return -sample if sign else sample


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode µ-law bytes → 16-bit little-endian PCM (8 kHz)."""
    if _HAVE_AUDIOOP:
        return audioop.ulaw2lin(data, 2)
    out = bytearray(len(data) * 2)
    for i, b in enumerate(data):
        s = _ulaw_to_linear_sample(b)
        out[2 * i] = s & 0xFF
        out[2 * i + 1] = (s >> 8) & 0xFF
    return bytes(out)


def pcm16_to_ulaw(data: bytes) -> bytes:
    """Encode 16-bit little-endian PCM → µ-law bytes."""
    if _HAVE_AUDIOOP:
        return audioop.lin2ulaw(data, 2)
    out = bytearray(len(data) // 2)
    for i in range(len(out)):
        lo = data[2 * i]
        hi = data[2 * i + 1]
        sample = lo | (hi << 8)
        if sample >= 0x8000:
            sample -= 0x10000
        out[i] = _linear_to_ulaw_sample(sample)
    return bytes(out)


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16. Used only for STT providers that want 16 kHz."""
    if src_rate == dst_rate:
        return data
    if _HAVE_AUDIOOP:
        converted, _ = audioop.ratecv(data, 2, 1, src_rate, dst_rate, None)
        return converted
    # naive linear fallback
    src = _to_samples(data)
    n_out = max(1, round(len(src) * dst_rate / src_rate))
    out = []
    for i in range(n_out):
        pos = i * (len(src) - 1) / max(1, n_out - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, len(src) - 1)
        out.append(int(src[lo] * (1 - frac) + src[hi] * frac))
    return _from_samples(out)


def rms_energy(pcm16: bytes) -> float:
    """Root-mean-square amplitude of a PCM16 frame, normalised to 0..1. Used by VAD."""
    if not pcm16:
        return 0.0
    if _HAVE_AUDIOOP:
        return audioop.rms(pcm16, 2) / 32768.0
    samples = _to_samples(pcm16)
    mean_sq = sum(s * s for s in samples) / len(samples)
    return (mean_sq**0.5) / 32768.0


def frame_ulaw(data: bytes, frame_bytes: int = FRAME_BYTES) -> Iterator[bytes]:
    """Slice a µ-law buffer into fixed 20 ms frames, padding the last with silence.

    Twilio plays smoothest with consistently paced 160-byte frames; TTS engines emit
    irregular chunk sizes, so the writer re-packetises through this before sending.
    """
    for off in range(0, len(data), frame_bytes):
        chunk = data[off : off + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\xff" * (frame_bytes - len(chunk))  # 0xFF = µ-law silence
        yield chunk


def silence_ulaw(ms: int) -> bytes:
    """A µ-law silence buffer of the given duration (0xFF is µ-law zero)."""
    return b"\xff" * (TWILIO_SAMPLE_RATE * ms // 1000)


def ulaw_to_wav(ulaw: bytes, rate: int = TWILIO_SAMPLE_RATE) -> bytes:
    """Wrap µ-law audio as a PCM16 WAV so a browser <audio> element can play it.

    (Browsers don't reliably decode µ-law WAV, so we decode to linear PCM16 first.)
    """
    import struct

    pcm = ulaw_to_pcm16(ulaw)
    n = len(pcm)
    header = (
        b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", n)
    )
    return header + pcm


def _to_samples(pcm16: bytes) -> list[int]:
    out = []
    for i in range(0, len(pcm16) - 1, 2):
        s = pcm16[i] | (pcm16[i + 1] << 8)
        if s >= 0x8000:
            s -= 0x10000
        out.append(s)
    return out


def _from_samples(samples: list[int]) -> bytes:
    out = bytearray(len(samples) * 2)
    for i, s in enumerate(samples):
        s = max(-32768, min(32767, int(s))) & 0xFFFF
        out[2 * i] = s & 0xFF
        out[2 * i + 1] = (s >> 8) & 0xFF
    return bytes(out)
