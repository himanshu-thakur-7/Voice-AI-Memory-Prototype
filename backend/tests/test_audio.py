"""µ-law codec + framing correctness — the audio path Twilio depends on."""

from __future__ import annotations

from app.telephony.audio import (
    FRAME_BYTES,
    frame_ulaw,
    pcm16_to_ulaw,
    rms_energy,
    silence_ulaw,
    ulaw_to_pcm16,
)


def test_frame_size_is_160():
    assert FRAME_BYTES == 160  # 8kHz * 20ms * 1 byte/sample


def test_ulaw_roundtrip_is_close():
    # Build a ramp PCM16 signal, encode→decode, expect small µ-law quantization error.
    pcm = bytearray()
    for i in range(160):
        s = (i * 200) - 16000
        pcm += int(s & 0xFFFF).to_bytes(2, "little")
    back = ulaw_to_pcm16(pcm16_to_ulaw(bytes(pcm)))
    assert len(back) == len(pcm)
    # µ-law is lossy but monotonic; check correlation of signs at least.
    orig = _samples(bytes(pcm))
    dec = _samples(back)
    sign_match = sum(1 for a, b in zip(orig, dec, strict=True) if (a >= 0) == (b >= 0))
    assert sign_match / len(orig) > 0.95


def test_silence_has_low_energy_and_tone_has_high():
    assert rms_energy(ulaw_to_pcm16(silence_ulaw(20))) < 0.01
    pcm = bytearray()
    for i in range(160):
        pcm += int((10000 if i % 2 else -10000) & 0xFFFF).to_bytes(2, "little")
    assert rms_energy(bytes(pcm)) > 0.2


def test_frame_ulaw_pads_last_frame():
    data = b"\x10" * (FRAME_BYTES + 5)
    frames = list(frame_ulaw(data))
    assert len(frames) == 2
    assert all(len(f) == FRAME_BYTES for f in frames)
    assert frames[1].endswith(b"\xff")  # padded with µ-law silence


def _samples(pcm16: bytes) -> list[int]:
    out = []
    for i in range(0, len(pcm16) - 1, 2):
        s = pcm16[i] | (pcm16[i + 1] << 8)
        out.append(s - 0x10000 if s >= 0x8000 else s)
    return out
