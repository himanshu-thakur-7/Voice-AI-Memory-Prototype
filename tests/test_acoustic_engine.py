"""Acoustic engine — librosa fallback path + pure helpers.

The rich SenseVoice path is gated by heavy deps (funasr/torch); these tests skip when
those aren't installed but still cover the deterministic logic on top: the parser, the
mapping, the affect heuristic, and the librosa-only ``analyze_audio`` round-trip via a
synthesized WAV.
"""

from __future__ import annotations

import importlib.util
import os
import wave

import pytest

from config import Settings
from memory.acoustic_engine import (
    AcousticResult,
    _affect_from_rich,
    _classify_acoustic_affect,
    _parse_sensevoice,
    analyze_audio,
    map_emotion,
    valence_arousal,
)
from memory.schemas import Emotion

# ── pure helpers ─────────────────────────────────────────────────────────────


def test_classify_acoustic_affect_thresholds():
    assert _classify_acoustic_affect(40.0, 0.08) == "agitated"
    assert _classify_acoustic_affect(3.0, 0.005) == "subdued"
    assert _classify_acoustic_affect(15.0, 0.03) == "neutral"


def test_classify_acoustic_affect_only_loud_or_only_wide_is_neutral():
    # "Loud but flat" (frustrated-but-controlled) ≠ agitated; we need BOTH signals.
    assert _classify_acoustic_affect(2.0, 0.10) == "neutral"
    assert _classify_acoustic_affect(50.0, 0.01) == "neutral"


def test_parse_sensevoice_dominant_emotion_and_events():
    clean, emotion, events = _parse_sensevoice(
        "<|zh|><|HAPPY|><|Speech|> i love this <|Laughter|>"
    )
    assert clean == "i love this"
    assert emotion == "happy"
    assert events == ["laughter"]


def test_parse_sensevoice_majority_vote():
    _, emotion, _ = _parse_sensevoice("<|SAD|> a <|SAD|> b <|HAPPY|> c")
    assert emotion == "sad"


def test_map_emotion_prefers_nuanced_state():
    assert map_emotion("neutral", "psychopathic_threat") is Emotion.ANGRY
    assert map_emotion("neutral", "cynical_or_masking_grief") is Emotion.SAD
    assert map_emotion("neutral", "tense_suppressed") is Emotion.ANXIOUS
    assert map_emotion("neutral", "genuine_joy") is Emotion.HAPPY


def test_map_emotion_falls_back_to_acoustic_when_final_state_empty():
    assert map_emotion("angry", "") is Emotion.ANGRY
    assert map_emotion("", "") is Emotion.NEUTRAL


def test_valence_arousal_bumps_with_jitter():
    base_v, base_a = valence_arousal(Emotion.FRUSTRATED, {})
    bumped_v, bumped_a = valence_arousal(Emotion.FRUSTRATED, {"jitter_local": 0.06})
    assert base_v == bumped_v        # valence unaffected
    assert bumped_a > base_a         # arousal climbs with vocal tension


def test_affect_from_rich_sarcasm_keywords_force_agitated():
    assert _affect_from_rich("cynical_or_masking_grief", "sad", {}) == "agitated"
    assert _affect_from_rich("psychopathic_threat", "neutral", {}) == "agitated"


def test_affect_from_rich_grief_keywords_force_subdued():
    assert _affect_from_rich("genuine_grief", "sad", {}) == "subdued"
    assert _affect_from_rich("defeated_apathy", "neutral", {}) == "subdued"


# ── librosa fallback (real synthesized audio) ────────────────────────────────


def _write_sine_wav(path: str, *, freq: float, duration: float = 1.0,
                    amplitude: float = 0.2, sample_rate: int = 16000) -> None:
    """Synthesize a pure tone WAV — librosa's f0 estimator can lock onto it deterministically."""
    import math
    import struct

    n = int(sample_rate * duration)
    frames = bytearray()
    for i in range(n):
        sample = int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", sample)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))


@pytest.mark.skipif(
    importlib.util.find_spec("librosa") is None,
    reason="librosa not installed",
)
def test_analyze_audio_on_synthesized_tone(tmp_path):
    """A 220 Hz tone @ amplitude 0.2 is a calm, steady voice analog → neutral or subdued.

    Either is acceptable; the point is the function ran end-to-end on a real WAV and
    populated ``pitch_variance`` / ``rms_energy`` rather than raising.
    """
    wav_path = str(tmp_path / "tone.wav")
    _write_sine_wav(wav_path, freq=220.0, duration=1.0, amplitude=0.2)
    out = analyze_audio(wav_path, Settings(affect_extractor="librosa"))
    assert isinstance(out, AcousticResult)
    assert out.engine == "librosa"
    assert out.rms_energy > 0.0
    # A steady pure tone has near-zero pitch variance.
    assert out.pitch_variance < 5.0
    assert out.acoustic_affect in {"neutral", "subdued"}


def test_analyze_audio_returns_empty_on_missing_path():
    out = analyze_audio("", Settings(affect_extractor="librosa"))
    assert out == AcousticResult()


def test_analyze_audio_returns_empty_on_corrupted_file(tmp_path):
    # An empty WAV header — fails cleanly without librosa falling through to audioread
    # (which is 30+s of MP3/FLAC probing on garbage bytes).
    bad = tmp_path / "garbage.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    out = analyze_audio(str(bad), Settings(affect_extractor="librosa"))
    # Soft-fail → defaults; the post-call worker should never crash on bad audio.
    assert isinstance(out, AcousticResult)
    assert out.pitch_variance == 0.0
    assert out.rms_energy == 0.0
    assert os.path.exists(bad)   # we didn't accidentally clobber it
