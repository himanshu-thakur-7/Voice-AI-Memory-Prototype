"""Paralinguistic extractor — pure mappers + graceful fallback when deps are absent."""

from __future__ import annotations

from app.config import Settings
from app.engines.base import CallContext
from app.memory.paralinguistics import (
    _parse_sensevoice,
    map_emotion,
    profile_to_state,
    run_paralinguistics,
)
from app.memory.schemas import Emotion


def test_parse_sensevoice_tags():
    clean, emotion, events = _parse_sensevoice(
        "<|zh|><|NEUTRAL|><|Speech|> Hello there <|Laughter|>"
    )
    assert clean == "Hello there"
    assert emotion == "neutral"
    assert events == ["laughter"]


def test_parse_sensevoice_dominant_emotion():
    _, emotion, _ = _parse_sensevoice("<|HAPPY|> a <|HAPPY|> b <|SAD|> c")
    assert emotion == "happy"  # majority vote


def test_map_emotion_uses_nuanced_state():
    assert map_emotion("neutral", "psychopathic_threat") is Emotion.ANGRY
    assert map_emotion("neutral", "cynical_or_masking_grief") is Emotion.SAD
    assert map_emotion("neutral", "tense_suppressed") is Emotion.ANXIOUS
    assert map_emotion("neutral", "genuine_joy") is Emotion.HAPPY


def test_map_emotion_falls_back_to_acoustic():
    assert map_emotion("angry", "") is Emotion.ANGRY
    assert map_emotion("happy", "") is Emotion.HAPPY
    assert map_emotion("", "") is Emotion.NEUTRAL


def test_profile_to_state_maps_and_stores_profile():
    profile = {
        "base_acoustic_emotion": "neutral",
        "final_affective_state": "tense_suppressed",
        "lexical_sentiment": "positive",
        "detected_events": ["cry"],
        "transcript": "i'm fine, really",
        "acoustic_biometrics": {
            "jitter_local": 0.06, "silence_ratio": 0.25,
            "pitch_trajectory": "falling (vocal fry, defeat, or fatigue)",
        },
    }
    state = profile_to_state(profile, CallContext(tenant_id="t", user_id="u"))
    assert state.emotion is Emotion.ANXIOUS               # masked stress, not the cheerful words
    assert state.paralinguistics["final_affective_state"] == "tense_suppressed"
    assert state.confidence >= 0.7                        # a real contradiction was reconciled
    assert "jitter_local" in state.features               # numeric biometric kept
    assert "pitch_trajectory" not in state.features       # non-numeric stays in paralinguistics only


async def test_run_paralinguistics_disabled_returns_none():
    # heuristic mode never invokes the heavy engine
    out = await run_paralinguistics(b"\xff" * 320, CallContext(), Settings(affect_extractor="heuristic"))
    assert out is None


async def test_run_paralinguistics_no_audio_returns_none():
    out = await run_paralinguistics(b"", CallContext(), Settings(affect_extractor="auto"))
    assert out is None
