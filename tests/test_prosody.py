"""Dynamic Prosody — emotion → ElevenLabs voice & settings."""

from __future__ import annotations

from memory.prosody import _PROFILES, profile_for, to_elevenlabs_voice
from memory.schemas import Emotion

DEFAULT = "voice-default"
CALM = "voice-calm"


def test_neutral_emotion_uses_default_voice_and_profile_settings():
    profile = profile_for(Emotion.NEUTRAL)
    voice_id, settings = to_elevenlabs_voice(
        Emotion.NEUTRAL, profile, default_voice_id=DEFAULT, calm_voice_id=CALM
    )
    assert voice_id == DEFAULT
    assert settings.stability == profile.stability
    assert settings.speed == profile.speed


def test_frustrated_caller_gets_calm_voice_and_high_stability():
    profile = profile_for(Emotion.FRUSTRATED)
    voice_id, settings = to_elevenlabs_voice(
        Emotion.FRUSTRATED, profile, default_voice_id=DEFAULT, calm_voice_id=CALM
    )
    assert voice_id == CALM                                # Dynamic Prosody headline
    assert settings.stability == 0.95                      # overrides profile's 0.35
    # Speed still tracks the empathetic-slow profile so cadence stays soothing.
    assert settings.speed == profile.speed


def test_angry_caller_also_takes_calm_path():
    profile = profile_for(Emotion.ANGRY)
    voice_id, settings = to_elevenlabs_voice(
        Emotion.ANGRY, profile, default_voice_id=DEFAULT, calm_voice_id=CALM
    )
    assert voice_id == CALM
    assert settings.stability == 0.95


def test_all_archived_profiles_still_present():
    # Guard rail: don't silently lose a ported profile in a future refactor.
    for emotion in (Emotion.FRUSTRATED, Emotion.ANGRY, Emotion.ANXIOUS, Emotion.HAPPY):
        assert emotion in _PROFILES
