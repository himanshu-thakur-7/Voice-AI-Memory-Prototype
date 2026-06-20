"""Prosody profiles + ElevenLabs translation.

Ports the ``_PROFILES`` dict from the archived prototype (Emotion → ProsodyProfile) so the
voice and the system prompt change together — the "what" and the "how" are inseparable. The
ProsodyProfile dataclass lives here rather than in ``schemas.py`` so the brain modules can
import it without dragging in the LiveKit plugins.

``to_elevenlabs_voice()`` is the only place that knows which voice ID to use for which
emotion — that's where **Dynamic Prosody** physically lands (frustrated/angry callers get
the calm voice ID with high stability).
"""

from __future__ import annotations

from dataclasses import dataclass

from memory.schemas import Emotion, VoiceSettings


@dataclass(frozen=True)
class ProsodyProfile:
    """How to *speak* + what extra system-prompt directive to inject."""

    label: str
    stability: float                # ElevenLabs stability  (0..1)
    style: float                    # ElevenLabs style      (0..1)
    speed: float                    # ElevenLabs playback speed (≈0.8..1.2)
    similarity_boost: float = 0.75
    system_prompt_suffix: str = ""

    @staticmethod
    def neutral() -> ProsodyProfile:
        return ProsodyProfile(label="neutral", stability=0.5, style=0.2, speed=1.0)


# Constants ported verbatim from archive/backend/app/memory/precall.py — the wording of the
# system_prompt_suffix lines has been tuned through demos; don't drift them casually.
_PROFILES: dict[Emotion, ProsodyProfile] = {
    Emotion.FRUSTRATED: ProsodyProfile(
        label="empathetic-slow", stability=0.35, style=0.25, speed=0.92,
        system_prompt_suffix=(
            "The caller was frustrated on a previous call. Open by acknowledging it, "
            "stay calm and warm, and get to a concrete solution quickly."
        ),
    ),
    Emotion.ANGRY: ProsodyProfile(
        label="de-escalate", stability=0.30, style=0.30, speed=0.90,
        system_prompt_suffix=(
            "The caller has been angry. De-escalate: be brief, validating, and solution-first; "
            "do not be cheerful or defensive."
        ),
    ),
    Emotion.ANXIOUS: ProsodyProfile(
        label="reassuring", stability=0.45, style=0.15, speed=0.95,
        system_prompt_suffix="The caller tends to be anxious. Be reassuring, clear, and steady.",
    ),
    Emotion.SAD: ProsodyProfile(
        label="gentle", stability=0.55, style=0.10, speed=0.93,
        system_prompt_suffix=(
            "The caller has been sad or low. Open gently, do not be cheerful or rushed."
        ),
    ),
    Emotion.HAPPY: ProsodyProfile(
        label="upbeat", stability=0.5, style=0.2, speed=1.03,
        system_prompt_suffix="The caller is upbeat. Match their positive energy, stay concise.",
    ),
}


def profile_for(emotion: Emotion) -> ProsodyProfile:
    """Look up the profile for an emotion; neutral fallback."""
    return _PROFILES.get(emotion, ProsodyProfile.neutral())


def to_elevenlabs_voice(
    emotion: Emotion,
    profile: ProsodyProfile,
    *,
    default_voice_id: str,
    calm_voice_id: str,
) -> tuple[str, VoiceSettings]:
    """Translate (emotion, profile) → (voice_id, VoiceSettings).

    The headline policy: **frustrated/angry callers get the calm voice with 0.95 stability**
    — overriding the per-profile stability — so the voice itself sounds soothing regardless
    of the model's pitch. Everyone else gets the default voice with the profile's settings.
    """
    if emotion in (Emotion.FRUSTRATED, Emotion.ANGRY):
        return calm_voice_id, VoiceSettings(
            stability=0.95,                          # Dynamic Prosody headline knob
            similarity_boost=profile.similarity_boost,
            style=profile.style,
            speed=profile.speed,
        )
    return default_voice_id, VoiceSettings(
        stability=profile.stability,
        similarity_boost=profile.similarity_boost,
        style=profile.style,
        speed=profile.speed,
    )
