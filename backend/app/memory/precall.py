"""Module 3a — Pre-Call Dynamic Prosody.

Before the bot says a word, look up the caller's stored affective state (or the hint the
Go scheduler passed). If they were, say, *frustrated* last time, override prosody to be
slower and more empathetic AND inject a matching directive into the LLM system prompt —
so both *what* the bot says and *how* it sounds adapt from the first syllable.

The prosody lands at ElevenLabs TTS-context init (voice_settings are locked per context —
see providers/tts_elevenlabs.py) and as instruction text for the realtime engine.
"""

from __future__ import annotations

from app.config import Settings
from app.engines.base import CallContext, ProsodyProfile
from app.logging import get_logger
from app.memory.schemas import AffectiveState, Emotion
from app.memory.store import affective_store

log = get_logger(__name__)

# emotion → (prosody, system-prompt directive)
_PROFILES: dict[Emotion, ProsodyProfile] = {
    Emotion.FRUSTRATED: ProsodyProfile(
        label="empathetic-slow", stability=0.35, style=0.25, speed=0.92,
        system_prompt_suffix=(
            "The caller was frustrated on a previous call. Open by acknowledging it, "
            "stay calm and warm, and get to a concrete solution quickly."
        ),
        realtime_instructions="Speak slowly and warmly; acknowledge any frustration first.",
    ),
    Emotion.ANGRY: ProsodyProfile(
        label="de-escalate", stability=0.30, style=0.30, speed=0.90,
        system_prompt_suffix=(
            "The caller has been angry. De-escalate: be brief, validating, and solution-first; "
            "do not be cheerful or defensive."
        ),
        realtime_instructions="Calm, low-energy, validating tone. De-escalate.",
    ),
    Emotion.ANXIOUS: ProsodyProfile(
        label="reassuring", stability=0.45, style=0.15, speed=0.95,
        system_prompt_suffix="The caller tends to be anxious. Be reassuring, clear, and steady.",
        realtime_instructions="Reassuring, steady, unhurried.",
    ),
    Emotion.HAPPY: ProsodyProfile(
        label="upbeat", stability=0.5, style=0.2, speed=1.03,
        system_prompt_suffix="The caller is upbeat. Match their positive energy, stay concise.",
        realtime_instructions="Warm, upbeat, energetic.",
    ),
}


async def resolve_prosody(ctx: CallContext, settings: Settings) -> None:
    """Populate ctx.prosody and ctx.system_prompt from the caller's affective state."""
    emotion = await _lookup_emotion(ctx, settings)
    profile = _PROFILES.get(emotion, ProsodyProfile.neutral())
    ctx.prosody = profile

    base = settings.base_system_prompt
    ctx.system_prompt = f"{base} {profile.system_prompt_suffix}".strip()

    log.info("precall.prosody", call=ctx.call_sid, user=ctx.user_id,
             emotion=emotion.value, profile=profile.label)


async def _lookup_emotion(ctx: CallContext, settings: Settings) -> Emotion:
    # Stored state wins; otherwise fall back to the scheduler's cheap hint.
    state = await affective_store(settings).get_state(ctx.tenant_id, ctx.user_id)
    if state is not None:
        return state.emotion
    if ctx.affective_hint:
        try:
            return Emotion(ctx.affective_hint)
        except ValueError:
            pass
    return Emotion.NEUTRAL
