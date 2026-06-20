"""Pre-call orchestration — runs once on participant_connected.

This is where three of the four "wow" features come together:
  • **Dynamic Prosody**     — emotion → ElevenLabs voice & settings (frustrated → calm voice)
  • **Adaptive Verbosity**  — conversation_style → "10 words or less" constraint
  • **Proactive Empathy**   — severe negative last event → spoken opening line about it

The actual Neo4j read lives in ``graph_engine.CognitiveGraph`` (Step 2). This module talks
to it through a tiny Protocol so pre_call is testable without a database and ``agent.py``
can boot with the ``NullGraph`` stub until Step 2 lands.
"""

from __future__ import annotations

from typing import Protocol

import structlog

from config import Settings
from memory import prosody
from memory.schemas import (
    ConversationStyle,
    Emotion,
    ParticipantContext,
    PrecallResult,
    UserGraphContext,
)

log = structlog.get_logger(__name__)


# ── what pre_call needs from the graph ───────────────────────────────────────


class GraphContextReader(Protocol):
    """The slice of CognitiveGraph that pre_call depends on."""

    async def get_user_context(
        self, tenant_id: str, user_id: str
    ) -> UserGraphContext: ...


class NullGraph:
    """Stub used until ``graph_engine.py`` lands — always returns an empty context, so the
    agent boots and behaves as a stock voice assistant (no history → no personalization)."""

    async def get_user_context(
        self, tenant_id: str, user_id: str
    ) -> UserGraphContext:
        return UserGraphContext()


# ── main entrypoint ──────────────────────────────────────────────────────────


async def build_precall_context(
    graph: GraphContextReader,
    pctx: ParticipantContext,
    settings: Settings,
) -> PrecallResult:
    """Compose the per-call system prompt, voice, settings, and opening greeting."""
    user_ctx = await graph.get_user_context(pctx.tenant_id, pctx.user_id)
    emotion = (
        user_ctx.affective_state.emotion if user_ctx.affective_state else Emotion.NEUTRAL
    )
    profile = prosody.profile_for(emotion)

    # 1. Dynamic Prosody — voice + settings.
    voice_id, voice_settings = prosody.to_elevenlabs_voice(
        emotion, profile,
        default_voice_id=settings.elevenlabs_default_voice_id,
        calm_voice_id=settings.elevenlabs_calm_voice_id,
    )

    # 2-3. Compose the system prompt: base + prosody suffix + verbosity + trusted facts.
    prompt_parts: list[str] = [settings.base_system_prompt]
    if profile.system_prompt_suffix:
        prompt_parts.append(profile.system_prompt_suffix)
    if user_ctx.conversation_style is ConversationStyle.IMPATIENT:
        prompt_parts.append(
            "User is highly impatient. Give answers in 10 words or less."
        )
    if user_ctx.trusted_facts:
        facts = "; ".join(user_ctx.trusted_facts[:5])
        prompt_parts.append(f"Known about this caller: {facts}.")
    system_prompt = " ".join(p.strip() for p in prompt_parts if p.strip())

    # 4. Proactive Empathy — if a severe negative event is on record, open with it.
    greeting = _proactive_greeting(user_ctx)

    log.info(
        "precall.built",
        user=pctx.user_id, tenant=pctx.tenant_id, room=pctx.room,
        emotion=emotion.value, prosody=profile.label,
        style=user_ctx.conversation_style.value, voice_id=voice_id,
        proactive=greeting is not None,
    )
    return PrecallResult(
        system_prompt=system_prompt,
        voice_id=voice_id,
        voice_settings=voice_settings,
        greeting=greeting,
        prosody_label=profile.label,
        used_conversation_style=user_ctx.conversation_style,
    )


def _proactive_greeting(user_ctx: UserGraphContext) -> str | None:
    """Build a natural-sounding opener that asks about the last severe negative event.

    The wording stays factual and warm — never melodramatic — so callers feel remembered
    without feeling surveilled. If no event is on record, return None and the agent falls
    back to a normal first turn.
    """
    ev = user_ctx.last_negative_event
    if ev is None or not ev.summary:
        return None
    summary = ev.summary.strip().rstrip(".")
    return (
        f"Hi again. Last time we spoke, {summary} — has that been sorted, "
        "or is there still something I can help with?"
    )
