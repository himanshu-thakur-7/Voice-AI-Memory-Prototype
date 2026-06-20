"""Pre-call orchestration — Dynamic Prosody + Adaptive Verbosity + Proactive Empathy."""

from __future__ import annotations

from config import Settings
from memory.pre_call import NullGraph, build_precall_context
from memory.schemas import (
    AffectiveState,
    ConversationStyle,
    Emotion,
    NegativeEvent,
    ParticipantContext,
    UserGraphContext,
)


class StubGraph:
    def __init__(self, ctx: UserGraphContext) -> None:
        self._ctx = ctx

    async def get_user_context(self, tenant_id: str, user_id: str) -> UserGraphContext:
        return self._ctx


def _pctx() -> ParticipantContext:
    return ParticipantContext(
        room="r1", participant_identity="lk-id-1",
        user_id="u1", tenant_id="t1",
    )


async def test_null_graph_yields_neutral_defaults() -> None:
    pre = await build_precall_context(NullGraph(), _pctx(), Settings())
    assert pre.prosody_label == "neutral"
    assert pre.greeting is None
    assert "10 words or less" not in pre.system_prompt
    assert pre.voice_id == Settings().elevenlabs_default_voice_id


async def test_frustrated_user_triggers_calm_voice_and_empathy_suffix() -> None:
    graph = StubGraph(UserGraphContext(
        affective_state=AffectiveState(
            tenant_id="t1", user_id="u1", emotion=Emotion.FRUSTRATED,
        ),
    ))
    pre = await build_precall_context(graph, _pctx(), Settings())
    assert pre.prosody_label == "empathetic-slow"
    assert pre.voice_id == Settings().elevenlabs_calm_voice_id
    assert pre.voice_settings.stability == 0.95
    assert "frustrated" in pre.system_prompt.lower()


async def test_impatient_style_injects_verbosity_constraint() -> None:
    graph = StubGraph(UserGraphContext(
        conversation_style=ConversationStyle.IMPATIENT,
    ))
    pre = await build_precall_context(graph, _pctx(), Settings())
    assert "10 words or less" in pre.system_prompt
    assert pre.used_conversation_style is ConversationStyle.IMPATIENT


async def test_severe_negative_event_produces_proactive_greeting() -> None:
    graph = StubGraph(UserGraphContext(
        last_negative_event=NegativeEvent(
            kind="Flight_Cancellation",
            summary="your flight to Bangalore was cancelled the day of",
            emotion=Emotion.SAD,
        ),
    ))
    pre = await build_precall_context(graph, _pctx(), Settings())
    assert pre.greeting is not None
    assert "flight to bangalore" in pre.greeting.lower()
    # No melodrama — wording stays warm-and-factual.
    assert "i'm so sorry" not in pre.greeting.lower()


async def test_trusted_facts_inserted_into_prompt() -> None:
    graph = StubGraph(UserGraphContext(
        trusted_facts=["plan is enterprise", "primary contact is Mother"],
    ))
    pre = await build_precall_context(graph, _pctx(), Settings())
    assert "plan is enterprise" in pre.system_prompt
    assert "primary contact is Mother" in pre.system_prompt
