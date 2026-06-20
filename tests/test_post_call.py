"""Post-call orchestrator — verifies the end-to-end pipeline against a fake graph.

Asserts:
  * acoustic → fact-extract → apply_facts → record_affective_state are called in order
    with the right arguments,
  * interruption_count ≥ threshold flips ConversationStyle to IMPATIENT,
  * a transcript with a canonical negative-event hint becomes a ``NegativeEvent``,
  * a 'cynical_or_masking_grief' final state propagates to ``apply_facts``' acoustic_affect
    (which is what the sarcasm floor inside graph_engine keys off — covered by the
    integration tests).
"""

from __future__ import annotations

from typing import Any

import pytest

from config import Settings
from memory import post_call as pc
from memory.acoustic_engine import AcousticResult
from memory.schemas import (
    AffectiveState,
    ConversationStyle,
    Emotion,
    NegativeEvent,
    ParticipantContext,
)


class FakeGraph:
    """Records every call so the test can assert the orchestration order + payloads."""

    def __init__(self, triplets: list[dict[str, str]] | None = None) -> None:
        self._triplets = triplets or []
        self.extract_calls: list[str] = []
        self.apply_calls: list[dict[str, Any]] = []
        self.record_calls: list[tuple[AffectiveState, ConversationStyle, NegativeEvent | None]] = []

    def extract_facts_to_triplets(self, transcript: str) -> list[dict[str, str]]:
        self.extract_calls.append(transcript)
        return self._triplets

    async def apply_facts(self, *, user_id: str, new_triplets: list[dict[str, str]],
                          affective_state: str, acoustic_affect: str) -> list[dict[str, Any]]:
        call = {"user_id": user_id, "triplets": new_triplets,
                "affective_state": affective_state, "acoustic_affect": acoustic_affect}
        self.apply_calls.append(call)
        return [{"action": "ADD", "subject": t.get("subject"),
                 "predicate": t.get("predicate"), "object": t.get("object")}
                for t in new_triplets]

    async def record_affective_state(self, state: AffectiveState,
                                     conversation_style: ConversationStyle,
                                     last_negative_event: NegativeEvent | None = None) -> None:
        self.record_calls.append((state, conversation_style, last_negative_event))


def _pctx() -> ParticipantContext:
    return ParticipantContext(
        room="r1", participant_identity="lk-id-1", user_id="u1", tenant_id="t1",
    )


@pytest.fixture(autouse=True)
def stub_acoustic(monkeypatch):
    """Replace ``analyze_audio`` with a fixture-supplied AcousticResult per test."""
    holder = {"result": AcousticResult()}

    def _stub(_path, _settings):
        return holder["result"]

    monkeypatch.setattr(pc, "analyze_audio", _stub)
    return holder


# ── orchestration order ──────────────────────────────────────────────────────


async def test_pipeline_runs_in_the_documented_order(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        engine="librosa", acoustic_affect="neutral",
        base_acoustic_emotion="neutral", transcript="hello world",
    )
    graph = FakeGraph(triplets=[{"subject": "User", "predicate": "likes", "object": "Tea"}])
    summary = await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/fake.wav",
        transcript_turns=[{"role": "user", "content": "ignored — acoustic transcript wins"}],
        interruption_count=0, settings=Settings(),
    )
    # Order: extract over the rich transcript → apply_facts → record_affective_state.
    assert graph.extract_calls == ["hello world"]
    assert len(graph.apply_calls) == 1
    assert graph.apply_calls[0]["acoustic_affect"] == "neutral"
    assert summary["facts_extracted"] == 1
    assert summary["ops_applied"] == 1
    assert len(graph.record_calls) == 1


async def test_falls_back_to_livekit_transcript_when_acoustic_has_none():
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path=None,
        transcript_turns=[
            {"role": "user", "content": "my plan is enterprise"},
            {"role": "assistant", "content": "noted."},
        ],
        interruption_count=0, settings=Settings(),
    )
    assert any("my plan is enterprise" in t for t in graph.extract_calls)


# ── adaptive verbosity ───────────────────────────────────────────────────────


async def test_high_interruption_count_flips_to_impatient():
    graph = FakeGraph()
    settings = Settings(impatience_threshold=2)
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path=None,
        transcript_turns=[{"role": "user", "content": "hi"}],
        interruption_count=3, settings=settings,
    )
    state, style, _ = graph.record_calls[0]
    assert style is ConversationStyle.IMPATIENT
    assert state.features["interruption_count"] == 3.0


async def test_below_threshold_stays_normal():
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path=None,
        transcript_turns=[{"role": "user", "content": "hi"}],
        interruption_count=1, settings=Settings(impatience_threshold=2),
    )
    _, style, _ = graph.record_calls[0]
    assert style is ConversationStyle.NORMAL


# ── proactive empathy seed ───────────────────────────────────────────────────


async def test_canonical_flight_hint_in_transcript_becomes_negative_event(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        acoustic_affect="subdued", final_affective_state="frustrated",
        transcript="So my flight was cancelled the day of the meeting. It was awful.",
    )
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/x.wav",
        transcript_turns=[], interruption_count=0, settings=Settings(),
    )
    _, _, event = graph.record_calls[0]
    assert event is not None
    assert event.kind == "Flight"
    assert "flight was cancelled" in event.summary.lower()


async def test_grief_final_state_falls_back_to_first_sentence(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        acoustic_affect="subdued", final_affective_state="cynical_or_masking_grief",
        transcript="My mom died last spring. I'm still adjusting.",
    )
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/x.wav",
        transcript_turns=[], interruption_count=0, settings=Settings(),
    )
    _, _, event = graph.record_calls[0]
    assert event is not None
    assert event.kind in {"Loss", "Affect"}
    assert "mom died" in event.summary.lower() or "still adjusting" in event.summary.lower()


async def test_no_event_when_call_was_neutral(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        acoustic_affect="neutral", final_affective_state="",
        transcript="My plan is pro. Thanks for the update.",
    )
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/x.wav",
        transcript_turns=[], interruption_count=0, settings=Settings(),
    )
    _, _, event = graph.record_calls[0]
    assert event is None


# ── sarcasm signal threads through correctly ────────────────────────────────


async def test_agitated_voice_propagates_acoustic_affect_to_apply_facts(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        acoustic_affect="agitated", final_affective_state="cynical_or_masking_grief",
        transcript="I love waiting on hold.",
    )
    graph = FakeGraph(triplets=[{"subject": "User", "predicate": "loves", "object": "Waiting"}])
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/x.wav",
        transcript_turns=[], interruption_count=0, settings=Settings(),
    )
    # The graph-engine sarcasm floor is what actually pins trust=0.2; the orchestrator's
    # job is to make sure the agitated signal gets there.
    assert graph.apply_calls[0]["acoustic_affect"] == "agitated"
    assert graph.apply_calls[0]["affective_state"] == "cynical_or_masking_grief"


# ── coarse emotion mapping ──────────────────────────────────────────────────


async def test_final_state_grief_becomes_sad_emotion(stub_acoustic):
    stub_acoustic["result"] = AcousticResult(
        final_affective_state="cynical_or_masking_grief", transcript="empty",
    )
    graph = FakeGraph()
    await pc.process_post_call(
        graph=graph, pctx=_pctx(), audio_path="/tmp/x.wav",
        transcript_turns=[], interruption_count=0, settings=Settings(),
    )
    state, _, _ = graph.record_calls[0]
    assert state.emotion is Emotion.SAD
    # Valence should reflect SAD's base (negative-ish).
    assert state.valence < 0
