"""Data models for the cognitive memory layer.

The bottom six types (Emotion, AffectiveState, MemoryOp, Assertion, ContradictionDecision)
are ported verbatim from the archived prototype's affective-memory module. The remainder is
new for the LiveKit / Neo4j pivot:

- ``ConversationStyle``  — Adaptive Verbosity signal stored per-user.
- ``ParticipantContext`` — what the agent learns about a caller from LiveKit metadata.
- ``NegativeEvent``      — the "Flight cancellation, etc." event Proactive Empathy reads.
- ``UserGraphContext``   — the pre-call read from Neo4j (state + style + facts + event).
- ``ExtractedFact``      — what the post-call LLM extraction emits per assertion.
- ``PrecallResult``      — what pre_call hands to ``agent.py`` (system prompt, voice, greeting).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Affective primitives (ported) ────────────────────────────────────────────


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"


class ConversationStyle(StrEnum):
    """How the user prefers to be spoken to. Drives Adaptive Verbosity in pre-call."""

    NORMAL = "normal"
    IMPATIENT = "impatient"


class AffectiveState(BaseModel):
    """The caller's emotional state — written post-call, read pre-call.

    ``emotion`` is the coarse label that drives pre-call prosody. ``paralinguistics``
    carries the rich profile from the SenseVoice+openSMILE engine when available
    (acoustic emotion, lexical sentiment, the LLM's reconciled ``final_affective_state``,
    detected audio events, eGeMAPS biometrics) — a superset that a future, more
    nuanced pre-call policy can use.
    """

    tenant_id: str
    user_id: str
    emotion: Emotion = Emotion.NEUTRAL
    valence: float = 0.0        # -1 (negative) .. +1 (positive)
    arousal: float = 0.0        #  0 (calm)     ..  1 (activated)
    confidence: float = 0.0
    features: dict[str, Any] = Field(default_factory=dict)
    paralinguistics: dict[str, Any] = Field(default_factory=dict)


class MemoryOp(StrEnum):
    """The four operations the contradiction engine chooses between."""

    ADD = "ADD"        # genuinely new fact
    UPDATE = "UPDATE"  # supersedes an existing, now-stale fact
    DELETE = "DELETE"  # existing fact is contradicted/retracted
    NOOP = "NOOP"      # already known; nothing to do


class Assertion(BaseModel):
    """A candidate fact extracted from a transcript turn (regex pass)."""

    text: str
    subject: str = ""       # normalized key, e.g. "address"; "" means free-form
    value: str = ""
    negated: bool = False
    evidence: str = ""


class ContradictionDecision(BaseModel):
    op: MemoryOp
    new_fact: str
    superseded: str | None = None
    reasoning: str = ""


# ── LiveKit-side context ─────────────────────────────────────────────────────


class ParticipantContext(BaseModel):
    """What we learn from the LiveKit participant at connect time.

    LiveKit gives us ``participant.identity`` (string) and ``participant.metadata``
    (free-form string — clients typically pack JSON). The agent expects:
      {"user_id": "u123", "tenant_id": "acme"}
    Fallbacks: ``user_id`` defaults to ``participant.identity``; ``tenant_id`` to "default".
    """

    room: str
    participant_identity: str
    user_id: str
    tenant_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Graph-side context (Neo4j read shape) ────────────────────────────────────


class NegativeEvent(BaseModel):
    """A severe negative event captured during a prior call — drives Proactive Empathy.

    Stored as ``(:User)-[:EXPERIENCED]->(:Event {kind, summary, emotion, ts})``.
    """

    kind: str                   # e.g. "Flight_Cancellation"
    summary: str                # e.g. "Flight to Bangalore cancelled the day of"
    emotion: Emotion = Emotion.SAD
    ts: float = 0.0             # epoch seconds; 0 = unknown


class UserGraphContext(BaseModel):
    """Result of ``CognitiveGraph.get_user_context`` — everything pre-call needs."""

    affective_state: AffectiveState | None = None
    conversation_style: ConversationStyle = ConversationStyle.NORMAL
    trusted_facts: list[str] = Field(default_factory=list)   # ≤ N short summaries
    last_negative_event: NegativeEvent | None = None


# ── Post-call extraction shape ──────────────────────────────────────────────


class ExtractedFact(BaseModel):
    """One fact emitted by the LLM extraction pass over the call transcript.

    ``entity`` is the *canonical* form ("mom" → "Mother") so the graph dedupes cleanly.
    ``text_sentiment`` is what the LLM read from the WORDS only; the post-call worker
    cross-references it against the acoustic engine's ``acoustic_affect`` to fire the
    Sarcasm & Truth filter.
    """

    subject: str                                # short normalized key, e.g. "plan"
    value: str                                  # the asserted value, e.g. "enterprise"
    entity: str = ""                            # canonical entity name (optional)
    entity_type: str = ""                       # Person / Company / Product / ...
    text_sentiment: str = "neutral"             # positive | negative | neutral
    negated: bool = False
    evidence: str = ""                          # original transcript span


# ── Pre-call output ─────────────────────────────────────────────────────────


class VoiceSettings(BaseModel):
    """ElevenLabs-style prosody knobs. We keep them in our own schema so the rest of the
    code never has to import the ElevenLabs SDK; ``prosody.to_elevenlabs_voice`` does the
    final translation right before ``AgentSession`` is built."""

    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.2
    speed: float = 1.0
    use_speaker_boost: bool = True


class PrecallResult(BaseModel):
    """Everything ``build_precall_context`` hands back to the LiveKit entrypoint."""

    system_prompt: str
    voice_id: str
    voice_settings: VoiceSettings
    greeting: str | None = None                 # set when Proactive Empathy fires
    prosody_label: str = "neutral"              # for logging/observability
    used_conversation_style: ConversationStyle = ConversationStyle.NORMAL
