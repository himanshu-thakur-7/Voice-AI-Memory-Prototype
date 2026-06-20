"""Data models for the affective memory layer (Module 3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"


class AffectiveState(BaseModel):
    """The caller's emotional state — written post-call, read pre-call.

    ``emotion`` is the coarse label that drives pre-call prosody. ``paralinguistics``
    carries the rich profile from the SenseVoice+openSMILE engine when available
    (acoustic emotion, lexical sentiment, the LLM's reconciled ``final_affective_state``,
    detected audio events, and eGeMAPS biometrics) — a superset that a future, more
    nuanced pre-call policy can use.
    """

    tenant_id: str
    user_id: str
    emotion: Emotion = Emotion.NEUTRAL
    valence: float = 0.0   # -1 (negative) .. +1 (positive)
    arousal: float = 0.0   #  0 (calm)     ..  1 (activated)
    confidence: float = 0.0
    features: dict[str, Any] = Field(default_factory=dict)
    #: Rich post-call profile (mirrors the extractor's `memory_payload["paralinguistics"]`).
    paralinguistics: dict[str, Any] = Field(default_factory=dict)


class MemoryOp(StrEnum):
    """The four operations Mem0's resolver chooses between — this IS the contradiction
    engine's vocabulary."""

    ADD = "ADD"        # genuinely new fact
    UPDATE = "UPDATE"  # supersedes an existing, now-stale fact
    DELETE = "DELETE"  # existing fact is contradicted/retracted
    NOOP = "NOOP"      # already known; nothing to do


class Assertion(BaseModel):
    """A candidate fact extracted from the call transcript."""

    text: str
    subject: str = ""    # normalized key, e.g. "address"; "" means free-form
    value: str = ""
    negated: bool = False
    evidence: str = ""


class ContradictionDecision(BaseModel):
    op: MemoryOp
    new_fact: str
    superseded: str | None = None
    reasoning: str = ""
