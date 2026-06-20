"""Graph-engine pure logic — sanitizer, sarcasm floor, coercions.

These tests exercise the deterministic policy code that lives *around* the LLM/Neo4j
calls — the bits the demo actually depends on for reproducibility. The Neo4j integration
tests live in ``test_graph_engine_integration.py`` and skip unless a DB is reachable.
"""

from __future__ import annotations

from memory.graph_engine import (
    SARCASM_TRUST,
    UPDATE_DECAY_FACTOR,
    _apply_sarcasm_floor,
    _coerce_emotion,
    _coerce_style,
    _sanitize_rel_type,
)
from memory.schemas import ConversationStyle, Emotion

# ── relationship-type sanitizer ─────────────────────────────────────────────


def test_sanitize_rel_type_basic_predicates():
    assert _sanitize_rel_type("loves") == "LOVES"
    assert _sanitize_rel_type("hates") == "HATES"
    assert _sanitize_rel_type("works at") == "WORKS_AT"


def test_sanitize_rel_type_strips_punctuation_and_injection():
    # Cypher labels can't be parameterized — make sure injection can't slip through.
    assert _sanitize_rel_type("`); DROP NODES; //") == "DROP_NODES"
    assert _sanitize_rel_type("--evil--") == "EVIL"


def test_sanitize_rel_type_empty_falls_back():
    assert _sanitize_rel_type("") == "RELATES_TO"
    assert _sanitize_rel_type("___") == "RELATES_TO"


# ── sarcasm floor (the headline rule) ───────────────────────────────────────


def test_sarcasm_floor_forces_low_trust_on_positive_text_plus_agitation():
    ops = [{"action": "ADD", "subject": "User", "predicate": "loves",
            "object": "Job", "reasoning": "from transcript"}]
    out = _apply_sarcasm_floor(ops, acoustic_affect="agitated")
    assert out[0]["trust_score"] == SARCASM_TRUST
    assert "Likely Sarcastic" in out[0]["reasoning"]
    # The resolver's own reasoning is preserved alongside the override.
    assert "from transcript" in out[0]["reasoning"]


def test_sarcasm_floor_skips_neutral_voice():
    ops = [{"action": "ADD", "subject": "User", "predicate": "loves", "object": "Job"}]
    out = _apply_sarcasm_floor(ops, acoustic_affect="neutral")
    assert "trust_score" not in out[0]
    assert "Likely Sarcastic" not in out[0].get("reasoning", "")


def test_sarcasm_floor_skips_negative_text_even_if_agitated():
    # A negative fact + agitated voice is just genuine anger; trust shouldn't be capped.
    ops = [{"action": "ADD", "subject": "User", "predicate": "hates", "object": "Mother"}]
    out = _apply_sarcasm_floor(ops, acoustic_affect="agitated")
    assert "trust_score" not in out[0]


def test_sarcasm_floor_applies_to_updates_too():
    ops = [{"action": "UPDATE", "subject": "User", "predicate": "adores", "object": "X"}]
    out = _apply_sarcasm_floor(ops, acoustic_affect="agitated")
    assert out[0]["trust_score"] == SARCASM_TRUST


# ── coercions ──────────────────────────────────────────────────────────────


def test_coerce_emotion_handles_bad_values():
    assert _coerce_emotion("happy") is Emotion.HAPPY
    assert _coerce_emotion("HAPPY") is Emotion.HAPPY
    assert _coerce_emotion(None) is Emotion.NEUTRAL
    assert _coerce_emotion("not-a-real-label") is Emotion.NEUTRAL


def test_coerce_style_defaults_to_normal():
    assert _coerce_style("impatient") is ConversationStyle.IMPATIENT
    assert _coerce_style("") is ConversationStyle.NORMAL
    assert _coerce_style("bogus") is ConversationStyle.NORMAL


def test_decay_factor_is_70_percent():
    # The plan and the readme both promise 70% decay — guardrail against drift.
    assert UPDATE_DECAY_FACTOR == 0.3
