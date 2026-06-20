"""End-to-end Neo4j integration — runs only when a local DB answers on bolt://.

These tests exercise the actual Cypher: per-user entity scoping, ``apply_facts``
contradiction decay, ``record_affective_state`` user-property writes, and the
``get_user_context`` pre-call read shape. Each test seeds and tears down its own
slice so they're independent and re-runnable.

Skip strategy: try ``verify_connectivity()``; if it fails we skip the whole module.
That keeps the suite green without Docker while still catching real Cypher errors when
``docker compose up`` is running.
"""

from __future__ import annotations

import contextlib
import importlib.util

import pytest
import pytest_asyncio

if importlib.util.find_spec("neo4j") is None:
    pytest.skip("neo4j driver not installed", allow_module_level=True)

from neo4j import AsyncGraphDatabase  # noqa: E402

from config import Settings  # noqa: E402
from memory.graph_engine import (  # noqa: E402
    SARCASM_TRUST,
    UPDATE_DECAY_FACTOR,
    CognitiveGraph,
)
from memory.schemas import (  # noqa: E402
    AffectiveState,
    ConversationStyle,
    Emotion,
    NegativeEvent,
)

_NEO4J_ALIVE: bool | None = None


async def _neo4j_alive(settings: Settings) -> bool:
    """One quick probe per test session. Cached so each test skips fast when DB is down."""
    global _NEO4J_ALIVE
    if _NEO4J_ALIVE is not None:
        return _NEO4J_ALIVE
    try:
        d = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=2.0,
        )
        try:
            await d.verify_connectivity()
            _NEO4J_ALIVE = True
        finally:
            await d.close()
    except Exception:
        _NEO4J_ALIVE = False
    return _NEO4J_ALIVE


@pytest_asyncio.fixture
async def graph():
    settings = Settings()
    if not await _neo4j_alive(settings):
        pytest.skip("Neo4j unreachable on bolt:// — run `docker compose up neo4j` to enable.")
    g = CognitiveGraph(settings)
    await g.connect()
    # Clean slate for this test's tenant/user combo.
    test_user = "test-user-int"
    test_tenant = "test-tenant"
    async with g._driver_or_raise().session() as s:   # noqa: SLF001
        await s.run(
            "MATCH (e:Entity {user_id:$u}) DETACH DELETE e", u=test_user
        )
        await s.run(
            "MATCH (u:User {tenant_id:$t, id:$u}) DETACH DELETE u",
            t=test_tenant, u=test_user,
        )
    try:
        yield g, test_tenant, test_user
    finally:
        with contextlib.suppress(Exception):
            async with g._driver_or_raise().session() as s:   # noqa: SLF001
                await s.run("MATCH (e:Entity {user_id:$u}) DETACH DELETE e", u=test_user)
                await s.run(
                    "MATCH (u:User {tenant_id:$t, id:$u}) DETACH DELETE u",
                    t=test_tenant, u=test_user,
                )
        await g.close()


# ── tests ───────────────────────────────────────────────────────────────────


async def test_record_affective_state_and_read_back(graph):
    g, tenant, user = graph
    state = AffectiveState(
        tenant_id=tenant, user_id=user, emotion=Emotion.FRUSTRATED,
        valence=-0.6, arousal=0.7, confidence=0.9,
        paralinguistics={"final_affective_state": "tense_suppressed"},
    )
    await g.record_affective_state(state, ConversationStyle.IMPATIENT)
    ctx = await g.get_user_context(tenant, user)
    assert ctx.affective_state is not None
    assert ctx.affective_state.emotion is Emotion.FRUSTRATED
    assert ctx.conversation_style is ConversationStyle.IMPATIENT
    assert ctx.affective_state.paralinguistics["final_affective_state"] == "tense_suppressed"


async def test_negative_event_surfaces_via_pre_call_read(graph):
    g, tenant, user = graph
    state = AffectiveState(tenant_id=tenant, user_id=user, emotion=Emotion.SAD)
    event = NegativeEvent(
        kind="Flight_Cancellation",
        summary="your flight to Bangalore was cancelled the day of",
        emotion=Emotion.SAD,
    )
    await g.record_affective_state(state, ConversationStyle.NORMAL, last_negative_event=event)
    ctx = await g.get_user_context(tenant, user)
    assert ctx.last_negative_event is not None
    assert "bangalore" in ctx.last_negative_event.summary.lower()


async def test_apply_facts_writes_relationship_and_get_user_context_surfaces_it(graph):
    g, tenant, user = graph
    # Bypass the OpenAI resolver by injecting operations directly through the commit path.
    ops = [{"action": "ADD", "subject": "User", "predicate": "loves",
            "object": "Tea", "trust_score": 0.9, "reasoning": "explicit"}]
    await g._commit_operations(user, ops, affective_state="neutral")   # noqa: SLF001
    ctx = await g.get_user_context(tenant, user)
    assert any("Tea" in fact for fact in ctx.trusted_facts)


async def test_update_decays_old_relationship_by_70_percent(graph):
    g, _tenant, user = graph
    # Seed a high-trust love-of-Mother.
    await g._commit_operations(   # noqa: SLF001
        user,
        [{"action": "ADD", "subject": "User", "predicate": "loves",
          "object": "Mother", "trust_score": 0.9, "reasoning": "initial"}],
        affective_state="neutral",
    )
    # Now UPDATE: a contradicting fact arrives.
    await g._commit_operations(   # noqa: SLF001
        user,
        [{"action": "UPDATE", "subject": "User", "predicate": "loves",
          "object": "Mother", "trust_score": 0.7, "reasoning": "reversal"}],
        affective_state="cynical_or_masking_grief",
    )
    async with g._driver_or_raise().session() as s:   # noqa: SLF001
        rows = await (await s.run(
            "MATCH (:Entity {user_id:$u, id:'User'})-[r:LOVES]->(:Entity {user_id:$u, id:'Mother'}) "
            "RETURN r.trust_score AS t, r.superseded_at AS sup "
            "ORDER BY r.updated_at DESC",
            u=user,
        )).values()
    trust_scores = [float(r[0]) for r in rows]
    # Expect two rows: the new (live, trust=0.7) and the old (superseded, decayed).
    assert any(abs(t - 0.7) < 1e-6 for t in trust_scores)
    decayed = [t for t in trust_scores if abs(t - 0.9 * UPDATE_DECAY_FACTOR) < 1e-6]
    assert decayed, f"expected a decayed (~0.27) rel; got trust scores {trust_scores}"


async def test_sarcasm_floor_lowers_trust_when_voice_was_agitated(graph):
    g, tenant, user = graph
    # Simulate a positive triplet riding agitated tone — apply_facts should pin it to 0.2.
    triplets = [{"subject": "User", "predicate": "loves", "object": "Job"}]
    # Bypass LLM by patching _resolve_with_llm to echo the triplets as ADDs.
    g._resolve_with_llm = lambda existing, new, affect: [   # type: ignore[method-assign]
        {"action": "ADD", "subject": t["subject"], "predicate": t["predicate"],
         "object": t["object"], "reasoning": "test"} for t in new
    ]
    committed = await g.apply_facts(
        user, triplets, affective_state="cynical_or_masking_grief", acoustic_affect="agitated",
    )
    assert committed[0]["trust_score"] == SARCASM_TRUST
    # Read it back from the graph too.
    ctx = await g.get_user_context(tenant, user)
    # The fact's trust (0.2) is below the 0.6 cutoff, so it should NOT appear in trusted_facts.
    assert all("Job" not in fact for fact in ctx.trusted_facts)
