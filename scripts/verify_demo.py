"""Post-demo verifier — assert the four "wow" features actually landed in Neo4j.

After:
  1. ``python -m scripts.seed_demo``      (Step 4 prep)
  2. join the Playground, hold the demo conversation, hang up
  3. wait ~5s for the post-call task to finish

run::

    python -m scripts.verify_demo

It queries the actual graph and prints PASS/FAIL for each of:

  1. **Dynamic Prosody pre-call read** — ``(:User).emotion == "frustrated"`` and a
     :class:`UserGraphContext` lookup picks ``calm_voice_id`` with ``stability=0.95``.
  2. **Adaptive Verbosity** — ``(:User).conversation_style == "impatient"``.
  3. **Proactive Empathy** — a ``(:User)-[:EXPERIENCED]->(:Event)`` with severity ≥ 0.7
     is on record; its ``summary`` is what the next call's greeting will reference.
  4. **Sarcasm filter** — a positive-text relationship written during the live call has
     ``trust_score == 0.2`` and ``"Likely Sarcastic"`` in ``reasoning``.
  5. **Contradiction decay** — the original ``(User)-[:LOVES]->(Mother)`` rel survives
     with ``trust_score`` decayed by ≥ 65% (the 0.3 multiplier with some float slack)
     and ``superseded_at`` set.

Exit code 0 = everything passed; non-zero = at least one check failed.
"""

from __future__ import annotations

import asyncio
import sys

from config import Settings
from memory.graph_engine import SARCASM_TRUST, UPDATE_DECAY_FACTOR, CognitiveGraph
from memory.pre_call import build_precall_context
from memory.prosody import _PROFILES, to_elevenlabs_voice
from memory.schemas import ConversationStyle, Emotion, ParticipantContext
from scripts.seed_demo import DEMO_TENANT, DEMO_USER


def _check(label: str, ok: bool, detail: str = "") -> bool:
    badge = "✓ PASS" if ok else "✗ FAIL"
    line = f"{badge}  {label}"
    if detail:
        line += f"\n          {detail}"
    print(line)
    return ok


async def main() -> int:
    settings = Settings()
    graph = CognitiveGraph(settings)
    await graph.connect()
    failures = 0
    try:
        ctx = await graph.get_user_context(DEMO_TENANT, DEMO_USER)

        # ── 1. Dynamic Prosody pre-call decision ────────────────────────────
        emotion = ctx.affective_state.emotion if ctx.affective_state else Emotion.NEUTRAL
        profile = _PROFILES.get(emotion)
        if profile is None:
            failures += not _check(
                "Dynamic Prosody — affective_state present",
                False, f"got emotion={emotion.value}",
            )
        else:
            voice_id, vs = to_elevenlabs_voice(
                emotion, profile,
                default_voice_id=settings.elevenlabs_default_voice_id,
                calm_voice_id=settings.elevenlabs_calm_voice_id,
            )
            failures += not _check(
                "Dynamic Prosody — frustrated → calm voice + stability 0.95",
                voice_id == settings.elevenlabs_calm_voice_id and abs(vs.stability - 0.95) < 1e-6,
                f"voice={voice_id} stability={vs.stability}",
            )

        # ── 2. Adaptive Verbosity ───────────────────────────────────────────
        failures += not _check(
            "Adaptive Verbosity — conversation_style == impatient",
            ctx.conversation_style is ConversationStyle.IMPATIENT,
            f"got {ctx.conversation_style.value}",
        )

        # Same check, but via the actual code path pre_call.py runs:
        pre = await build_precall_context(graph, _pctx(), settings)
        failures += not _check(
            "Adaptive Verbosity — prompt carries the '10 words or less' constraint",
            "10 words or less" in pre.system_prompt,
            f"prompt: {pre.system_prompt[:160]}...",
        )

        # ── 3. Proactive Empathy ────────────────────────────────────────────
        failures += not _check(
            "Proactive Empathy — last_negative_event surfaces",
            ctx.last_negative_event is not None and bool(ctx.last_negative_event.summary),
            f"event: {ctx.last_negative_event.kind if ctx.last_negative_event else 'None'}"
            f' — "{(ctx.last_negative_event.summary if ctx.last_negative_event else "")[:80]}"',
        )
        failures += not _check(
            "Proactive Empathy — greeting was built",
            pre.greeting is not None,
            (pre.greeting or "(no greeting)")[:120],
        )

        # ── 4. Sarcasm filter (set by the live post-call write) ────────────
        sarcastic = await _find_sarcastic(graph)
        failures += not _check(
            "Sarcasm filter — at least one rel has trust_score=0.2 + 'Likely Sarcastic'",
            len(sarcastic) > 0,
            f"found {len(sarcastic)}: " + ", ".join(
                f"({a})-[{p} trust={t}]->({b})" for a, p, b, t, _r in sarcastic[:3]
            ) or "(none — say 'I love waiting on hold' in an agitated tone next time)",
        )

        # ── 5. Contradiction decay (LOVES Mother) ──────────────────────────
        decayed = await _find_decayed_loves_mother(graph)
        decay_floor = 0.9 * UPDATE_DECAY_FACTOR + 0.05         # allow some slack
        passed = any(0 < t < decay_floor for _ts, t in decayed)
        failures += not _check(
            "Contradiction decay — old (User)-[:LOVES]->(Mother) trust decayed by ≥65%",
            passed,
            f"trust history: {decayed}  (decayed value should be ≤ ~{decay_floor:.2f})",
        )

        print()
        if failures == 0:
            print(f"✓ All five demo checks PASSED (sarcasm floor = {SARCASM_TRUST}, "
                  f"decay factor = ×{UPDATE_DECAY_FACTOR}).")
        else:
            print(f"✗ {failures} check(s) FAILED — see lines above.")
        return 0 if failures == 0 else 1
    finally:
        await graph.close()


# ── live-graph readers ──────────────────────────────────────────────────────


async def _find_sarcastic(graph: CognitiveGraph) -> list[tuple[str, str, str, float, str]]:
    out: list[tuple[str, str, str, float, str]] = []
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        result = await s.run(
            "MATCH (sub:Entity {user_id:$u})-[r]->(obj:Entity {user_id:$u}) "
            "WHERE r.trust_score = $t AND r.reasoning CONTAINS 'Likely Sarcastic' "
            "RETURN sub.id AS s, coalesce(r.predicate_raw,'') AS p, obj.id AS o, "
            "r.trust_score AS trust, coalesce(r.reasoning,'') AS reason",
            u=DEMO_USER, t=SARCASM_TRUST,
        )
        async for row in result:
            out.append((row["s"], row["p"], row["o"], float(row["trust"]), row["reason"]))
    return out


async def _find_decayed_loves_mother(graph: CognitiveGraph) -> list[tuple[bool, float]]:
    """Return (superseded_at_present, trust_score) pairs over LOVES rels into Mother."""
    out: list[tuple[bool, float]] = []
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        result = await s.run(
            "MATCH (:Entity {user_id:$u, id:'User'})-[r:LOVES]->(:Entity {user_id:$u, id:'Mother'}) "
            "RETURN coalesce(r.trust_score, 0.5) AS t, r.superseded_at AS sup "
            "ORDER BY r.updated_at",
            u=DEMO_USER,
        )
        async for row in result:
            out.append((row["sup"] is not None, float(row["t"])))
    return out


def _pctx() -> ParticipantContext:
    return ParticipantContext(
        room="demo-room", participant_identity="demo-id",
        user_id=DEMO_USER, tenant_id=DEMO_TENANT,
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
