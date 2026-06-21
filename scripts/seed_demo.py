"""Seed Neo4j for the cognitive-agent demo.

After ``docker compose up neo4j``, run::

    python -m scripts.seed_demo

This primes the graph so when a caller joins with metadata
``{"user_id":"demo-u1","tenant_id":"demo-t1"}``, the **next** call's pre-call read
triggers the three pre-call wow features in one go:

  • Frustrated affective_state ............ → Dynamic Prosody: calm voice + stability 0.95
  • conversation_style="impatient" ......... → Adaptive Verbosity: "10 words or less"
  • A severe negative (:Event) ............. → Proactive Empathy: agent opens by asking
                                              "Last time your flight to Bangalore was
                                              cancelled — has that been sorted?"

It also seeds two ``(:Entity)-[:LOVES]->(:Entity)`` relationships you can target with the
**post-call** demo: hang up after saying "I love waiting on hold" in an agitated tone, and
the sarcasm floor should pin trust to 0.2; say "actually I hate my mother" in a normal
tone and the contradiction-decay path should leave the old LOVES rel with ``trust_score``
multiplied by 0.3 (≈ 0.27).

Side effects:
  * Wipes the (:User {tenant_id:'demo-t1', id:'demo-u1'}) and its facts before seeding,
    so the script is idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from config import Settings
from memory.graph_engine import CognitiveGraph
from memory.schemas import (
    AffectiveState,
    ConversationStyle,
    Emotion,
    NegativeEvent,
)

DEMO_USER = "demo-anil"     # matches the web UI's persona key so the two seeds agree
DEMO_TENANT = "demo-t1"


async def seed(reset: bool = True) -> None:
    settings = Settings()
    graph = CognitiveGraph(settings)
    await graph.connect()
    try:
        if reset:
            await _wipe(graph)

        # 1) Affective state + conversation style on the (:User) node.
        state = AffectiveState(
            tenant_id=DEMO_TENANT, user_id=DEMO_USER,
            emotion=Emotion.FRUSTRATED,
            valence=-0.6, arousal=0.75, confidence=0.85,
            features={"pitch_variance": 28.0, "rms_energy": 0.06},
            paralinguistics={
                "base_acoustic_emotion": "angry",
                "final_affective_state": "tense_suppressed",
                "lexical_sentiment": "negative",
                "acoustic_affect": "agitated",
                "engine": "seed",
            },
        )
        event = NegativeEvent(
            kind="ServiceOutage",
            summary=("your service outage three weeks ago cost BharatPay an estimated "
                     "₹2 crore in missed loan-collection touches"),
            emotion=Emotion.FRUSTRATED,
            ts=time.time(),
        )
        await graph.record_affective_state(state, ConversationStyle.IMPATIENT,
                                           last_negative_event=event)

        # 2) Seed the BharatPay business storyline (Anil Mehta, COO, at-risk enterprise).
        #    See web/server.py:_seed_anil_bharatpay for the full narrative.
        await graph._commit_operations(   # noqa: SLF001 — direct write, no LLM needed for seed
            DEMO_USER,
            [
                {"action": "ADD", "subject": "Anil", "predicate": "is COO of",
                 "object": "BharatPay", "trust_score": 0.95,
                 "reasoning": "seed: stable employment fact"},
                {"action": "ADD", "subject": "BharatPay", "predicate": "uses",
                 "object": "Enterprise_Plan", "trust_score": 0.9,
                 "reasoning": "seed: contract on record"},
                {"action": "ADD", "subject": "Anil", "predicate": "works with",
                 "object": "CSM_Priya", "trust_score": 0.85,
                 "reasoning": "seed: named dedicated CSM after escalation"},
                {"action": "ADD", "subject": "Anil", "predicate": "is owed",
                 "object": "Service_Credits_Q3", "trust_score": 0.7,
                 "reasoning": "seed: promised after the outage; not yet posted"},
                {"action": "ADD", "subject": "Anil", "predicate": "is considering",
                 "object": "Competitor_AlphaVoice", "trust_score": 0.6,
                 "reasoning": "seed: competitive intel; churn risk"},
            ],
            affective_state="tense_suppressed",
        )

        print("✓ Seeded Neo4j for the demo.")
        print(f"  user_id   = {DEMO_USER}")
        print(f"  tenant_id = {DEMO_TENANT}")
        print()
        print("Join the LiveKit Playground with participant metadata:")
        print(f'  {{"user_id":"{DEMO_USER}","tenant_id":"{DEMO_TENANT}"}}')
        print()
        print("Expected: the agent opens with the empathy line in a calm voice")
        print("and gives short replies (impatient style).")
        print()
        print("Browse the graph:  http://localhost:7474/browser/")
        print(f'  MATCH (u:User {{tenant_id:"{DEMO_TENANT}", id:"{DEMO_USER}"}})'
              f'-[r]->(n) RETURN u, r, n')
    finally:
        await graph.close()


async def _wipe(graph: CognitiveGraph) -> None:
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        await s.run("MATCH (e:Entity {user_id:$u}) DETACH DELETE e", u=DEMO_USER)
        await s.run("MATCH (u:User {tenant_id:$t, id:$u}) DETACH DELETE u",
                    t=DEMO_TENANT, u=DEMO_USER)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-reset", action="store_true",
                   help="Append to existing state instead of wiping it first.")
    args = p.parse_args()
    asyncio.run(seed(reset=not args.no_reset))


if __name__ == "__main__":
    main()
