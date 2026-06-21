"""Demo-grade web service for the Cognitive Voice AI Agent.

What it does:
  • ``POST /api/token``  — mint a LiveKit JWT for a chosen persona. ``user_id`` and
    ``tenant_id`` are packed into ``participant.metadata`` exactly the way ``agent.py``
    expects them, so connecting with this token from a browser is enough to drive every
    feature end-to-end.
  • ``POST /api/seed``   — seed the cognitive graph for a built-in persona before joining.
  • ``GET  /api/verify`` — post-call read of the graph: which of the four wow features
    landed and what trust scores / events are on record.
  • Serves the single-page UI at ``/``.

This is *demo* infrastructure. It runs as its own process — the LiveKit Agent worker
(``python agent.py dev``) is a separate process that dials out to LiveKit Cloud, and the
browser dials in to the same room. They never talk to each other directly; the graph and
the LiveKit room are the only shared state.

Run with::

    uvicorn web.server:app --host 0.0.0.0 --port 8000

For deployment notes see the README.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from livekit import api
from pydantic import BaseModel, Field

from config import get_settings
from memory.graph_engine import CognitiveGraph
from memory.pre_call import build_precall_context
from memory.prosody import _PROFILES, to_elevenlabs_voice
from memory.schemas import ConversationStyle, Emotion, ParticipantContext

log = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Cognitive Voice AI Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo; tighten when you have a real production hostname
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── personas: canned demo identities the UI can choose from ─────────────────


PERSONAS: dict[str, dict] = {
    "anil": {
        "label": "Anil Mehta — COO @ BharatPay Fintech (at-risk enterprise account)",
        "tenant_id": "demo-t1",
        "user_id": "demo-anil",
        "blurb": (
            "B2B churn risk. Three weeks ago a partial outage cost BharatPay ~₹2 cr in "
            "missed loan-collection touches. Service credits still pending; he's "
            "shopping a competitor. Impatient style, tense voice. Seeding triggers all "
            "three pre-call features and lines up the contradiction-decay + sarcasm "
            "post-call demos."
        ),
        "seed": "anil_bharatpay",
    },
    "first_time": {
        "label": "First-time caller — neutral, no graph state",
        "tenant_id": "demo-t1",
        "user_id": "demo-fresh",
        "blurb": "No history. Agent behaves as a stock voice assistant — control case "
                 "so you can A/B the personalization.",
        "seed": "none",
    },
}


# ── request / response models ───────────────────────────────────────────────


class TokenRequest(BaseModel):
    persona: str | None = None
    room: str = "cognitive-demo"
    user_id: str | None = None      # only used when persona is None
    tenant_id: str | None = None
    display_name: str | None = None


class TokenResponse(BaseModel):
    url: str
    token: str
    identity: str
    metadata: dict
    persona_label: str | None = None


class SeedRequest(BaseModel):
    persona: str


class VerifyResponse(BaseModel):
    available: bool
    user_id: str
    tenant_id: str
    affective_state: dict | None = None
    conversation_style: str = "normal"
    trusted_facts: list[str] = Field(default_factory=list)
    last_negative_event: dict | None = None
    precall_voice_id: str | None = None
    precall_voice_stability: float | None = None
    precall_greeting: str | None = None
    precall_system_prompt: str | None = None
    sarcastic_facts: list[dict] = Field(default_factory=list)
    decayed_facts: list[dict] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)


# ── routes ──────────────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "ts": time.time()}


@app.get("/api/personas")
async def list_personas() -> dict:
    """Surface the persona catalog so the UI can render without hard-coding."""
    return {"personas": [{"key": k, **{kk: vv for kk, vv in v.items() if kk != "seed"}}
                         for k, v in PERSONAS.items()]}


@app.post("/api/token", response_model=TokenResponse)
async def mint_token(req: TokenRequest) -> TokenResponse:
    """Mint a LiveKit JWT carrying the metadata our agent expects.

    The token's participant ``metadata`` field is a JSON blob with ``user_id`` and
    ``tenant_id`` — that's exactly what ``agent.parse_participant_context`` reads when the
    participant connects. Anything else in metadata is preserved (so a UI can add e.g. a
    display name without changing the worker).
    """
    settings = get_settings()
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(500, "LIVEKIT_API_KEY/SECRET not configured on the server")

    if req.persona:
        if req.persona not in PERSONAS:
            raise HTTPException(400, f"unknown persona '{req.persona}'")
        p = PERSONAS[req.persona]
        tenant_id = p["tenant_id"]
        user_id = p["user_id"]
        persona_label = p["label"]
    else:
        if not req.user_id:
            raise HTTPException(400, "user_id required when persona is unset")
        tenant_id = req.tenant_id or "demo-t1"
        user_id = req.user_id
        persona_label = None

    metadata = {"user_id": user_id, "tenant_id": tenant_id}
    if req.display_name:
        metadata["display_name"] = req.display_name

    identity = f"web-{user_id}-{int(time.time()) % 1_000_000}"
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(req.display_name or user_id)
        .with_metadata(json.dumps(metadata))
        .with_grants(api.VideoGrants(
            room_join=True, room=req.room,
            can_publish=True, can_subscribe=True,
        ))
        .with_ttl(_TOKEN_TTL)
        .to_jwt()
    )
    return TokenResponse(
        url=settings.livekit_url, token=token, identity=identity,
        metadata=metadata, persona_label=persona_label,
    )


_TOKEN_TTL = timedelta(hours=1)   # well over a demo call's length


@app.post("/api/seed")
async def seed_persona(req: SeedRequest) -> dict:
    """Apply the persona's seed plan to Neo4j. Idempotent (wipes then re-seeds)."""
    if req.persona not in PERSONAS:
        raise HTTPException(400, f"unknown persona '{req.persona}'")
    p = PERSONAS[req.persona]
    plan = p["seed"]
    settings = get_settings()
    graph = CognitiveGraph(settings)
    try:
        await graph.connect()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Neo4j unreachable: {e}") from e
    try:
        await _wipe(graph, p["tenant_id"], p["user_id"])
        if plan == "anil_bharatpay":
            await _seed_anil_bharatpay(graph, p["tenant_id"], p["user_id"])
        # plan=="none" → wipe only
        return {"ok": True, "persona": req.persona,
                "tenant_id": p["tenant_id"], "user_id": p["user_id"], "plan": plan}
    finally:
        await graph.close()


@app.get("/api/verify", response_model=VerifyResponse)
async def verify(tenant_id: str, user_id: str) -> VerifyResponse:
    """Post-call inspection — what landed in the graph for this caller.

    The UI polls this after the user hangs up. The post-call worker is async (it can take
    up to a few seconds for facts + trust scores to appear), so the UI calls this on a
    short retry loop and re-renders as new fields populate.
    """
    settings = get_settings()
    graph = CognitiveGraph(settings)
    try:
        await graph.connect()
    except Exception as e:  # noqa: BLE001
        log.warning("verify.graph_unavailable", err=str(e))
        return VerifyResponse(available=False, user_id=user_id, tenant_id=tenant_id)

    try:
        ctx = await graph.get_user_context(tenant_id, user_id)

        # Compute what pre_call WOULD do for this user, end-to-end. This is the same
        # code path agent.py runs — guarantees the UI shows ground truth, not simulation.
        pctx = ParticipantContext(
            room="verify", participant_identity="verify",
            user_id=user_id, tenant_id=tenant_id,
        )
        pre = await build_precall_context(graph, pctx, settings)

        # Live graph reads for the post-call wow features.
        sarcastic = await _read_sarcastic(graph, user_id)
        decayed = await _read_decayed(graph, user_id)

        # Translate UserGraphContext.last_negative_event into a JSON-safe dict for the UI.
        last_event_dict = None
        if ctx.last_negative_event:
            last_event_dict = {
                "kind": ctx.last_negative_event.kind,
                "summary": ctx.last_negative_event.summary,
                "emotion": ctx.last_negative_event.emotion.value,
            }
        state_dict = None
        if ctx.affective_state:
            s = ctx.affective_state
            state_dict = {
                "emotion": s.emotion.value, "valence": s.valence,
                "arousal": s.arousal, "confidence": s.confidence,
                "paralinguistics": s.paralinguistics,
            }

        # Pass/fail flags the UI tile-renders.
        checks = {
            "dynamic_prosody": pre.voice_id == settings.elevenlabs_calm_voice_id
                               and abs(pre.voice_settings.stability - 0.95) < 1e-6,
            "adaptive_verbosity": ctx.conversation_style is ConversationStyle.IMPATIENT
                                  and "10 words or less" in pre.system_prompt,
            "proactive_empathy": ctx.last_negative_event is not None
                                 and pre.greeting is not None,
            "sarcasm_filter": len(sarcastic) > 0,
            "contradiction_decay": len(decayed) > 0,
        }
        return VerifyResponse(
            available=True, user_id=user_id, tenant_id=tenant_id,
            affective_state=state_dict,
            conversation_style=ctx.conversation_style.value,
            trusted_facts=ctx.trusted_facts,
            last_negative_event=last_event_dict,
            precall_voice_id=pre.voice_id,
            precall_voice_stability=pre.voice_settings.stability,
            precall_greeting=pre.greeting,
            precall_system_prompt=pre.system_prompt,
            sarcastic_facts=sarcastic,
            decayed_facts=decayed,
            checks=checks,
        )
    finally:
        await graph.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ── seed helpers ────────────────────────────────────────────────────────────


async def _wipe(graph: CognitiveGraph, tenant_id: str, user_id: str) -> None:
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        await s.run("MATCH (e:Entity {user_id:$u}) DETACH DELETE e", u=user_id)
        await s.run("MATCH (u:User {tenant_id:$t, id:$u}) DETACH DELETE u",
                    t=tenant_id, u=user_id)


async def _seed_anil_bharatpay(
    graph: CognitiveGraph, tenant_id: str, user_id: str
) -> None:
    """Anil Mehta, COO @ BharatPay Fintech — an at-risk enterprise account.

    Storyline (the agent can refer to ANY of these if asked):
      • Anil runs BharatPay, a high-volume customer-support shop on our voice-AI platform.
      • Three weeks ago a partial outage cost BharatPay ~₹2 cr in missed loan touches.
      • Anil escalated to enterprise support; promised service credits + a dedicated CSM.
      • The credits **still haven't posted** — that's the live grievance.
      • Priya is his CSM. He's now talking to **AlphaVoice**, a competitor → churn risk.
      • Voice prior: tense_suppressed (polite words, agitated under the hood).

    Demo levers we plant for the post-call wow features:
      • ``(Anil)-[:IS_OWED]->(Service_Credits_Q3)`` trust 0.7 — target for
        contradiction-decay (say "the credits actually posted last week" in the call).
      • ``(Anil)-[:IS_CONSIDERING]->(Competitor_AlphaVoice)`` trust 0.6 — churn risk.
      • An open invitation to be sarcastic about wait times, e.g. say
        "I love waiting three weeks for a billing fix" in an agitated tone → sarcasm
        filter pins the new ``(Anil)-[:LOVES]->(Waiting)`` rel to trust 0.2.
    """
    from memory.schemas import AffectiveState, NegativeEvent

    state = AffectiveState(
        tenant_id=tenant_id, user_id=user_id, emotion=Emotion.FRUSTRATED,
        valence=-0.6, arousal=0.75, confidence=0.85,
        paralinguistics={
            "base_acoustic_emotion": "angry",
            "final_affective_state": "tense_suppressed",
            "lexical_sentiment": "negative", "acoustic_affect": "agitated",
            "engine": "seed",
        },
    )
    event = NegativeEvent(
        kind="ServiceOutage",
        summary=("your service outage three weeks ago cost BharatPay an estimated "
                 "₹2 crore in missed loan-collection touches"),
        emotion=Emotion.FRUSTRATED, ts=time.time(),
    )
    await graph.record_affective_state(state, ConversationStyle.IMPATIENT,
                                       last_negative_event=event)
    await graph._commit_operations(   # noqa: SLF001
        user_id,
        [
            # Professional context — high-trust, surfaces as "Known about this caller".
            {"action": "ADD", "subject": "Anil", "predicate": "is COO of",
             "object": "BharatPay", "trust_score": 0.95,
             "reasoning": "seed: stable employment fact"},
            {"action": "ADD", "subject": "BharatPay", "predicate": "uses",
             "object": "Enterprise_Plan", "trust_score": 0.9,
             "reasoning": "seed: contract on record"},
            {"action": "ADD", "subject": "Anil", "predicate": "works with",
             "object": "CSM_Priya", "trust_score": 0.85,
             "reasoning": "seed: named dedicated CSM after escalation"},
            # Live grievance — trust ≥0.6 so it makes the top-facts cut. Contradicting it
            # in the call will decay this rel by 70% and add a fresh "received" rel.
            {"action": "ADD", "subject": "Anil", "predicate": "is owed",
             "object": "Service_Credits_Q3", "trust_score": 0.7,
             "reasoning": "seed: promised after the outage; not yet posted"},
            # Churn signal — moderate trust; the agent can probe it.
            {"action": "ADD", "subject": "Anil", "predicate": "is considering",
             "object": "Competitor_AlphaVoice", "trust_score": 0.6,
             "reasoning": "seed: competitive intel; churn risk"},
        ],
        affective_state="tense_suppressed",
    )


# ── verify helpers ──────────────────────────────────────────────────────────


async def _read_sarcastic(graph: CognitiveGraph, user_id: str) -> list[dict]:
    out: list[dict] = []
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        result = await s.run(
            "MATCH (sub:Entity {user_id:$u})-[r]->(obj:Entity {user_id:$u}) "
            "WHERE coalesce(r.reasoning,'') CONTAINS 'Likely Sarcastic' "
            "RETURN sub.id AS s, coalesce(r.predicate_raw,'') AS p, obj.id AS o, "
            "r.trust_score AS trust, coalesce(r.reasoning,'') AS reason",
            u=user_id,
        )
        async for row in result:
            out.append({"subject": row["s"], "predicate": row["p"], "object": row["o"],
                        "trust_score": float(row["trust"]), "reasoning": row["reason"]})
    return out


async def _read_decayed(graph: CognitiveGraph, user_id: str) -> list[dict]:
    out: list[dict] = []
    async with graph._driver_or_raise().session() as s:   # noqa: SLF001
        result = await s.run(
            "MATCH (sub:Entity {user_id:$u})-[r]->(obj:Entity {user_id:$u}) "
            "WHERE r.superseded_at IS NOT NULL "
            "RETURN sub.id AS s, coalesce(r.predicate_raw,'') AS p, obj.id AS o, "
            "r.trust_score AS trust, r.superseded_at AS sup",
            u=user_id,
        )
        async for row in result:
            out.append({"subject": row["s"], "predicate": row["p"], "object": row["o"],
                        "trust_score": float(row["trust"]), "superseded_at": row["sup"]})
    return out


# Keep an explicit reference to ``to_elevenlabs_voice`` so a future maintainer reading the
# imports sees that pre_call's voice decision is exposed to /api/verify (the helper is
# called transitively via build_precall_context).
_ = _PROFILES, to_elevenlabs_voice, asyncio


# ── dev entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.web_host, port=settings.web_port,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
