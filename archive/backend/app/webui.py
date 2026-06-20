"""Ringg-styled test console — a small dashboard to exercise the prototype in a browser.

It is a thin layer over the REAL system: each chat turn runs the pre-call prosody hook +
the LLM + TTS exactly as a call would; "Analyze" runs the same contradiction engine the
post-call worker uses. With no API keys it transparently uses the mock providers, so the
UI is fully demoable offline (replies are templated, audio is a tone) — and lights up with
real voices/answers the moment you add keys.

Endpoints (all under the FastAPI app):
    GET  /                 → the dashboard SPA
    GET  /api/config       → engine + which providers are live vs mock
    POST /api/seed-emotion → set the caller's stored affect (drives pre-call prosody)
    POST /api/chat         → one turn: prosody → LLM → TTS; returns reply text + audio
    POST /api/analyze      → contradiction engine over the transcript + current affect
    POST /api/reset        → clear a session
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import get_settings
from app.engines.base import CallContext
from app.memory.contradiction import ContradictionEngine, extract_assertions
from app.memory.precall import resolve_prosody
from app.memory.schemas import AffectiveState, Emotion
from app.memory.store import affective_store
from app.metrics import TurnTimer
from app.providers.factory import make_llm, make_tts
from app.telephony.audio import ulaw_to_wav

router = APIRouter()
_INDEX = Path(__file__).parent / "static" / "index.html"

# In-memory per-browser-session transcript store (prototype; Redis in production).
_SESSIONS: dict[str, dict] = {}
_TENANT = "demo"


# ── request models ───────────────────────────────────────────────────────────


class ChatIn(BaseModel):
    session_id: str
    user_id: str = "web-caller"
    engine: str = "cascade"
    text: str


class EmotionIn(BaseModel):
    session_id: str
    emotion: Emotion


class SessionIn(BaseModel):
    session_id: str
    user_id: str = "web-caller"


# ── routes ───────────────────────────────────────────────────────────────────


@router.get("/")
async def index() -> FileResponse:
    return FileResponse(_INDEX)


@router.get("/api/config")
async def config() -> dict:
    s = get_settings()
    # Build the providers to report their *actual* effective names (mock vs live).
    from app.providers.factory import make_stt

    stt, llm, tts = make_stt(s).name, make_llm(s).name, make_tts(s).name
    return {
        "engine": s.engine,
        "providers": {
            "stt": {"name": stt, "live": stt != "mock"},
            "llm": {"name": llm, "model": s.llm_model, "live": llm != "mock"},
            "tts": {"name": tts, "live": tts != "mock"},
        },
        "emotions": [e.value for e in Emotion],
        "engines": ["cascade", "realtime"],
        "latency_note": "Honest target ~800ms p50 (cascade). Numbers below are live-measured.",
    }


@router.post("/api/seed-emotion")
async def seed_emotion(body: EmotionIn) -> dict:
    """Pretend a previous call left the caller in this emotional state, so the NEXT turn's
    pre-call prosody adapts. This is the affective feedback loop, made interactive."""
    valence, arousal = _EMOTION_VA.get(body.emotion, (0.0, 0.3))
    # Memory is keyed by session_id so each browser test session is isolated and starts
    # clean — no cross-contamination between demos.
    await affective_store(get_settings()).upsert_state(
        AffectiveState(tenant_id=_TENANT, user_id=body.session_id, emotion=body.emotion,
                       valence=valence, arousal=arousal, confidence=0.9)
    )
    # Report the prosody that will now apply.
    ctx = CallContext(tenant_id=_TENANT, user_id=body.session_id)
    await resolve_prosody(ctx, get_settings())
    return {"ok": True, "prosody": ctx.prosody.label, "system_prompt": ctx.system_prompt}


@router.post("/api/chat")
async def chat(body: ChatIn) -> dict:
    s = get_settings()
    sess = _SESSIONS.setdefault(body.session_id, {"history": []})

    # Pre-call prosody (Module 3a) — runs every turn so a seeded emotion takes effect.
    # user_id == session_id keeps each test session's memory isolated.
    ctx = CallContext(call_sid=body.session_id, tenant_id=_TENANT, user_id=body.session_id,
                      engine=body.engine)
    await resolve_prosody(ctx, s)

    history: list[dict] = sess["history"]
    if not history:
        history.append({"role": "system", "content": ctx.system_prompt or s.base_system_prompt})
    history.append({"role": "user", "content": body.text})

    timer = TurnTimer()
    started = time.monotonic()
    llm, tts = make_llm(s), make_tts(s)

    parts: list[str] = []
    async for sentence in llm.stream_sentences(history, timer):
        parts.append(sentence)
    reply = " ".join(parts).strip() or "..."
    history.append({"role": "assistant", "content": reply})

    ulaw = bytearray()
    async for chunk in tts.synthesize(reply, ctx.prosody, timer):
        ulaw.extend(chunk)
    timer.mark_first_audio_out()
    total_ms = round((time.monotonic() - started) * 1000, 1)

    wav_b64 = base64.b64encode(ulaw_to_wav(bytes(ulaw))).decode() if ulaw else ""
    return {
        "reply": reply,
        "audio": wav_b64,
        "prosody": ctx.prosody.label,
        "latency": {**timer.summary(), "total_ms": total_ms},
    }


@router.post("/api/analyze")
async def analyze(body: SessionIn) -> dict:
    """Run the same Contradiction Engine the post-call worker uses, over the transcript.

    Only facts not yet processed in this session are reconciled (so re-clicking Analyze is
    idempotent), and decisions accumulate — mirroring a post-call worker that runs on new
    information rather than re-litigating the whole transcript each time.
    """
    s = get_settings()
    sess = _SESSIONS.setdefault(body.session_id, {"history": []})
    seen: set = sess.setdefault("seen", set())
    decisions_acc: list = sess.setdefault("decisions", [])

    new = []
    for a in extract_assertions(sess.get("history", []), s):
        key = (a.subject, a.value, a.negated, a.text)
        if key not in seen:
            seen.add(key)
            new.append(a)
    if new:
        ds = await ContradictionEngine(s).reconcile(
            new, tenant_id=_TENANT, user_id=body.session_id, call_sid=body.session_id
        )
        decisions_acc.extend(ds)

    state = await affective_store(s).get_state(_TENANT, body.session_id)
    return {
        "decisions": [d.model_dump() for d in decisions_acc],
        "ops": _op_counts(decisions_acc),
        "affective_state": state.model_dump() if state else None,
    }


@router.post("/api/reset")
async def reset(body: SessionIn) -> dict:
    _SESSIONS.pop(body.session_id, None)
    return {"ok": True}


# ── helpers ──────────────────────────────────────────────────────────────────

_EMOTION_VA = {
    Emotion.FRUSTRATED: (-0.6, 0.7),
    Emotion.ANGRY: (-0.8, 0.85),
    Emotion.ANXIOUS: (-0.3, 0.6),
    Emotion.HAPPY: (0.7, 0.6),
    Emotion.SAD: (-0.5, 0.2),
    Emotion.NEUTRAL: (0.0, 0.3),
}


def _op_counts(decisions) -> dict:  # noqa: ANN001
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.op.value] = counts.get(d.op.value, 0) + 1
    return counts
